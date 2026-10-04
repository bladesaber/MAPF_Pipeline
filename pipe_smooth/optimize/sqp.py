"""SQP hard phase on packed Hermite DOF (sequential convex + OSQP).

Problem definition (Stage-1 Phase-3; detection outside AD — freeze active sets
via refresh_clear / refresh_union, same pattern as adam/langevin)
================================================================

Decision variable
    x  = packed free Hermite DOF (see pack.py): free node.pos and node.v.

Objective (soft; minimize)
    J(x) = α L + β K + λ J_union
    - L, K     : §6.4.1 path chord energy ΣΔs² / curvature (graph_length, graph_curvature)
    - J_union  : §6.4.3 Chamfer on frozen union hits (union_cost)
    Clearance is NOT in J here (moved to constraints). η J_space / μ_v stay out
    of the SQP objective — they become inequalities below (§7.1).

Main constraints (hard clearance; g_i ≤ 0)
    For each frozen active pair / hit i (capsule or hull):
        g_clear,i(x) = d_req,i - d_geom,i(x)  ≤  0
    d_geom recomputed from live Hermite + frozen witnesses/weights (§7.2).
    Refresh active set every k outer SQP rounds (sequential convexification).

Auxiliary constraints (§7.1)
    (1) Sample-spacing inequality band (§7.1.1 (2))
            (1-ε) mean(Δs)  ≤  Δs_i  ≤  (1+ε) mean(Δs)
        Residuals g≤0 (graph_space_band_constraint): upper always; lower optional
        (upper_only first if feasibility is tight). Linearize mean at iterate
        or use L/(N-1) form as in ROADMAP.

    (2) Tangent-norm bounds (§7.1.2 / graph_tangent_constraint)
            ||v0|| ≤ β_v ||p1-p0|| ,  ||v1|| ≤ β_v ||p1-p0||
        per Hermite end from hermite_ends (ports already clamped in pack).
        Residuals: ||v|| - β_v ||Δp|| <= 0.

Not constraints (by construction / soft-only)
    Ports, trunk share, interface C0/C1 — eliminated in pack.
    Soft κ preference — only via β K in J, never hard κ_max.

Solver (this file)
    Each outer step: ∇J + linearize g(x)≤0 → OSQP elastic QP
        min  ½||Δx||² + (∇J)ᵀΔx + w 1ᵀs
        s.t. g + Jg Δx - s ≤ 0,  s ≥ 0,  ||Δx||_∞ ≤ δ
    Accept/reject via ρ=ared/pred; adapt δ (real trust-region).
    Large-scale interior-point NLP: see ipopt.py (same problem).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import osqp
from scipy import sparse

from pipe_smooth.costs.clear import (
    clear_capsule_constraint_pack,
    clear_hull_constraint_pack,
    pack_capsule_active,
    pack_hull_active,
)
from pipe_smooth.costs.space import graph_space_band_constraint, graph_tangent_constraint
from pipe_smooth.costs.union import pack_union_hits
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.optimize.loss import LossWeights, loss_and_grad, total_loss
from pipe_smooth.optimize.pack import PackLayout, pack, unpack, refresh_from_x
from pipe_smooth.optimize.recorder import Recorder
from pipe_smooth.utils import backend as _backend


def constraints(
    graphs: Sequence[OptimizeGraph],
    *,
    x=None,
    layout: PackLayout | None = None,
    clear_obs_buf=None,
    clear_grp_buf=None,
    clear_kind: Literal["capsule", "hull"] = "capsule",
    eps_band: float = 0.2,
    upper_only: bool = True,
    beta_v: float = 0.4,
    use_space: bool = True,
    use_tangent: bool = True,
    samples=None,
    backend: str = "jax",
):
    """SQP inequality stack (§7.1–7.2). Feasible iff all residuals ``g(x) <= 0``.

    Flat inequality vector ``g(x)≤0`` = packed clearance (+ optional space / tangent).

    Args:
        graphs: OptimizeGraph sequence (``refresh`` before space / tangent blocks)
        x: packed free DOF; required with layout/bufs for clearance
        layout: ``PackLayout`` for ``ends_from_x`` gather
        clear_obs_buf, clear_grp_buf: frozen packed clear buffers from
            ``pack_capsule_active`` / ``pack_hull_active``, or ``None`` (no clearance)
        clear_kind: ``'capsule'`` | ``'hull'`` → ``clear_*_constraint_pack``
        eps_band: relative Δs band width ε (§7.1.1)
        upper_only: if True, only spacing upper bound
        beta_v: Hermite tangent scale vs chord (§7.1.2)
        use_space: append ``graph_space_band_constraint`` blocks
        use_tangent: append ``graph_tangent_constraint`` blocks
        backend: ``'numpy'`` | ``'jax'``

    Returns:
        Flat inequality vector ``g(x)≤0`` (empty if no active blocks).
    """
    xp = _backend(backend)
    blocks = []

    if clear_obs_buf is not None and clear_grp_buf is not None and layout is not None and x is not None:
        if clear_kind == "capsule":
            blocks.append(
                clear_capsule_constraint_pack(
                    x, clear_obs_buf, clear_grp_buf, layout, backend=backend
                )
            )
        else:
            blocks.append(
                clear_hull_constraint_pack(
                    x, clear_obs_buf, clear_grp_buf, layout, backend=backend
                )
            )

    if use_space:
        for i, g in enumerate(graphs):
            xyz = None if samples is None else samples[i][1]
            parts = graph_space_band_constraint(
                g, eps=eps_band, upper_only=upper_only, backend=backend, xyz=xyz
            )
            if parts.size > 0:
                blocks.append(parts)

    if use_tangent:
        for i, g in enumerate(graphs):
            ends = None if samples is None else samples[i][0]
            parts = graph_tangent_constraint(g, beta=beta_v, backend=backend, ends=ends)
            if parts.size > 0:
                blocks.append(parts)

    blocks = [b for b in blocks if b.size > 0]
    if not blocks:
        return xp.zeros((0,), dtype=jnp.float32)
    return xp.concatenate(blocks, axis=0)


def c_and_jac(c_fn, x):
    """One primal: constraint vector ``c`` and dense Jacobian ``Jc`` via VJP.

    ``c_fn(x) -> (m,)``, returns ``(c, Jc)`` with ``Jc.shape == (m, n)``.
    """
    xx = jnp.asarray(x, dtype=jnp.float32)
    c, vjp = jax.vjp(c_fn, xx)
    c = jnp.reshape(c, (-1,))
    n = int(xx.size)
    if c.size == 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, n), dtype=np.float64),
        )
    Jc = jax.vmap(lambda e: vjp(e)[0])(jnp.eye(c.size, dtype=c.dtype))
    return (
        np.asarray(c, dtype=np.float64).reshape(-1),
        np.asarray(Jc, dtype=np.float64).reshape(c.size, n),
    )


def active_g(c, clear_obs_buf=None, clear_grp_buf=None, Jc=None, *, n: int | None = None):
    """Drop padded clear rows (``mask==0``); keep space/tangent tail.

    JAX/pack stay full ``p_max``; call this after numpy conversion before OSQP/IPOPT.
    ``Jc`` may be ``(m, n)`` or flat ``(m*n,)``; returned ``Jc`` is always ``(m_active, n)``.
    Pass ``n`` (DOF) when ``Jc`` is flat; for 2D ``Jc`` it defaults to ``Jc.shape[1]``.
    """
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    m_full = int(c.size)
    keep = None
    if clear_obs_buf is not None and clear_grp_buf is not None and m_full:
        mask = np.concatenate(
            [
                np.asarray(clear_obs_buf["mask"], dtype=np.float64).reshape(-1),
                np.asarray(clear_grp_buf["mask"], dtype=np.float64).reshape(-1),
            ]
        )
        keep = np.ones(m_full, dtype=bool)
        keep[: mask.size] = mask > 0.5
        c = c[keep]
    if Jc is None:
        return c
    Jc = np.asarray(Jc, dtype=np.float64)
    if Jc.ndim == 1:
        if n is None:
            raise ValueError("active_g: pass n=... when Jc is flat")
        Jc = Jc.reshape(m_full, int(n)) if m_full else np.zeros((0, int(n)), dtype=np.float64)
    elif n is not None and int(Jc.shape[1]) != int(n):
        raise ValueError(f"active_g: Jc.shape[1]={Jc.shape[1]} != n={n}")
    if keep is not None:
        Jc = Jc[keep]
    return c, Jc


def elastic_qp(
    graphs: Sequence[OptimizeGraph],
    layout: PackLayout,
    weights: LossWeights,
    *,
    steps: int = 20,
    trust: float = 0.1,
    eps_band: float = 0.2,
    upper_only: bool = True,
    beta_v: float = 0.4,
    w_slack: float = 1e3,
    clear_kind: str = "capsule",
    refresh_clear: Callable[[Sequence[OptimizeGraph]], Any] | None = None,
    refresh_union: Callable[[Sequence[OptimizeGraph]], dict] | None = None,
    refresh_every: int = 5,
    recorder: Recorder,
    verbose: bool = False,
) -> None:
    """Elastic QP hard phase (§7 / OSQP sequential convex).

    Each outer step linearizes soft ``J`` and hard ``g(x)≤0``, then solves an
    **Quadratic programming** in OSQP form

        min_{z}  ½ zᵀ P z + qᵀ z
        s.t.     l ≤ A z ≤ u

    Trust region (∞-norm): accept/reject via ``ρ = ared/pred`` on soft ``J``
    with model ``m(d)=J+gᵀd+½‖d‖²``; shrink/expand ``delta`` from ``ρ``.

    Args:
        graphs: OptimizeGraph sequence (packed / refreshed in-place)
        layout: PackLayout for free Hermite DOF
        weights: LossWeights; η/μ_v > 0 enable space / tangent blocks; μ_obs/μ_grp ignored here
        steps: outer sequential-convex iterations
        trust: initial ∞-norm trust radius on ``Δx`` (adapted online)
        eps_band: spacing band ε passed to ``constraints``
        upper_only: spacing upper-only flag
        beta_v: tangent-norm scale vs chord
        w_slack: elastic slack weight in ``w 1ᵀs`` (larger → harder g≤0)
        clear_kind: ``'capsule'`` | ``'hull'`` clearance residual
        refresh_clear: callback → frozen clear active set (outside AD)
        refresh_union: callback → frozen union hits for soft ``J_union``
        refresh_every: refresh period in outer steps
        recorder: history / best-x logger
        verbose: print OSQP status / ``J``
    """
    if not recorder.phase:
        recorder.phase = "elastic_qp"

    w = replace(weights, mu_obs=0.0, mu_grp=0.0, eta=0.0, mu_v=0.0)
    x = jnp.asarray(pack(graphs, layout, backend="numpy"), dtype=jnp.float32)
    clear_obs_buf = clear_grp_buf = None
    union_hits = None
    n = int(layout.n)
    delta = float(trust)
    delta_min, delta_max = 1e-4, max(1.0, delta)

    for t in range(1, steps + 1):
        if (refresh_clear is not None or refresh_union is not None) and (
            t == 1 or (t - 1) % refresh_every == 0
        ):
            unpack(np.asarray(x), graphs, layout, backend="numpy")
            for g in graphs:
                g.refresh(backend="numpy", store=True)
            if refresh_clear is not None:
                active = refresh_clear(graphs)
                if clear_kind == "capsule":
                    clear_obs_buf, clear_grp_buf = pack_capsule_active(
                        active, layout, backend="jax"
                    )
                else:
                    clear_obs_buf, clear_grp_buf = pack_hull_active(
                        active, layout, backend="jax"
                    )
            if refresh_union is not None:
                union_hits = refresh_union(graphs)

        val, grad, parts = loss_and_grad(
            x, graphs, layout, w, clear_active=None, union_hits=union_hits
        )

        def c_fn(xx):
            samples = refresh_from_x(xx, graphs, layout, backend="jax")
            return constraints(
                graphs,
                x=xx,
                layout=layout,
                clear_obs_buf=clear_obs_buf,
                clear_grp_buf=clear_grp_buf,
                clear_kind=clear_kind,  # type: ignore[arg-type]
                eps_band=eps_band,
                upper_only=upper_only,
                beta_v=beta_v,
                use_space=weights.eta > 0.0,
                use_tangent=weights.mu_v > 0.0,
                samples=samples,
                backend="jax",
            )

        c, Jc = c_and_jac(c_fn, x)
        c, Jc = active_g(c, clear_obs_buf, clear_grp_buf, Jc, n=n)
        
        # ------ deprecated code (use c_and_jac instead) ------
        # c = np.asarray(c_fn(x), dtype=np.float64).reshape(-1)
        # if c.size:
        #     Jc = np.asarray(jax.jacobian(c_fn)(x), dtype=np.float64)
        # else:
        #     Jc = np.zeros((0, n), dtype=np.float64)
        # ------------------------------

        # Elastic QP: min ½||Δx||² + gradᵀΔx + w 1ᵀs
        #   g + Jg Δx - s ≤ 0,  s≥0,  |Δx|≤delta
        m = c.size
        if t == 1 or t == steps or t % refresh_every == 0:
            val_f = float(val)
            max_viol = float(np.max(c)) if m else 0.0
            recorder.record(
                t, val_f, x, n_cons=m, max_viol=max_viol,
                **{k: float(v) for k, v in parts.items()},  # type: ignore[arg-type]
            )

        P = sparse.block_diag(
            [sparse.eye(n, format="csc"), sparse.csc_matrix((m, m))], format="csc"
        )
        gvec = np.asarray(grad, dtype=np.float64).reshape(-1)
        q = np.concatenate([gvec, w_slack * np.ones(m)])
        if m:
            A = sparse.vstack(
                [
                    sparse.hstack(
                        [sparse.csc_matrix(Jc), -sparse.eye(m, format="csc")]
                    ),
                    sparse.hstack([sparse.eye(n, format="csc"), sparse.csc_matrix((n, m))]),
                    sparse.hstack([sparse.csc_matrix((m, n)), sparse.eye(m, format="csc")]),
                ],
                format="csc",
            )
            l = np.concatenate([np.full(m, -np.inf), -delta * np.ones(n), np.zeros(m)])
            u = np.concatenate([-c, delta * np.ones(n), np.full(m, np.inf)])
        else:
            A = sparse.eye(n, format="csc")
            l = -delta * np.ones(n)
            u = delta * np.ones(n)
            P = sparse.eye(n, format="csc")
            q = gvec

        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=True)
        res = prob.solve()
        ok = res.info.status_val in (1, 2)  # solved / solved inaccurate
        if not ok or res.x is None:
            delta = max(delta_min, 0.25 * delta)
            if verbose:
                print(f"elastic_qp {t}/{steps}  OSQP={res.info.status}  shrink δ={delta:.3g}")
            if delta <= delta_min:
                break
            continue

        dx = np.asarray(res.x[:n], dtype=np.float64).reshape(-1)
        # model m(d)=J+gᵀd+½‖d‖²  →  pred = m(0)-m(d)
        pred = float(-(gvec @ dx + 0.5 * (dx @ dx)))
        if pred <= 1e-16:
            delta = max(delta_min, 0.25 * delta)
            continue

        x_trial = x + jnp.asarray(dx, dtype=x.dtype)
        samples = refresh_from_x(x_trial, graphs, layout, backend="jax")
        uni = pack_union_hits(union_hits, layout, backend="jax")
        val_new, _ = total_loss(
            graphs, w, layout=layout, x=x_trial, union_buf=uni, samples=samples, backend="jax"
        )
        ared = float(val) - float(val_new)
        rho = ared / pred

        if rho < 0.25:
            delta = max(delta_min, 0.25 * delta)
        elif rho > 0.75 and float(np.linalg.norm(dx, ord=np.inf)) >= 0.9 * delta:
            delta = min(delta_max, 2.0 * delta)

        accepted = rho > 0.1
        if accepted:
            x = x_trial

        if verbose and (t == 1 or t == steps or t % refresh_every == 0):
            print(
                f"elastic_qp {t}/{steps}  J={float(val):.6g}  ρ={rho:.3g}  "
                f"δ={delta:.3g}  {'ok' if accepted else 'rej'}  m={m}"
            )

    if recorder.best_x is None:
        raise RuntimeError("recorder.best_x is None; no step recorded a state")
    unpack(recorder.best_x, graphs, layout, backend="numpy")
    for g in graphs:
        g.refresh(backend="numpy", store=True)
