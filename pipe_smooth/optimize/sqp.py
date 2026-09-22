"""SQP hard phase on packed Hermite DOF (sequential convex + OSQP).

Problem definition (Stage-1 Phase-3; detection outside AD — freeze active sets
via refresh_clear / refresh_union, same pattern as adam/langevin)
================================================================

Decision variable
    x  = packed free Hermite DOF (see pack.py): free node.pos and node.v.

Objective (soft; minimize)
    J(x) = α L + β K + λ J_union
    - L, K     : §6.4.1 path length / curvature (graph_length, graph_curvature)
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
        s.t. g + Jg Δx - s ≤ 0,  s ≥ 0,  ||Δx||_∞ ≤ trust
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
    clear_capsule_active_constraint,
    clear_hull_active_constraint,
)
from pipe_smooth.costs.space import graph_space_band_constraint, graph_tangent_constraint
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.optimize.loss import LossWeights, loss_and_grad
from pipe_smooth.optimize.pack import PackLayout, pack, unpack
from pipe_smooth.optimize.recorder import Recorder
from pipe_smooth.utils import backend as _backend


def constraints(
    graphs: Sequence[OptimizeGraph],
    *,
    clear_active=None,
    clear_kind: Literal["capsule", "hull"] = "capsule",
    eps_band: float = 0.2,
    upper_only: bool = True,
    beta_v: float = 0.4,
    use_space: bool = True,
    use_tangent: bool = True,
    backend: str = "jax",
):
    """SQP inequality stack (§7.1–7.2). Feasible iff all residuals ``g(x) <= 0``.

    Flat inequality vector ``g(x)≤0`` = clearance (+ optional space / tangent).

    Args:
        graphs: OptimizeGraph sequence (``hermite_ends`` / ``refresh``)
        clear_active: frozen narrowphase pairs, or ``None`` (no clearance block)
        clear_kind: ``'capsule'`` | ``'hull'`` active-set residual
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

    if clear_active is not None:
        if clear_kind == "capsule":
            blocks.append(clear_capsule_active_constraint(clear_active, graphs, backend=backend))
        else:
            blocks.append(clear_hull_active_constraint(clear_active, graphs, backend=backend))

    if use_space:
        for g in graphs:
            parts = graph_space_band_constraint(
                g, eps=eps_band, upper_only=upper_only, backend=backend
            )
            if parts.size > 0:
                blocks.append(parts)

    if use_tangent:
        for g in graphs:
            parts = graph_tangent_constraint(g, beta=beta_v, backend=backend)
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
    recorder: Recorder | None = None,
    verbose: bool = False,
):
    """Elastic QP hard phase (§7 / OSQP sequential convex).

    Each outer step linearizes soft ``J`` and hard ``g(x)≤0``, then solves an
    **Quadratic programming** in OSQP form

        min_{z}  ½ zᵀ P z + qᵀ z
        s.t.     l ≤ A z ≤ u

    Args:
        graphs: OptimizeGraph sequence (packed / refreshed in-place)
        layout: PackLayout for free Hermite DOF
        weights: LossWeights; η/μ_v > 0 enable space / tangent blocks; μ_obs/μ_grp ignored here
        steps: outer sequential-convex iterations
        trust: ∞-norm trust on ``Δx``
        eps_band: spacing band ε passed to ``constraints``
        upper_only: spacing upper-only flag
        beta_v: tangent-norm scale vs chord
        w_slack: elastic slack weight in ``w 1ᵀs`` (larger → harder g≤0)
        clear_kind: ``'capsule'`` | ``'hull'`` clearance residual
        refresh_clear: callback → frozen clear active set (outside AD)
        refresh_union: callback → frozen union hits for soft ``J_union``
        refresh_every: refresh period in outer steps
        recorder: optional history / best-x logger
        verbose: print OSQP status / ``J``

    Returns:
        ``(x, history)`` packed DOF and per-step soft ``J`` values.
    """
    if recorder is not None and not recorder.phase:
        recorder.phase = "elastic_qp"

    w = replace(weights, mu_obs=0.0, mu_grp=0.0, eta=0.0, mu_v=0.0)
    x = jnp.asarray(pack(graphs, layout, backend="numpy"), dtype=jnp.float32)
    history: list[float] = []
    clear_active = None
    union_hits = None
    n = int(layout.n)

    for t in range(1, steps + 1):
        if (refresh_clear is not None or refresh_union is not None) and (
            t == 1 or (t - 1) % refresh_every == 0
        ):
            unpack(np.asarray(x), graphs, layout, backend="numpy")
            for g in graphs:
                g.refresh(backend="numpy")
            if refresh_clear is not None:
                clear_active = refresh_clear(graphs)
            if refresh_union is not None:
                union_hits = refresh_union(graphs)

        val, grad = loss_and_grad(
            x, graphs, layout, w, clear_active=None, union_hits=union_hits
        )
        val_f = float(val)
        history.append(val_f)

        def c_fn(xx):
            unpack(xx, graphs, layout, backend="jax")
            for gr in graphs:
                gr.refresh(backend="jax")
            return constraints(
                graphs,
                clear_active=clear_active,
                clear_kind=clear_kind,  # type: ignore[arg-type]
                eps_band=eps_band,
                upper_only=upper_only,
                beta_v=beta_v,
                use_space=weights.eta > 0.0,
                use_tangent=weights.mu_v > 0.0,
                backend="jax",
            )

        c, Jc = c_and_jac(c_fn, x)
        
        # ------ deprecated code (use c_and_jac instead) ------
        # c = np.asarray(c_fn(x), dtype=np.float64).reshape(-1)
        # if c.size:
        #     Jc = np.asarray(jax.jacobian(c_fn)(x), dtype=np.float64)
        # else:
        #     Jc = np.zeros((0, n), dtype=np.float64)
        # ------------------------------

        # Elastic QP: min ½||Δx||² + gradᵀΔx + w 1ᵀs
        #   g + Jg Δx - s ≤ 0,  s≥0,  |Δx|≤trust  (always feasible)
        m = c.size
        if recorder is not None:
            max_viol = float(np.max(c)) if m else 0.0
            recorder.record(t, val_f, x, n_cons=m, max_viol=max_viol)

        P = sparse.block_diag(
            [sparse.eye(n, format="csc"), sparse.csc_matrix((m, m))], format="csc"
        )
        q = np.concatenate(
            [np.asarray(grad, dtype=np.float64).reshape(-1), w_slack * np.ones(m)]
        )
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
            l = np.concatenate([np.full(m, -np.inf), -trust * np.ones(n), np.zeros(m)])
            u = np.concatenate([-c, trust * np.ones(n), np.full(m, np.inf)])
        else:
            A = sparse.eye(n, format="csc")
            l = -trust * np.ones(n)
            u = trust * np.ones(n)
            P = sparse.eye(n, format="csc")
            q = np.asarray(grad, dtype=np.float64).reshape(-1)

        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=True)
        res = prob.solve()
        ok = res.info.status_val in (1, 2)  # solved / solved inaccurate
        if not ok or res.x is None:
            if verbose:
                print(f"elastic_qp {t}/{steps}  OSQP status={res.info.status} — stop")
            break
        x = x + jnp.asarray(res.x[:n], dtype=x.dtype)

        if verbose and (t == 1 or t % max(1, steps // 5) == 0 or t == steps):
            print(f"elastic_qp {t}/{steps}  J={history[-1]:.6g}  m={m}  status={res.info.status}")

    unpack(np.asarray(x), graphs, layout, backend="numpy")
    for g in graphs:
        g.refresh(backend="numpy")
    return x, history
