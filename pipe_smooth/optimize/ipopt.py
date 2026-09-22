"""IPOPT hard phase on packed Hermite DOF (cyipopt interior-point NLP).

Same NLP as ``sqp.py`` / ``elastic_qp``; different solver backend.
Detection stays outside AD — freeze active sets via refresh_clear / refresh_union.

Problem definition (Stage-1 Phase-3)
====================================

Decision variable
    x  = packed free Hermite DOF (see pack.py): free node.pos and node.v.

Objective (soft; minimize)
    J(x) = α L + β K + λ J_union
    - L, K     : §6.4.1 path length / curvature (graph_length, graph_curvature)
    - J_union  : §6.4.3 Chamfer on frozen union hits (union_cost)
    Clearance is NOT in J (moved to constraints). η J_space / μ_v are inequalities
    below (§7.1), not terms in J.

Main constraints (hard clearance; g_i ≤ 0)
    For each frozen active pair / hit i (capsule or hull):
        g_clear,i(x) = d_req,i - d_geom,i(x)  ≤  0
    d_geom from live Hermite + frozen witnesses/weights (§7.2).
    Refresh active set every k outer rounds (rebuild NLP / warm-start).

Auxiliary constraints (§7.1)
    (1) Sample-spacing inequality band (§7.1.1 (2))
            (1-ε) mean(Δs)  ≤  Δs_i  ≤  (1+ε) mean(Δs)
        Residuals g≤0; prefer upper-only first if feasibility is tight.

    (2) Tangent-norm bounds (§7.1.2)
            ||v|| - β_v ||p1-p0|| ≤ 0
        per Hermite end (hermite_ends); ports already out of x via pack.

Not constraints (by construction / soft-only)
    Ports, trunk share, interface C0/C1 — eliminated in pack.
    Soft κ — only via β K in J, never hard κ_max.

Solver (this file)
    cyipopt Problem + JAX ``loss_and_grad`` / ``constraints`` (dense Jg for MVP).
    Interior-point NLP; entry point ``interior_nlp`` (avoids clashing with ``cyipopt``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import cyipopt
import jax
import jax.numpy as jnp
import numpy as np

from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.optimize.loss import LossWeights, loss_and_grad
from pipe_smooth.optimize.pack import PackLayout, pack, unpack
from pipe_smooth.optimize.recorder import Recorder
from pipe_smooth.optimize.sqp import constraints


def interior_nlp(
    graphs: Sequence[OptimizeGraph],
    layout: PackLayout,
    weights: LossWeights,
    *,
    steps: int = 5,
    max_iter: int = 5,
    eps_band: float = 0.2,
    upper_only: bool = True,
    beta_v: float = 0.4,
    clear_kind: str = "capsule",
    refresh_clear: Callable[[Sequence[OptimizeGraph]], Any] | None = None,
    refresh_union: Callable[[Sequence[OptimizeGraph]], dict] | None = None,
    refresh_every: int = 1,
    recorder: Recorder,
    verbose: bool = False,
) -> None:
    """Interior-point NLP hard phase (§7 / cyipopt).

    Same soft ``J`` + flat ``g(x)≤0`` as ``elastic_qp``; IPOPT solves the NLP
    with frozen active sets between refreshes.

    Args:
        graphs: OptimizeGraph sequence (packed / refreshed in-place)
        layout: PackLayout for free Hermite DOF
        weights: LossWeights; η/μ_v > 0 enable space / tangent; μ_obs/μ_grp ignored
        steps: outer refresh / re-solve rounds
        max_iter: IPOPT iterations per outer round
        eps_band: spacing band ε passed to ``constraints``
        upper_only: spacing upper-only flag
        beta_v: tangent-norm scale vs chord
        clear_kind: ``'capsule'`` | ``'hull'`` clearance residual
        refresh_clear: callback → frozen clear active set (outside AD)
        refresh_union: callback → frozen union hits for soft ``J_union``
        refresh_every: refresh period in outer steps
        recorder: history / best-x logger (one record per outer round)
        verbose: IPOPT print_level / outer ``J``
    """
    if not recorder.phase:
        recorder.phase = "interior_nlp"

    w = replace(weights, mu_obs=0.0, mu_grp=0.0, eta=0.0, mu_v=0.0)
    x = np.asarray(pack(graphs, layout, backend="numpy"), dtype=np.float64)
    clear_active = None
    union_hits = None
    n = int(layout.n)

    for t in range(1, steps + 1):
        if (refresh_clear is not None or refresh_union is not None) and (
            t == 1 or (t - 1) % refresh_every == 0
        ):
            unpack(x, graphs, layout, backend="numpy")
            for g in graphs:
                g.refresh(backend="numpy")
            if refresh_clear is not None:
                clear_active = refresh_clear(graphs)
            if refresh_union is not None:
                union_hits = refresh_union(graphs)

        def _obj_grad(xx):
            xx_j = jnp.asarray(xx, dtype=jnp.float32)
            val, grad = loss_and_grad(
                xx_j, graphs, layout, w, clear_active=None, union_hits=union_hits
            )
            return float(val), np.asarray(grad, dtype=np.float64).reshape(-1)

        def _c_fn(xx):
            xx_j = jnp.asarray(xx, dtype=jnp.float32)
            unpack(xx_j, graphs, layout, backend="jax")
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

        def _c_and_jac(xx):
            """One primal: g and dense Jg via VJP (row-major flat jac)."""
            xx_j = jnp.asarray(xx, dtype=jnp.float32)
            c, vjp = jax.vjp(_c_fn, xx_j)
            c = jnp.reshape(c, (-1,))
            if c.size == 0:
                return np.zeros((0,), np.float64), np.zeros((0,), np.float64)
            Jc = jax.vmap(lambda e: vjp(e)[0])(jnp.eye(c.size, dtype=c.dtype))
            return (
                np.asarray(c, dtype=np.float64).reshape(-1),
                np.asarray(Jc, dtype=np.float64).reshape(-1),
            )

        val0, _ = _obj_grad(x)

        g0 = np.asarray(_c_fn(jnp.asarray(x, dtype=jnp.float32)), dtype=np.float64).reshape(-1)
        m = int(g0.size)

        class _Prob:
            """One-shot handoff: objective→gradient, constraints→jacobian."""

            def __init__(self):
                self._df = None
                self._jc = None
                self.last_f: float | None = None
                self.last_c: np.ndarray | None = None

            def objective(self, xx):
                f, self._df = _obj_grad(xx)
                self.last_f = float(f)
                return f

            def gradient(self, xx):
                if self._df is None:
                    _, self._df = _obj_grad(xx)
                df, self._df = self._df, None
                return df

            def constraints(self, xx):
                c, self._jc = _c_and_jac(xx)
                self.last_c = np.asarray(c, dtype=np.float64).reshape(-1)
                return c

            def jacobian(self, xx):
                if self._jc is None:
                    _, self._jc = _c_and_jac(xx)
                jc, self._jc = self._jc, None
                return jc

        prob = _Prob()
        if m:
            nlp = cyipopt.Problem(  # type: ignore[attr-defined]
                n=n,
                m=m,
                problem_obj=prob,
                lb=None,
                ub=None,
                cl=np.full(m, -1e19),
                cu=np.zeros(m),
            )
        else:
            nlp = cyipopt.Problem(  # type: ignore[attr-defined]
                n=n, m=0, problem_obj=prob, lb=None, ub=None
            )
        nlp.add_option("max_iter", int(max_iter))
        nlp.add_option("print_level", 5 if verbose else 0)
        x, info = nlp.solve(x)

        # reuse last IPOPT callback evals (avoid extra _obj_grad / _c_fn)
        val_f = float(info.get("obj_val", prob.last_f if prob.last_f is not None else val0))
        g_f = prob.last_c
        max_viol = float(np.max(g_f)) if g_f is not None and g_f.size else 0.0
        recorder.record(
            t, val_f, x, n_cons=m, max_viol=max_viol, status=int(info["status"])
        )

        if verbose:
            print(f"interior_nlp {t}/{steps}  J={val_f:.6g}  m={m}  status={info['status']}")
        if info["status"] < 0:
            break

    if recorder.best_x is None:
        raise RuntimeError("recorder.best_x is None; no step recorded a state")
    unpack(recorder.best_x, graphs, layout, backend="numpy")
    for g in graphs:
        g.refresh(backend="numpy")
