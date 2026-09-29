"""Weighted Stage-1 soft loss J and ∇_x J through packed Hermite DOF."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal
from warnings import warn

import jax
import numpy as np

from pipe_smooth.costs.clear import (
    pack_capsule_active,
    clear_capsule_packed,
    pack_hull_active,
    clear_hull_packed,
)
from pipe_smooth.costs.path import graph_curvature, graph_length
from pipe_smooth.costs.space import graph_space, graph_tangent_reg
from pipe_smooth.costs.union import pack_union_hits, union_cost_packed

from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.optimize.pack import PackLayout, unpack
from pipe_smooth.optimize.pack_helper import sync_layout_budgets
from pipe_smooth.utils import backend as _backend


@dataclass
class LossWeights:
    """Scalar multipliers for J = αL + βK + μ_clear + λ_union + η_space + μ_v."""

    alpha: float = 1.0  # path chord energy L=ΣΔs² (§6.4.1); [length²]
    beta: float = 1.0  # curvature K (§6.4.1)
    mu_obs: float = 1.0  # obstacle clearance J_obs (§6.4.2)
    mu_grp: float = 1.0  # cross-group clearance J_grp (§6.4.2)
    lam: float = 0.1  # junction union Chamfer J_union; (d/(r_a+r_b))² Δs (§6.4.3)
    eta: float = 10.0  # spacing energy J_space=Σ(Δs−mean)² (§7.1); [length²] vs α
    mu_v: float = 0.0  # Hermite ||v|| scale hinge J_v (§7.1)


def total_loss(
    graphs: Sequence[OptimizeGraph],
    weights: LossWeights,
    *,
    clear_kind: Literal["capsule", "hull"] = "capsule",
    layout: PackLayout | None = None,
    x=None,
    clear_obs_buf: dict[str, Any] | None = None,
    clear_grp_buf: dict[str, Any] | None = None,
    union_buf: dict[str, Any] | None = None,
    backend: str = "numpy",
):
    """(J, parts) with packed clear/union buffers (detection outside AD).

    Clear/union need ``layout`` + ``x`` + bufs. ``parts``: L, K, space, v, union, clear.
    Graphs must be refreshed (``seg_xyz`` / ``seg_ends``).
    """
    xp = _backend(backend)
    w = weights
    z = xp.asarray(0.0)
    parts = {"L": z, "K": z, "space": z, "v": z, "union": z, "clear": z}
    for g in graphs:
        if w.alpha != 0:
            parts["L"] = parts["L"] + w.alpha * graph_length(g, backend=backend)
        if w.beta != 0:
            parts["K"] = parts["K"] + w.beta * graph_curvature(g, backend=backend)
        if w.eta != 0.0:
            parts["space"] = parts["space"] + graph_space(g, eta=w.eta, backend=backend)
        if w.mu_v != 0.0:
            parts["v"] = parts["v"] + graph_tangent_reg(g, mu=w.mu_v, backend=backend)

    if w.lam != 0.0 and union_buf is not None and layout is not None and x is not None:
        parts["union"] = union_cost_packed(
            x, union_buf, layout, lam=w.lam, backend=backend
        )

    if (w.mu_obs != 0.0 or w.mu_grp != 0.0) and (
        clear_obs_buf is not None
        and clear_grp_buf is not None
        and layout is not None
        and x is not None
    ):
        clear_fn = clear_capsule_packed if clear_kind == "capsule" else clear_hull_packed
        parts["clear"] = clear_fn(
            x,
            clear_obs_buf,
            clear_grp_buf,
            layout,
            mu_obs=w.mu_obs,
            mu_grp=w.mu_grp,
            backend=backend,
        )

    j = parts["L"] + parts["K"] + parts["space"] + parts["v"] + parts["union"] + parts["clear"]
    return j, parts


# Recompiled when layout budgets / clear_kind / weights / graph identity change
# — not when packed clear/union buffer *values* change.
_vg_cache: dict[str, Any] = {"key": None, "fn": None}


def loss_and_grad(
    x: Any,
    graphs: Sequence[OptimizeGraph],
    layout: PackLayout,
    weights: LossWeights,
    *,
    clear_active=None,
    clear_kind: Literal["capsule", "hull"] = "capsule",
    union_hits: dict[str, Any] | None = None,
):
    """(J, ∇_x J, parts). Unpack → refresh → packed clear/union on ``x``.

    Packs active sets outside AD; ``jit`` only depends on fixed ``p_max_*`` shapes.
    """
    w = weights
    empty_clear = ([], [])
    active = clear_active if clear_active is not None else empty_clear
    obs_l, grp_l = active
    if clear_kind == "hull":
        n_obs, n_grp = len(obs_l), len(grp_l)
    else:
        n_obs = sum(len(p.hits) for p in obs_l)
        n_grp = sum(len(p.hits) for p in grp_l)
    n_union = 0
    if union_hits:
        for hits in union_hits.values():
            for h in hits:
                n_union += int(np.asarray(h.ts_a).reshape(-1).shape[0])
    # TODO: non-stable version
    # sync_layout_budgets(layout, n_obs=n_obs, n_grp=n_grp, n_union=n_union)

    obs_l, grp_l = list(obs_l), list(grp_l)
    rng = np.random.default_rng()
    if n_obs > layout.p_max_obs:
        warn(
            f"clear obs slots {n_obs} > p_max_obs={layout.p_max_obs}; shuffle+truncate",
            stacklevel=2,
        )
        rng.shuffle(obs_l)
    if n_grp > layout.p_max_grp:
        warn(
            f"clear grp slots {n_grp} > p_max_grp={layout.p_max_grp}; shuffle+truncate",
            stacklevel=2,
        )
        rng.shuffle(grp_l)
    active = (obs_l, grp_l)
    if clear_kind == "capsule":
        obs_b, grp_b = pack_capsule_active(active, layout, backend="jax")
    else:
        obs_b, grp_b = pack_hull_active(active, layout, backend="jax")

    union_b = pack_union_hits(union_hits, layout, backend="jax")

    key = (
        id(graphs),
        id(layout),
        layout.p_max_obs,
        layout.p_max_grp,
        layout.p_max_union,
        clear_kind,
        w.alpha,
        w.beta,
        w.mu_obs,
        w.mu_grp,
        w.lam,
        w.eta,
        w.mu_v,
    )
    if _vg_cache["key"] != key:

        def f(xx, obs_buf, grp_buf, uni_buf):
            unpack(xx, graphs, layout, backend="jax")
            for g in graphs:
                g.refresh(backend="jax")
            return total_loss(
                graphs,
                weights,
                clear_kind=clear_kind,
                layout=layout,
                x=xx,
                clear_obs_buf=obs_buf,
                clear_grp_buf=grp_buf,
                union_buf=uni_buf,
                backend="jax",
            )

        _vg_cache["fn"] = jax.jit(jax.value_and_grad(f, argnums=0, has_aux=True))
        _vg_cache["key"] = key

    (val, parts), grad = _vg_cache["fn"](x, obs_b, grp_b, union_b)
    return val, grad, parts
