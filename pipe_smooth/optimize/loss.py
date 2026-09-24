"""Weighted Stage-1 soft loss J and ∇_x J through packed Hermite DOF."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax
import numpy as np

from pipe_smooth.costs.clear import (
    clear_capsule_active,
    clear_hull_active,
    pack_capsule_active,
    clear_capsule_packed,
    pack_hull_active,
    clear_hull_packed,
)
from pipe_smooth.costs.path import graph_curvature, graph_length
from pipe_smooth.costs.space import graph_space, graph_tangent_reg
from pipe_smooth.costs.union import union_cost

from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.optimize.pack import PackLayout, unpack
from pipe_smooth.utils import backend as _backend


@dataclass
class LossWeights:
    """Scalar multipliers for J = αL + βK + μ_clear + λ_union + η_space + μ_v."""

    alpha: float = 1.0  # path chord energy L=ΣΔs² (§6.4.1); [length²]
    beta: float = 1.0  # curvature K (§6.4.1)
    mu_obs: float = 1.0  # obstacle clearance J_obs (§6.4.2)
    mu_grp: float = 1.0  # cross-group clearance J_grp (§6.4.2)
    lam: float = 0.0  # junction union Chamfer J_union (§6.4.3)
    eta: float = 1.0  # spacing energy J_space=Σ(Δs−mean)² (§7.1); [length²] vs α
    mu_v: float = 0.0  # Hermite ||v|| scale hinge J_v (§7.1)


def total_loss(
    graphs: Sequence[OptimizeGraph],
    weights: LossWeights,
    *,
    clear_active=None,
    clear_kind: Literal["capsule", "hull"] = "capsule",
    union_hits: dict[str, Any] | None = None,
    backend: str = "numpy",
):
    """(J, parts) with frozen active clearance / union sets (outside AD detection).

    ``parts`` holds weighted contributions: L, K, space, v, union, clear.
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
        if union_hits and g.id in union_hits and w.lam != 0.0:
            parts["union"] = parts["union"] + union_cost(
                union_hits[g.id], g, lam=w.lam, backend=backend
            )

    if clear_active is not None and (w.mu_obs != 0.0 or w.mu_grp != 0.0):
        clear_fn = clear_capsule_active if clear_kind == "capsule" else clear_hull_active
        parts["clear"] = clear_fn(
            clear_active,
            graphs,
            mu_obs=w.mu_obs,
            mu_grp=w.mu_grp,
            backend=backend,
        )
    j = parts["L"] + parts["K"] + parts["space"] + parts["v"] + parts["union"] + parts["clear"]
    return j, parts


def _union_key(union_hits) -> Any:
    if not union_hits:
        return None
    return tuple(
        (gid, tuple((h.seg_a, h.seg_b, int(np.asarray(h.ts_a).shape[0])) for h in hits))
        for gid, hits in sorted(union_hits.items())
    )


# Recompiled when layout budgets / union structure / weights change — not clear hit values.
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
    """(J, ∇_x J, parts). Unpack → refresh → total_loss; detection stays outside.

    ``jit`` while active-set *content signature* is unchanged (not ``id(...)``).
    """
    w = weights
    key = (
        id(graphs),
        id(layout),
        layout.p_max_obs,
        layout.p_max_grp,
        _union_key(union_hits),
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

        def f(xx, obs_b, grp_b):
            unpack(xx, graphs, layout, backend="jax")
            for g in graphs:
                g.refresh(backend="jax")
            j, parts = total_loss(
                graphs,
                weights,
                clear_active=None,
                union_hits=union_hits,
                backend="jax",
            )

        _vg_cache["fn"] = jax.jit(jax.value_and_grad(f, has_aux=True))
        _vg_cache["key"] = key

    (val, parts), grad = _vg_cache["fn"](x)
    return val, grad, parts
