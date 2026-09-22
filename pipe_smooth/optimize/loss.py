"""Weighted Stage-1 soft loss J and ∇_x J through packed Hermite DOF."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax

from pipe_smooth.costs.clear import clear_capsule_active, clear_hull_active
from pipe_smooth.costs.path import graph_curvature, graph_length
from pipe_smooth.costs.space import graph_space, graph_tangent_reg
from pipe_smooth.costs.union import union_cost

from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.optimize.pack import PackLayout, unpack
from pipe_smooth.utils import backend as _backend


@dataclass
class LossWeights:
    """Scalar multipliers for J = αL + βK + μ_clear + λ_union + η_space + μ_v."""

    alpha: float = 1.0  # path length L (§6.4.1)
    beta: float = 1.0  # curvature K (§6.4.1)
    mu_obs: float = 1.0  # obstacle clearance J_obs (§6.4.2)
    mu_grp: float = 1.0  # cross-group clearance J_grp (§6.4.2)
    lam: float = 0.0  # junction union Chamfer J_union (§6.4.3)
    eta: float = 0.0  # sample spacing J_space (§7.1)
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
    """J(graphs) with frozen active clearance / union sets (outside AD detection)."""
    xp = _backend(backend)
    w = weights
    j = xp.asarray(0.0)
    for g in graphs:
        if w.alpha != 0:
            j = j + w.alpha * graph_length(g, backend=backend)
        if w.beta != 0:
            j = j + w.beta * graph_curvature(g, backend=backend)
        if w.eta != 0.0:
            j = j + graph_space(g, eta=w.eta, backend=backend)
        if w.mu_v != 0.0:
            j = j + graph_tangent_reg(g, mu=w.mu_v, backend=backend)            
        if union_hits and g.id in union_hits and w.lam != 0.0:
            j = j + union_cost(union_hits[g.id], g, lam=w.lam, backend=backend)

    if clear_active is not None and (w.mu_obs != 0.0 or w.mu_grp != 0.0):
        clear_fn = clear_capsule_active if clear_kind == "capsule" else clear_hull_active
        j = j + clear_fn(
            clear_active,
            graphs,
            mu_obs=w.mu_obs,
            mu_grp=w.mu_grp,
            backend=backend,
        )
    return j


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
    """(J, ∇_x J). Unpack → refresh → total_loss; detection stays outside."""

    def f(xx):
        unpack(xx, graphs, layout, backend="jax")
        for g in graphs:
            g.refresh(backend="jax")
        return total_loss(
            graphs,
            weights,
            clear_active=clear_active,
            clear_kind=clear_kind,
            union_hits=union_hits,
            backend="jax",
        )

    return jax.value_and_grad(f)(x)
