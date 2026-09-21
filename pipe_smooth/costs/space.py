"""§7.1 — sample spacing J_space + Hermite ||v|| scale hinge J_v.

Soft AD: nonuniform Δs; optional ||γ'|| vs chord length (anti long-handle spikes).
SQP: optional relative band residuals about mean Δs.
"""

from __future__ import annotations

from pipe_smooth.utils import backend as _backend
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.costs.path import softplus

_EPS = 1e-12


def space_cost(xyz, eta: float = 1.0, ds_max=None, soft_sigma: float = 1e-2, backend: str = "numpy"):
    """η Σ (Δs_i - mean(Δs))²; optional η Σ [Δs_i - Δs_max]_+^σ ²."""
    xp = _backend(backend)
    xyz = xp.asarray(xyz)
    ds = xp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    mean = xp.mean(ds)
    j = xp.sum((ds - mean) ** 2)
    if ds_max is not None:
        j = j + xp.sum(softplus(ds - ds_max, sigma=soft_sigma, backend=backend) ** 2)
    return eta * j


def space_band_residuals(xyz, eps: float = 0.2, upper_only: bool = False, backend: str = "numpy"):
    """SQP-style residuals ≤ 0: ds ≤ (1+ε)mean; optionally ds ≥ (1-ε)mean.

    Returns flat vector of g(X) with g≤0 feasible (§7.1 (2)). Linearize at iterate in SQP.
    """
    xp = _backend(backend)
    xyz = xp.asarray(xyz)
    ds = xp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    mean = xp.mean(ds)
    hi = ds - (1.0 + eps) * mean
    if upper_only:
        return hi
    lo = (1.0 - eps) * mean - ds
    return xp.concatenate([hi, lo], axis=0)


def graph_space(
    graph: OptimizeGraph,
    eta: float = 1.0,
    ds_max=None,
    soft_sigma: float = 1e-2,
    backend: str = "numpy",
):
    """Σ space_cost over Hermite segments; each seg once. Requires ``graph.refresh``."""
    xp = _backend(backend)
    total = xp.asarray(0.0)
    for seg in graph.segments.values():
        _, xyz, _ = seg.samples(backend=backend)
        total = total + space_cost(
            xyz, eta=eta, ds_max=ds_max, soft_sigma=soft_sigma, backend=backend
        )
    return total


def graph_tangent_reg(
    graph: OptimizeGraph,
    beta: float = 0.4,
    mu: float = 1.0,
    soft_sigma: float = 1e-2,
    backend: str = "numpy",
):
    """μ Σ [||v|| - β||p1-p0||]_+^σ ² on each Hermite end (port clamps via hermite_ends).

    β=1 ≈ straight uniform-t speed ||γ'||=chord; β≪1 tightens handles (anti κ-spike).
    """
    xp = _backend(backend)
    j = xp.asarray(0.0)
    for p0, v0, p1, v1 in graph.hermite_ends(backend=backend).values():
        vmax = beta * (xp.linalg.norm(p1 - p0) + _EPS)
        j = j + softplus(xp.linalg.norm(v0) - vmax, sigma=soft_sigma, backend=backend) ** 2
        j = j + softplus(xp.linalg.norm(v1) - vmax, sigma=soft_sigma, backend=backend) ** 2
    return mu * j
