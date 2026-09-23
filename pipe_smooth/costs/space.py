"""§7.1 — sample spacing J_space + Hermite ||v|| scale hinge J_v.

Soft AD: spacing energy Σ(Δs−mean)² (same [length²] as path chord energy); optional hinge.
SQP: optional relative band residuals about mean Δs.
"""

from __future__ import annotations

from pipe_smooth.utils import backend as _backend
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.costs.path import softplus

_EPS = 1e-12


def space_cost(xyz, eta: float = 1.0, ds_max=None, soft_sigma: float = 1e-2, backend: str = "numpy"):
    """η Σ (Δs_i - mean(Δs))²; optional η Σ [Δs_i - Δs_max]_+^σ ².

    Absolute spacing energy [length²], matched to ``path_length`` chord energy so
    α/η stay scale-comparable under geometric rescaling.
    """
    xp = _backend(backend)
    xyz = xp.asarray(xyz)
    ds = xp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    mean = xp.mean(ds)
    j = xp.sum((ds - mean) ** 2)
    if ds_max is not None:
        j = j + xp.sum(softplus(ds - ds_max, sigma=soft_sigma, backend=backend) ** 2)
    return eta * j


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


def graph_tangent_constraint(
    graph: OptimizeGraph,
    beta: float = 0.4,
    backend: str = "numpy",
):
    """SQP Hermite tangent-norm bound (§7.1.2). Feasible iff all residuals ``<= 0``.

    Per Hermite end from ``hermite_ends`` (ports via path clamps):
      ``||v0|| - β ||p1-p0||  <= 0``
      ``||v1|| - β ||p1-p0||  <= 0``

    Args:
        graph: OptimizeGraph
        beta: scale vs chord; β=1 ≈ uniform-t straight speed; β≪1 tighter handles
        backend: ``'numpy'`` | ``'jax'``

    Returns:
        Flat vector of residuals ``g`` with ``g <= 0`` feasible (two entries per segment).
    """
    xp = _backend(backend)
    parts = []
    for p0, v0, p1, v1 in graph.hermite_ends(backend=backend).values():
        vmax = beta * (xp.linalg.norm(p1 - p0) + _EPS)
        parts.append(xp.linalg.norm(v0) - vmax)
        parts.append(xp.linalg.norm(v1) - vmax)
    if not parts:
        return xp.zeros((0,))
    return xp.stack(parts)


def graph_space_band_constraint(
    graph: OptimizeGraph,
    eps: float = 0.2,
    upper_only: bool = False,
    backend: str = "numpy",
):
    """SQP sample-spacing band (§7.1.1). Feasible iff all returned residuals ``<= 0``.

    Per Hermite segment (call ``graph.refresh`` first):
      ``hi = Δs - (1+ε) mean(Δs)  <= 0``
      ``lo = (1-ε) mean(Δs) - Δs  <= 0``  (omitted if ``upper_only``)

    Args:
        graph: OptimizeGraph with refreshed segment samples
        eps: relative band width ε around mean Δs
        upper_only: if True, only upper bound ``hi <= 0``
        backend: ``'numpy'`` | ``'jax'``

    Returns:
        Flat vector of residuals ``g`` with ``g <= 0`` feasible (one block per segment).
    """
    xp = _backend(backend)
    parts = []
    for seg in graph.segments.values():
        _, xyz, _ = seg.samples(backend=backend)
        ds = xp.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
        mean = xp.mean(ds)
        hi = ds - (1.0 + eps) * mean
        if upper_only:
            parts.append(hi)
        else:
            lo = (1.0 - eps) * mean - ds
            parts.append(xp.concatenate([hi, lo], axis=0))
    if not parts:
        return xp.zeros((0,))
    return xp.concatenate(parts, axis=0)
