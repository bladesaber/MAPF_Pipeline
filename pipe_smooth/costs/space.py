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
    assert xyz.ndim == 3 and xyz.shape[-1] == 3, f"xyz need (B,T,3), got {xyz.shape}"
    ds = xp.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
    mean = xp.mean(ds, axis=-1, keepdims=True)  # (B, 1)
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
    xyz=None,
):
    """Σ space_cost over Hermite segments; each seg once. Pass ``xyz`` (no ``seg_*``)."""
    if xyz is None:
        _, xyz, _ = graph.refresh(backend=backend)
    return space_cost(
        xyz, eta=eta, ds_max=ds_max, soft_sigma=soft_sigma, backend=backend
    )


def graph_tangent_reg(
    graph: OptimizeGraph,
    beta: float = 0.4,
    mu: float = 1.0,
    soft_sigma: float = 1e-2,
    backend: str = "numpy",
    ends=None,
):
    """μ Σ [||v|| - β||p1-p0||]_+^σ ² on each Hermite end (port clamps via hermite_ends).

    β=1 ≈ straight uniform-t speed ||γ'||=chord; β≪1 tightens handles (anti κ-spike).
    """
    xp = _backend(backend)
    if ends is None:
        ends, _, _ = graph.refresh(backend=backend)
    ends = xp.asarray(ends)  # (S, 4, 3)
    p0, v0, p1, v1 = ends[:, 0], ends[:, 1], ends[:, 2], ends[:, 3] # (S, 3)
    vmax = beta * (xp.linalg.norm(p1 - p0, axis=-1) + _EPS) # (S, )
    j = softplus(xp.linalg.norm(v0, axis=-1) - vmax, sigma=soft_sigma, backend=backend) ** 2
    j = j + softplus(xp.linalg.norm(v1, axis=-1) - vmax, sigma=soft_sigma, backend=backend) ** 2
    return mu * xp.sum(j)


def graph_tangent_constraint(
    graph: OptimizeGraph,
    beta: float = 0.4,
    backend: str = "numpy",
    ends=None,
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
    if ends is None:
        ends, _, _ = graph.refresh(backend=backend)
    ends = xp.asarray(ends)
    if int(ends.shape[0]) == 0:
        return xp.zeros((0,))
    p0, v0, p1, v1 = ends[:, 0], ends[:, 1], ends[:, 2], ends[:, 3]
    vmax = beta * (xp.linalg.norm(p1 - p0, axis=-1) + _EPS) # (S, )
    g = xp.stack(
        [xp.linalg.norm(v0, axis=-1) - vmax, xp.linalg.norm(v1, axis=-1) - vmax],
        axis=1,
    )
    return g.reshape(-1)


def graph_space_band_constraint(
    graph: OptimizeGraph,
    eps: float = 0.2,
    upper_only: bool = False,
    backend: str = "numpy",
    xyz=None,
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
    if xyz is None:
        _, xyz, _ = graph.refresh(backend=backend)
    xyz = xp.asarray(xyz)
    assert xyz.ndim == 3 and xyz.shape[-1] == 3, f"seg_xyz need (S,T,3), got {xyz.shape}"
    if int(xyz.shape[0]) == 0:
        return xp.zeros((0,))
    ds = xp.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
    mean = xp.mean(ds, axis=-1, keepdims=True)
    hi = ds - (1.0 + eps) * mean
    if upper_only:
        return hi.reshape(-1)
    lo = (1.0 - eps) * mean - ds
    return xp.concatenate([hi, lo], axis=-1).reshape(-1)
