"""§6.4.2 — soft clearance on active pairs.

Detection / broadphase stay outside AD (§7.2). Use `phi` on (d_geom, d_req) from
collision kernels, or `clear_balls` for AD-friendly center–center pairs.
"""

from __future__ import annotations

from pipe_smooth.utils import backend as _backend
from .path import softplus


def phi(d_geom, d_req, soft_sigma: float = 1e-2, backend: str = "numpy"):
    """φ = [d_req - d_geom]_+^σ ² (scalar or per-pair array)."""
    xp = _backend(backend)
    gap = softplus(xp.asarray(d_req) - xp.asarray(d_geom), sigma=soft_sigma, backend=backend)
    return gap ** 2


def clear_balls(
    a,
    b,
    ra,
    rb,
    margin: float = 0.0,
    mu: float = 1.0,
    soft_sigma: float = 1e-2,
    sign: float = 1.0,
    backend: str = "numpy",
):
    """Soft ball–ball clearance μ Σ φ on an **active set** (§6.4.2 / §7.2).

    Call only after broadphase / narrow-phase has selected near pairs
    (d_geom < d_req + δ_buf). Do not pass the full all-pairs set.

    Args:
        a, b: pair centers, shape (..., 3) — one row/entry per active pair
        ra, rb: radii, broadcastable to the pair batch
        margin: extra separation δ in d_req = ra + rb + margin
        mu: weight in front of Σ φ
        soft_sigma: softplus hinge scale
        sign: +1 outside, -1 inside AABB (broadcastable); d_signed = sign * ||a-b||
        backend: 'numpy' | 'jax'

    Returns:
        Scalar μ Σ [d_req - sign*||a-b||]_+^σ ² over the active pairs.
    """
    xp = _backend(backend)
    a, b = xp.asarray(a), xp.asarray(b)
    d_geom = xp.asarray(sign) * xp.linalg.norm(a - b, axis=-1)
    d_req = xp.asarray(ra) + xp.asarray(rb) + margin
    return mu * xp.sum(phi(d_geom, d_req, soft_sigma=soft_sigma, backend=backend))
