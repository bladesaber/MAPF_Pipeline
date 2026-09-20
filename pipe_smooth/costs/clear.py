"""§6.4.2 — soft clearance on active pairs.

Detection / broadphase stay outside AD (§7.2). Recompute witness points from
frozen hits / hull features on the current Hermite graph, then apply `phi`.
"""

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

from pipe_smooth.utils import backend as _backend
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.geometry.hermite import (
    eval_hermite,
    hermite_bezier_controls,
    bezier_on_interval,
)
from .path import softplus

if TYPE_CHECKING:
    from pipe_smooth.collision.capsule import NarrowObsPair, NarrowGrpPair
    from pipe_smooth.collision.gjk import NarrowHullObsPair, NarrowHullGrpPair


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


def _hermite_ends(g: OptimizeGraph, sid: str):
    """Segment endpoints in the same orientation as collision narrowphase."""
    seg = g.segments[sid]
    nu, nv = g.nodes[seg.u], g.nodes[seg.v]
    for path in g.usage_paths:
        for i, s in enumerate(path.segments):
            if s != sid:
                continue
            if path.reverse[i]:
                p0, v0, s0 = nv.pos, -nv.v, nv.sigma  # type: ignore[operator]
                p1, v1, s1 = nu.pos, -nu.v, nu.sigma  # type: ignore[operator]
            else:
                p0, v0, s0 = nu.pos, nu.v, nu.sigma
                p1, v1, s1 = nv.pos, nv.v, nv.sigma
            if i == 0:
                v0, s0 = path.v_start, path.sigma_start
            if i == len(path.segments) - 1:
                v1, s1 = path.v_end, path.sigma_end
            return p0, v0, s0, p1, v1, s1
    return nu.pos, nu.v, nu.sigma, nv.pos, nv.v, nv.sigma


def clear_capsule_active(
    active: tuple[list[NarrowObsPair], list[NarrowGrpPair]],
    graphs: Sequence[OptimizeGraph],
    mu_obs: float = 1.0,
    mu_grp: float = 1.0,
    soft_sigma: float = 1e-2,
    backend: str = "numpy",
):
    """Soft J_clear from capsule narrowphase pairs (frozen hits, live Hermite)."""
    xp = _backend(backend)
    by_g = {g.id: g for g in graphs}
    narrow_obs, narrow_grp = active
    j = xp.asarray(0.0)

    for p in narrow_obs:
        ends = _hermite_ends(by_g[p.group_id], p.seg_id)
        for h in p.hits:
            g0 = eval_hermite(*ends, h.hermite_t0, backend=backend)
            g1 = eval_hermite(*ends, h.hermite_t1, backend=backend)
            qa = (1.0 - h.chord_t) * g0 + h.chord_t * g1
            d = xp.asarray(h.sign) * xp.linalg.norm(qa - xp.asarray(h.qb))
            j = j + mu_obs * phi(d, p.d_req, soft_sigma=soft_sigma, backend=backend)

    for p in narrow_grp:
        ends_a = _hermite_ends(by_g[p.group_a], p.seg_a)
        ends_b = _hermite_ends(by_g[p.group_b], p.seg_b)
        for h in p.hits:
            a0 = eval_hermite(*ends_a, h.hermite_t0_a, backend=backend)
            a1 = eval_hermite(*ends_a, h.hermite_t1_a, backend=backend)
            b0 = eval_hermite(*ends_b, h.hermite_t0_b, backend=backend)
            b1 = eval_hermite(*ends_b, h.hermite_t1_b, backend=backend)
            qa = (1.0 - h.chord_t_a) * a0 + h.chord_t_a * a1
            qb = (1.0 - h.chord_t_b) * b0 + h.chord_t_b * b1
            d = xp.asarray(h.sign) * xp.linalg.norm(qa - qb)
            j = j + mu_grp * phi(d, p.d_req, soft_sigma=soft_sigma, backend=backend)

    return j


def clear_hull_active(
    active: tuple[list[NarrowHullObsPair], list[NarrowHullGrpPair]],
    graphs: Sequence[OptimizeGraph],
    mu_obs: float = 1.0,
    mu_grp: float = 1.0,
    soft_sigma: float = 1e-2,
    backend: str = "numpy",
):
    """Soft J_clear from Hull4 GJK pairs (frozen weights/qb/shift, live Hermite)."""
    xp = _backend(backend)
    by_g = {g.id: g for g in graphs}
    narrow_obs, narrow_grp = active
    j = xp.asarray(0.0)

    for p in narrow_obs:
        C = hermite_bezier_controls(*_hermite_ends(by_g[p.group_id], p.seg_id), backend=backend)
        C_sub = bezier_on_interval(C, p.hermite_t0, p.hermite_t1, backend=backend)
        qa = xp.asarray(p.weights) @ (C_sub + xp.asarray(p.shift))
        d = xp.asarray(p.sign) * xp.linalg.norm(qa - xp.asarray(p.qb))
        j = j + mu_obs * phi(d, p.d_req, soft_sigma=soft_sigma, backend=backend)

    for p in narrow_grp:
        Ca = hermite_bezier_controls(*_hermite_ends(by_g[p.group_a], p.seg_a), backend=backend)
        Cb = hermite_bezier_controls(*_hermite_ends(by_g[p.group_b], p.seg_b), backend=backend)
        Ca_sub = bezier_on_interval(Ca, p.hermite_t0_a, p.hermite_t1_a, backend=backend)
        Cb_sub = bezier_on_interval(Cb, p.hermite_t0_b, p.hermite_t1_b, backend=backend)
        qa = xp.asarray(p.weights_a) @ (Ca_sub + xp.asarray(p.shift_a))
        qb = xp.asarray(p.weights_b) @ (Cb_sub + xp.asarray(p.shift_b))
        d = xp.asarray(p.sign) * xp.linalg.norm(qa - qb)
        j = j + mu_grp * phi(d, p.d_req, soft_sigma=soft_sigma, backend=backend)

    return j
