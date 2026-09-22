"""§6.4.2 soft clearance + §7.2 SQP hard residuals on active pairs.

Detection / broadphase stay outside AD (§7.2). Recompute witness points from
frozen hits / hull features on the current Hermite graph; soft path applies
``phi``, hard path returns ``g = d_req - sign*||qa-qb||`` (feasible ``g<=0``).
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
    d_req,
    mu: float = 1.0,
    soft_sigma: float = 1e-2,
    sign: float | object = 1.0,
    backend: str = "numpy",
):
    """Soft clearance μ Σ φ on an **active set** (§6.4.2 / §7.2).

    Narrowphase supplies the full required gap ``d_req`` (radii + margin already
    folded in). Call only for pairs with d_geom < d_req.

    Args:
        a, b: witness points, shape (..., 3)
        d_req: required separation (scalar or broadcastable to pairs)
        mu: weight in front of Σ φ
        soft_sigma: softplus hinge scale
        sign: +1 outside / separated; -1 inside AABB; d_geom = sign * ||a-b||
        backend: 'numpy' | 'jax'

    Returns:
        Scalar μ Σ [d_req - sign*||a-b||]_+^σ ².
    """
    xp = _backend(backend)
    a, b = xp.asarray(a), xp.asarray(b)
    d_geom = xp.asarray(sign) * xp.linalg.norm(a - b, axis=-1)
    return mu * xp.sum(phi(d_geom, d_req, soft_sigma=soft_sigma, backend=backend))


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
    ends = {(g.id, sid): e for g in graphs for sid, e in g.hermite_ends(backend=backend).items()}
    narrow_obs, narrow_grp = active
    j = xp.asarray(0.0)

    for p in narrow_obs:
        p0, v0, p1, v1 = ends[(p.group_id, p.seg_id)]
        for h in p.hits:
            g0 = eval_hermite(p0, v0, p1, v1, h.hermite_t0, backend=backend)
            g1 = eval_hermite(p0, v0, p1, v1, h.hermite_t1, backend=backend)
            qa = (1.0 - h.chord_t) * g0 + h.chord_t * g1
            j = j + clear_balls(
                qa, h.qb, h.d_req, mu=mu_obs, soft_sigma=soft_sigma, sign=h.sign, backend=backend
            )

    for p in narrow_grp:
        a0, av0, a1, av1 = ends[(p.group_a, p.seg_a)]
        b0, bv0, b1, bv1 = ends[(p.group_b, p.seg_b)]
        for h in p.hits:
            qa0 = eval_hermite(a0, av0, a1, av1, h.hermite_t0_a, backend=backend)
            qa1 = eval_hermite(a0, av0, a1, av1, h.hermite_t1_a, backend=backend)
            qb0 = eval_hermite(b0, bv0, b1, bv1, h.hermite_t0_b, backend=backend)
            qb1 = eval_hermite(b0, bv0, b1, bv1, h.hermite_t1_b, backend=backend)
            qa = (1.0 - h.chord_t_a) * qa0 + h.chord_t_a * qa1
            qb = (1.0 - h.chord_t_b) * qb0 + h.chord_t_b * qb1
            j = j + clear_balls(
                qa, qb, h.d_req, mu=mu_grp, soft_sigma=soft_sigma, sign=h.sign, backend=backend
            )

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
    ends = {(g.id, sid): e for g in graphs for sid, e in g.hermite_ends(backend=backend).items()}
    narrow_obs, narrow_grp = active
    j = xp.asarray(0.0)

    for p in narrow_obs:
        p0, v0, p1, v1 = ends[(p.group_id, p.seg_id)]
        C = hermite_bezier_controls(p0, v0, p1, v1, backend=backend)
        C_sub = bezier_on_interval(C, p.hermite_t0, p.hermite_t1, backend=backend)
        qa = xp.asarray(p.weights) @ (C_sub + xp.asarray(p.shift))
        j = j + clear_balls(
            qa, p.qb, p.d_req, mu=mu_obs, soft_sigma=soft_sigma, sign=p.sign, backend=backend
        )

    for p in narrow_grp:
        a0, av0, a1, av1 = ends[(p.group_a, p.seg_a)]
        b0, bv0, b1, bv1 = ends[(p.group_b, p.seg_b)]
        Ca = hermite_bezier_controls(a0, av0, a1, av1, backend=backend)
        Cb = hermite_bezier_controls(b0, bv0, b1, bv1, backend=backend)
        Ca_sub = bezier_on_interval(Ca, p.hermite_t0_a, p.hermite_t1_a, backend=backend)
        Cb_sub = bezier_on_interval(Cb, p.hermite_t0_b, p.hermite_t1_b, backend=backend)
        qa = xp.asarray(p.weights_a) @ (Ca_sub + xp.asarray(p.shift_a))
        qb = xp.asarray(p.weights_b) @ (Cb_sub + xp.asarray(p.shift_b))
        j = j + clear_balls(
            qa, qb, p.d_req, mu=mu_grp, soft_sigma=soft_sigma, sign=p.sign, backend=backend
        )

    return j


def clear_capsule_active_constraint(
    active: tuple[list[NarrowObsPair], list[NarrowGrpPair]],
    graphs: Sequence[OptimizeGraph],
    backend: str = "numpy",
):
    """SQP capsule clearance (§7.2.1). Feasible iff all residuals ``<= 0``.

    Frozen hits, live Hermite. Per hit (``req = 0`` if ``sign < 0``, else ``d_req``):
      ``g = req - sign * ||qa - qb||  <= 0``

    Args:
        active: ``(narrow_obs, narrow_grp)`` from capsule narrowphase
        graphs: OptimizeGraph sequence (``hermite_ends`` / refresh)
        backend: ``'numpy'`` | ``'jax'``

    Returns:
        Flat vector of residuals ``g`` with ``g <= 0`` feasible (one entry per hit).
    """
    xp = _backend(backend)
    ends = {(g.id, sid): e for g in graphs for sid, e in g.hermite_ends(backend=backend).items()}
    narrow_obs, narrow_grp = active
    parts = []

    for p in narrow_obs:
        p0, v0, p1, v1 = ends[(p.group_id, p.seg_id)]
        for h in p.hits:
            g0 = eval_hermite(p0, v0, p1, v1, h.hermite_t0, backend=backend)
            g1 = eval_hermite(p0, v0, p1, v1, h.hermite_t1, backend=backend)
            qa = (1.0 - h.chord_t) * g0 + h.chord_t * g1
            d = xp.asarray(h.sign) * xp.linalg.norm(qa - xp.asarray(h.qb))
            parts.append(xp.asarray(0.0 if h.sign < 0 else h.d_req) - d)

    for p in narrow_grp:
        a0, av0, a1, av1 = ends[(p.group_a, p.seg_a)]
        b0, bv0, b1, bv1 = ends[(p.group_b, p.seg_b)]
        for h in p.hits:
            qa0 = eval_hermite(a0, av0, a1, av1, h.hermite_t0_a, backend=backend)
            qa1 = eval_hermite(a0, av0, a1, av1, h.hermite_t1_a, backend=backend)
            qb0 = eval_hermite(b0, bv0, b1, bv1, h.hermite_t0_b, backend=backend)
            qb1 = eval_hermite(b0, bv0, b1, bv1, h.hermite_t1_b, backend=backend)
            qa = (1.0 - h.chord_t_a) * qa0 + h.chord_t_a * qa1
            qb = (1.0 - h.chord_t_b) * qb0 + h.chord_t_b * qb1
            d = xp.asarray(h.sign) * xp.linalg.norm(qa - qb)
            parts.append(xp.asarray(0.0 if h.sign < 0 else h.d_req) - d)

    if not parts:
        return xp.zeros((0,))
    return xp.stack([xp.reshape(p, ()) for p in parts])


def clear_hull_active_constraint(
    active: tuple[list[NarrowHullObsPair], list[NarrowHullGrpPair]],
    graphs: Sequence[OptimizeGraph],
    backend: str = "numpy",
):
    """SQP Hull4/GJK clearance (§7.2.2). Feasible iff all residuals ``<= 0``.

    Frozen weights/qb/shift, live Hermite. Per pair (``req = 0`` if ``sign < 0``, else ``d_req``):
      ``g = req - sign * ||qa - qb||  <= 0``

    Args:
        active: ``(narrow_obs, narrow_grp)`` from hull narrowphase
        graphs: OptimizeGraph sequence (``hermite_ends`` / refresh)
        backend: ``'numpy'`` | ``'jax'``

    Returns:
        Flat vector of residuals ``g`` with ``g <= 0`` feasible (one entry per pair).
    """
    xp = _backend(backend)
    ends = {(g.id, sid): e for g in graphs for sid, e in g.hermite_ends(backend=backend).items()}
    narrow_obs, narrow_grp = active
    parts = []

    for p in narrow_obs:
        p0, v0, p1, v1 = ends[(p.group_id, p.seg_id)]
        C = hermite_bezier_controls(p0, v0, p1, v1, backend=backend)
        C_sub = bezier_on_interval(C, p.hermite_t0, p.hermite_t1, backend=backend)
        qa = xp.asarray(p.weights) @ (C_sub + xp.asarray(p.shift))
        d = xp.asarray(p.sign) * xp.linalg.norm(qa - xp.asarray(p.qb))
        parts.append(xp.asarray(0.0 if p.sign < 0 else p.d_req) - d)

    for p in narrow_grp:
        a0, av0, a1, av1 = ends[(p.group_a, p.seg_a)]
        b0, bv0, b1, bv1 = ends[(p.group_b, p.seg_b)]
        Ca = hermite_bezier_controls(a0, av0, a1, av1, backend=backend)
        Cb = hermite_bezier_controls(b0, bv0, b1, bv1, backend=backend)
        Ca_sub = bezier_on_interval(Ca, p.hermite_t0_a, p.hermite_t1_a, backend=backend)
        Cb_sub = bezier_on_interval(Cb, p.hermite_t0_b, p.hermite_t1_b, backend=backend)
        qa = xp.asarray(p.weights_a) @ (Ca_sub + xp.asarray(p.shift_a))
        qb = xp.asarray(p.weights_b) @ (Cb_sub + xp.asarray(p.shift_b))
        d = xp.asarray(p.sign) * xp.linalg.norm(qa - qb)
        parts.append(xp.asarray(0.0 if p.sign < 0 else p.d_req) - d)

    if not parts:
        return xp.zeros((0,))
    return xp.stack([xp.reshape(p, ()) for p in parts])

