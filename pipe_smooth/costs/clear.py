"""§6.4.2 soft clearance + §7.2 SQP hard residuals on active pairs.

Detection / broadphase stay outside AD (§7.2). Recompute witness points from
frozen hits / hull features on the current Hermite graph; soft path applies
``phi``, hard path returns ``g = d_req - sign*||qa-qb||`` (feasible ``g<=0``).

Packed path (``pack_*_active`` + ``clear_*_packed``): fixed-shape buffers +
``PackLayout.ends_from_x`` gather for stable JAX JIT.
"""

from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING

import numpy as np

from pipe_smooth.utils import backend as _backend
from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.geometry.hermite import (
    eval_hermite,
    hermite_bezier_controls,
    hermite_bezier_on_interval,
    bezier_on_interval,
)
from .path import softplus

if TYPE_CHECKING:
    from pipe_smooth.collision.capsule import NarrowObsPair, NarrowGrpPair
    from pipe_smooth.collision.gjk import NarrowHullObsPair, NarrowHullGrpPair
    from pipe_smooth.optimize.pack import PackLayout


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


# ---------------------------------------------------------------------------
# Fixed-shape capsule clear (ends table + seg_idx); refresh buffers outside AD
# ---------------------------------------------------------------------------

def pack_capsule_active(
    active: tuple[list, list],
    layout: PackLayout,
    backend: str = "numpy",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Flatten narrow hits into fixed obs/grp buffers (truncate at ``p_max_*``)."""
    xp = _backend(backend)
    narrow_obs, narrow_grp = active
    po, pg = layout.p_max_obs, layout.p_max_grp
    idx = layout.seg_index

    obs = {
        "seg_idx": np.zeros(po, np.int32),
        "mask": np.zeros(po),
        "hermite_t0": np.zeros(po),
        "hermite_t1": np.zeros(po),
        "chord_t": np.zeros(po),
        "qb": np.zeros((po, 3)),
        "d_req": np.zeros(po),
        "sign": np.ones(po),
    }
    i = 0
    for p in narrow_obs:
        s = idx[(p.group_id, p.seg_id)]
        for h in p.hits:
            if i >= po:
                break
            obs["seg_idx"][i] = s
            obs["mask"][i] = 1.0
            obs["hermite_t0"][i] = h.hermite_t0
            obs["hermite_t1"][i] = h.hermite_t1
            obs["chord_t"][i] = h.chord_t
            obs["qb"][i] = np.asarray(h.qb, float).reshape(3)
            obs["d_req"][i] = h.d_req
            obs["sign"][i] = h.sign
            i += 1
        if i >= po:
            break

    grp = {
        "seg_a": np.zeros(pg, np.int32),
        "seg_b": np.zeros(pg, np.int32),
        "mask": np.zeros(pg),
        "hermite_t0_a": np.zeros(pg),
        "hermite_t1_a": np.zeros(pg),
        "chord_t_a": np.zeros(pg),
        "hermite_t0_b": np.zeros(pg),
        "hermite_t1_b": np.zeros(pg),
        "chord_t_b": np.zeros(pg),
        "d_req": np.zeros(pg),
        "sign": np.ones(pg),
    }
    j = 0
    for p in narrow_grp:
        sa, sb = idx[(p.group_a, p.seg_a)], idx[(p.group_b, p.seg_b)]
        for h in p.hits:
            if j >= pg:
                break
            grp["seg_a"][j] = sa
            grp["seg_b"][j] = sb
            grp["mask"][j] = 1.0
            grp["hermite_t0_a"][j] = h.hermite_t0_a
            grp["hermite_t1_a"][j] = h.hermite_t1_a
            grp["chord_t_a"][j] = h.chord_t_a
            grp["hermite_t0_b"][j] = h.hermite_t0_b
            grp["hermite_t1_b"][j] = h.hermite_t1_b
            grp["chord_t_b"][j] = h.chord_t_b
            grp["d_req"][j] = h.d_req
            grp["sign"][j] = h.sign
            j += 1
        if j >= pg:
            break

    return {k: xp.asarray(v) for k, v in obs.items()}, {k: xp.asarray(v) for k, v in grp.items()}


def clear_capsule_packed(
    x,
    obs_buf: dict[str, Any],
    grp_buf: dict[str, Any],
    layout: PackLayout,
    mu_obs: float = 1.0,
    mu_grp: float = 1.0,
    soft_sigma: float = 1e-2,
    backend: str = "numpy",
):
    """Soft J_clear from packed buffers; fixed shapes → JIT-stable when buf values change."""
    xp = _backend(backend)
    ends = layout.ends_from_x(x, backend=backend)
    j = xp.asarray(0.0)

    if layout.p_max_obs > 0:
        E = ends[obs_buf["seg_idx"]]
        p0, v0, p1, v1 = E[:, 0], E[:, 1], E[:, 2], E[:, 3]
        tc = obs_buf["chord_t"][:, None]
        qa = (1.0 - tc) * eval_hermite(
            p0, v0, p1, v1, obs_buf["hermite_t0"], backend=backend
        ) + tc * eval_hermite(p0, v0, p1, v1, obs_buf["hermite_t1"], backend=backend)
        d_geom = obs_buf["sign"] * xp.linalg.norm(qa - obs_buf["qb"], axis=-1)
        j = j + mu_obs * xp.sum(
            obs_buf["mask"] * phi(d_geom, obs_buf["d_req"], soft_sigma=soft_sigma, backend=backend)
        )

    if layout.p_max_grp > 0:
        Ea, Eb = ends[grp_buf["seg_a"]], ends[grp_buf["seg_b"]]
        a0, av0, a1, av1 = Ea[:, 0], Ea[:, 1], Ea[:, 2], Ea[:, 3]
        b0, bv0, b1, bv1 = Eb[:, 0], Eb[:, 1], Eb[:, 2], Eb[:, 3]
        ta = grp_buf["chord_t_a"][:, None]
        tb = grp_buf["chord_t_b"][:, None]
        qa = (1.0 - ta) * eval_hermite(
            a0, av0, a1, av1, grp_buf["hermite_t0_a"], backend=backend
        ) + ta * eval_hermite(a0, av0, a1, av1, grp_buf["hermite_t1_a"], backend=backend)
        qb = (1.0 - tb) * eval_hermite(
            b0, bv0, b1, bv1, grp_buf["hermite_t0_b"], backend=backend
        ) + tb * eval_hermite(b0, bv0, b1, bv1, grp_buf["hermite_t1_b"], backend=backend)
        d_geom = grp_buf["sign"] * xp.linalg.norm(qa - qb, axis=-1)
        j = j + mu_grp * xp.sum(
            grp_buf["mask"] * phi(d_geom, grp_buf["d_req"], soft_sigma=soft_sigma, backend=backend)
        )
    return j


# ---------------------------------------------------------------------------
# Fixed-shape hull clear (same ends gather; frozen weights/qb/shift)
# ---------------------------------------------------------------------------

def pack_hull_active(
    active: tuple[list, list],
    layout: PackLayout,
    backend: str = "numpy",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Flatten hull narrow pairs into fixed obs/grp buffers (truncate at ``p_max_*``)."""
    xp = _backend(backend)
    narrow_obs, narrow_grp = active
    po, pg = layout.p_max_obs, layout.p_max_grp
    idx = layout.seg_index

    obs = {
        "seg_idx": np.zeros(po, np.int32),
        "mask": np.zeros(po),
        "hermite_t0": np.zeros(po),
        "hermite_t1": np.zeros(po),
        "weights": np.zeros((po, 4)),
        "qb": np.zeros((po, 3)),
        "shift": np.zeros((po, 4, 3)),
        "d_req": np.zeros(po),
        "sign": np.ones(po),
    }
    for i, p in enumerate(narrow_obs):
        if i >= po:
            break
        obs["seg_idx"][i] = idx[(p.group_id, p.seg_id)]
        obs["mask"][i] = 1.0
        obs["hermite_t0"][i] = p.hermite_t0
        obs["hermite_t1"][i] = p.hermite_t1
        obs["weights"][i] = np.asarray(p.weights, float).reshape(4)
        obs["qb"][i] = np.asarray(p.qb, float).reshape(3)
        obs["shift"][i] = np.asarray(p.shift, float).reshape(4, 3)
        obs["d_req"][i] = p.d_req
        obs["sign"][i] = p.sign

    grp = {
        "seg_a": np.zeros(pg, np.int32),
        "seg_b": np.zeros(pg, np.int32),
        "mask": np.zeros(pg),
        "hermite_t0_a": np.zeros(pg),
        "hermite_t1_a": np.zeros(pg),
        "hermite_t0_b": np.zeros(pg),
        "hermite_t1_b": np.zeros(pg),
        "weights_a": np.zeros((pg, 4)),
        "weights_b": np.zeros((pg, 4)),
        "shift_a": np.zeros((pg, 4, 3)),
        "shift_b": np.zeros((pg, 4, 3)),
        "d_req": np.zeros(pg),
        "sign": np.ones(pg),
    }
    for j, p in enumerate(narrow_grp):
        if j >= pg:
            break
        grp["seg_a"][j] = idx[(p.group_a, p.seg_a)]
        grp["seg_b"][j] = idx[(p.group_b, p.seg_b)]
        grp["mask"][j] = 1.0
        grp["hermite_t0_a"][j] = p.hermite_t0_a
        grp["hermite_t1_a"][j] = p.hermite_t1_a
        grp["hermite_t0_b"][j] = p.hermite_t0_b
        grp["hermite_t1_b"][j] = p.hermite_t1_b
        grp["weights_a"][j] = np.asarray(p.weights_a, float).reshape(4)
        grp["weights_b"][j] = np.asarray(p.weights_b, float).reshape(4)
        grp["shift_a"][j] = np.asarray(p.shift_a, float).reshape(4, 3)
        grp["shift_b"][j] = np.asarray(p.shift_b, float).reshape(4, 3)
        grp["d_req"][j] = p.d_req
        grp["sign"][j] = p.sign

    return {k: xp.asarray(v) for k, v in obs.items()}, {k: xp.asarray(v) for k, v in grp.items()}


def clear_hull_packed(
    x,
    obs_buf: dict[str, Any],
    grp_buf: dict[str, Any],
    layout: PackLayout,
    mu_obs: float = 1.0,
    mu_grp: float = 1.0,
    soft_sigma: float = 1e-2,
    backend: str = "numpy",
):
    """Soft J_clear from packed hull buffers; fixed shapes → JIT-stable."""
    xp = _backend(backend)
    ends = layout.ends_from_x(x, backend=backend)
    j = xp.asarray(0.0)

    if layout.p_max_obs > 0:
        E = ends[obs_buf["seg_idx"]]
        p0, v0, p1, v1 = E[:, 0], E[:, 1], E[:, 2], E[:, 3]
        C_sub = hermite_bezier_on_interval(
            p0, v0, p1, v1, obs_buf["hermite_t0"], obs_buf["hermite_t1"], backend=backend
        )
        qa = xp.einsum("pi,pij->pj", obs_buf["weights"], C_sub + obs_buf["shift"])
        d_geom = obs_buf["sign"] * xp.linalg.norm(qa - obs_buf["qb"], axis=-1)
        j = j + mu_obs * xp.sum(
            obs_buf["mask"] * phi(d_geom, obs_buf["d_req"], soft_sigma=soft_sigma, backend=backend)
        )

    if layout.p_max_grp > 0:
        Ea, Eb = ends[grp_buf["seg_a"]], ends[grp_buf["seg_b"]]
        a0, av0, a1, av1 = Ea[:, 0], Ea[:, 1], Ea[:, 2], Ea[:, 3]
        b0, bv0, b1, bv1 = Eb[:, 0], Eb[:, 1], Eb[:, 2], Eb[:, 3]
        Ca_sub = hermite_bezier_on_interval(
            a0, av0, a1, av1, grp_buf["hermite_t0_a"], grp_buf["hermite_t1_a"], backend=backend
        )
        Cb_sub = hermite_bezier_on_interval(
            b0, bv0, b1, bv1, grp_buf["hermite_t0_b"], grp_buf["hermite_t1_b"], backend=backend
        )
        qa = xp.einsum("pi,pij->pj", grp_buf["weights_a"], Ca_sub + grp_buf["shift_a"])
        qb = xp.einsum("pi,pij->pj", grp_buf["weights_b"], Cb_sub + grp_buf["shift_b"])
        d_geom = grp_buf["sign"] * xp.linalg.norm(qa - qb, axis=-1)
        j = j + mu_grp * xp.sum(
            grp_buf["mask"] * phi(d_geom, grp_buf["d_req"], soft_sigma=soft_sigma, backend=backend)
        )
    return j

