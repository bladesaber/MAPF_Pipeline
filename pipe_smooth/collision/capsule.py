"""§7.2.1 — polyline capsule narrow-phase kernels (numpy).

Capsule = line segment ⊕ ball(r). Returns (d_geom, hits); d_req lives on each hit.
ObsContactHit: Hermite ends (hermite_t0/t1), local chord_t, qb on the obstacle.
GrpContactHit: same for both groups (…_a / …_b).
Neighbor chord_t's stabilize AD clearance (toggle via ``with_neighbors``).
Detection outside AD (§7.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np

from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.collision.obstacles import AabbObs, CapsuleObs, SphereObs
from pipe_smooth.collision.utils import BroadGrpPair, BroadObsPair

_EPS = 1e-12

Obstacle = SphereObs | CapsuleObs | AabbObs


@dataclass(frozen=True)
class ObsContactHit:
    """Contact on one Hermite sub-chord vs an obstacle witness qb.

    The pipe capsule is the straight chord between Hermite samples
    γ(hermite_t0) and γ(hermite_t1). ``chord_t`` is local on that chord:
    qa = (1-chord_t)·γ(hermite_t0) + chord_t·γ(hermite_t1).

    `sign`: +1 outside / separated (push increases ||qa-qb|| away from obstacle);
    -1 inside AABB (so d_signed = sign*||qa-qb|| is negative; hinge pushes out).
    `d_req`: full required gap r(+r_obs)+margin (keep even when inside / penetrating).
    """

    hermite_t0: float  # parent Hermite param at chord start, [0, 1]
    hermite_t1: float  # parent Hermite param at chord end, [0, 1]
    chord_t: float  # local parameter on the straight chord, [0, 1]
    qb: np.ndarray  # shape (3,)
    sign: float  # +1 outside, -1 inside
    d_req: float  # full clearance gap

    def export(self) -> dict:
        return {
            "hermite_t0": float(self.hermite_t0),
            "hermite_t1": float(self.hermite_t1),
            "chord_t": float(self.chord_t),
            "qb": np.asarray(self.qb, float).reshape(-1).tolist(),
            "sign": float(self.sign),
            "d_req": float(self.d_req),
        }


@dataclass(frozen=True)
class GrpContactHit:
    """Cross-group contact on two Hermite sub-chords.

    Seg_a chord = γ_a(hermite_t0_a)→γ_a(hermite_t1_a); likewise for b.
    ``chord_t_a`` / ``chord_t_b`` are local on those chords (frozen for AD).
    `sign`: +1 separated (same convention as ObsContactHit outside).
    `d_req`: full required gap (keep even when penetrating).
    """

    hermite_t0_a: float  # Hermite param at chord start on seg_a
    hermite_t1_a: float  # Hermite param at chord end on seg_a
    chord_t_a: float  # local chord param on seg_a, [0, 1]
    hermite_t0_b: float  # Hermite param at chord start on seg_b
    hermite_t1_b: float  # Hermite param at chord end on seg_b
    chord_t_b: float  # local chord param on seg_b, [0, 1]
    sign: float  # +1 outside
    d_req: float  # full clearance gap

    def export(self) -> dict:
        return {
            "hermite_t0_a": float(self.hermite_t0_a),
            "hermite_t1_a": float(self.hermite_t1_a),
            "chord_t_a": float(self.chord_t_a),
            "hermite_t0_b": float(self.hermite_t0_b),
            "hermite_t1_b": float(self.hermite_t1_b),
            "chord_t_b": float(self.chord_t_b),
            "sign": float(self.sign),
            "d_req": float(self.d_req),
        }


@dataclass(frozen=True)
class NarrowObsPair:
    """Active segment–obstacle pair after capsule narrowphase."""

    group_id: str
    seg_id: int
    obs_name: str
    d_geom: float
    hits: tuple[ObsContactHit, ...] = ()  # closest (+ neighbors); empty if unset

    def export(self) -> dict:
        return {
            "group_id": self.group_id,
            "seg_id": int(self.seg_id),
            "obs_name": self.obs_name,
            "d_geom": float(self.d_geom),
            "hits": [h.export() for h in self.hits],
        }


@dataclass(frozen=True)
class NarrowGrpPair:
    """Active cross-group segment–segment pair after capsule narrowphase."""

    group_a: str
    seg_a: int
    group_b: str
    seg_b: int
    d_geom: float
    hits: tuple[GrpContactHit, ...] = ()  # closest (+ neighbors)

    def export(self) -> dict:
        return {
            "group_a": self.group_a,
            "seg_a": int(self.seg_a),
            "group_b": self.group_b,
            "seg_b": int(self.seg_b),
            "d_geom": float(self.d_geom),
            "hits": [h.export() for h in self.hits],
        }


def capsule_ball(
    a,
    b,
    r,
    c,
    r_obs=0.0,
    margin=0.0,
    *,
    hermite_t0: float = 0.0,
    hermite_t1: float = 1.0,
    with_neighbors: bool = False,
) -> tuple[float, tuple[ObsContactHit, ...]]:
    """Distance from capsule (segment ab, radius r) to a ball (center c, radius r_obs).

    Projects c onto ab. Neighbor hits at t±Δt where Δs = sqrt(max(0, d_req² - d_geom²))
    (chord span still inside the d_req ball), Δt = Δs / ||b-a||. qb is the ball center.

    Args:
        a, b: segment endpoints, shape (3,)
        r: tube / capsule radius
        c: obstacle center, shape (3,)
        r_obs: obstacle ball radius (0 for a point)
        margin: extra separation δ in d_req = r + r_obs + margin
        hermite_t0, hermite_t1: parent Hermite params of chord ends a, b
        with_neighbors: if True, also emit mid/edge band samples (±dt/2, ±dt);
            if False, only the closest hit

    Returns:
        d_geom, hits (tuple of ObsContactHit; hits[0] is closest; d_req on each hit)
    """
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    d = b - a
    L2 = float(d @ d) + _EPS
    L = float(np.sqrt(L2))
    t0 = float(np.clip(((c - a) @ d) / L2, 0.0, 1.0))
    qa = a + t0 * d
    qb = c.copy()
    d_geom = float(np.linalg.norm(qa - qb))
    d_req = float(r) + float(r_obs) + float(margin)

    # No hits outside the d_req band (same rule as capsule_aabb).
    if d_geom >= d_req:
        return d_geom, ()

    ts = [t0]
    # Along-chord half-width where dist to ball center stays ≤ d_req:
    # d(Δs)^2 = d_geom^2 + Δs^2 ≤ d_req^2  ⇒  Δs = sqrt(d_req^2 - d_geom^2);
    # chord param step dt = Δs / L; hits at t0, t0±dt/2, t0±dt.
    if with_neighbors:
        dt = float(np.sqrt(max(0.0, d_req * d_req - d_geom * d_geom))) / L
        if dt > 1e-9:
            # neighbors: mid (±dt/2) and band edge (±dt); closest t0 already in ts
            for s in (-1.0, 1.0):
                for frac in (0.5, 1.0):
                    tk = float(np.clip(t0 + s * frac * dt, 0.0, 1.0))
                    if all(abs(tk - t) > 1e-9 for t in ts):
                        ts.append(tk)
    ht0, ht1 = float(hermite_t0), float(hermite_t1)
    hits = tuple(
        ObsContactHit(ht0, ht1, t, qb.copy(), 1.0, d_req) for t in ts
    )
    return d_geom, hits


def capsule_capsule(
    a,
    b,
    r,
    a2,
    b2,
    r2,
    margin=0.0,
    *,
    other_is_obs: bool = True,
    hermite_t0: float = 0.0,
    hermite_t1: float = 1.0,
    hermite_t0_b: float = 0.0,
    hermite_t1_b: float = 1.0,
    with_neighbors: bool = True,
) -> tuple[float, tuple[ObsContactHit, ...] | tuple[GrpContactHit, ...]]:
    """Distance between two capsules (segments ab / a2b2 with radii r / r2).

    Narrow-phase §7.2.1 Steps A–C. Neighbors like capsule_ball (t±dt/2, ±dt) along
    ab; each sample re-projects onto a2b2.

    Hit type depends on ``other_is_obs``:
      - True (obstacle capsule): ObsContactHit — freeze world qb.
      - False (Hermite peer): GrpContactHit — freeze both chords' params.

    Args:
        a, b: first segment endpoints, shape (3,)
        r: first capsule radius
        a2, b2: second segment endpoints, shape (3,)
        r2: second capsule radius
        margin: extra separation δ
        other_is_obs: True if (a2,b2,r2) is a static CapsuleObs; False if another
            group's Hermite chord
        hermite_t0, hermite_t1: parent Hermite params of chord ends a, b
        hermite_t0_b, hermite_t1_b: parent Hermite params of a2, b2 (GrpContactHit only)
        with_neighbors: if True, also emit mid/edge band samples (±dt/2, ±dt);
            if False, only the closest hit

    Returns:
        d_geom, hits (empty if inactive; hits[0] is closest; d_req on each hit)
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a2 = np.asarray(a2, float)
    b2 = np.asarray(b2, float)
    d, d2, w = b - a, b2 - a2, a - a2
    A, B, C = float(d @ d), float(d @ d2), float(d2 @ d2)
    D, E = float(d @ w), float(d2 @ w)
    denom = A * C - B * B
    L = float(np.sqrt(A + _EPS))

    def _closest_on_seg(p, sa, sb):
        """Closest point on segment sa–sb to p. Returns (t, q, dist2)."""
        dd = sb - sa
        denom_s = float(dd @ dd) + _EPS
        t = float(np.clip(((p - sa) @ dd) / denom_s, 0.0, 1.0))
        q = sa + t * dd
        e = p - q
        return t, q, float(e @ e)

    def _from_endpoints():
        # four endpoint–segment candidates → (alpha, beta, dist2)
        cands = []
        t, q, d2v = _closest_on_seg(a, a2, b2)
        cands.append((0.0, t, d2v))
        t, q, d2v = _closest_on_seg(b, a2, b2)
        cands.append((1.0, t, d2v))
        t, q, d2v = _closest_on_seg(a2, a, b)
        cands.append((t, 0.0, d2v))
        t, q, d2v = _closest_on_seg(b2, a, b)
        cands.append((t, 1.0, d2v))
        return min(cands, key=lambda x: x[2])

    if denom <= _EPS:
        alpha, beta, d2min = _from_endpoints()
    else:
        a0 = (B * E - C * D) / denom
        b0 = (A * E - B * D) / denom
        a_in, b_in = (0.0 <= a0 <= 1.0), (0.0 <= b0 <= 1.0)
        if a_in and b_in:
            alpha, beta = a0, b0
            e = (a + alpha * d) - (a2 + beta * d2)
            d2min = float(e @ e)
        elif (not a_in) and b_in:
            alpha = float(np.clip(a0, 0.0, 1.0))
            p = a + alpha * d
            beta = float(np.clip(((p - a2) @ d2) / (C + _EPS), 0.0, 1.0))
            e = p - (a2 + beta * d2)
            d2min = float(e @ e)
        elif a_in and (not b_in):
            beta = float(np.clip(b0, 0.0, 1.0))
            p2 = a2 + beta * d2
            alpha = float(np.clip(((p2 - a) @ d) / (A + _EPS), 0.0, 1.0))
            e = (a + alpha * d) - p2
            d2min = float(e @ e)
        else:
            alpha, beta, d2min = _from_endpoints()

    d_geom = float(np.sqrt(d2min + _EPS))
    d_req = float(r) + float(r2) + float(margin)
    if d_geom >= d_req:
        return d_geom, ()

    t0 = float(alpha)
    ts = [t0]
    if with_neighbors:
        dt = float(np.sqrt(max(0.0, d_req * d_req - d_geom * d_geom))) / L
        if dt > 1e-9:
            for s in (-1.0, 1.0):
                for frac in (0.5, 1.0):
                    tk = float(np.clip(t0 + s * frac * dt, 0.0, 1.0))
                    if all(abs(tk - t) > 1e-9 for t in ts):
                        ts.append(tk)

    sgn = 1.0
    ht0, ht1 = float(hermite_t0), float(hermite_t1)
    if other_is_obs:
        hits_obs: list[ObsContactHit] = []
        for t in ts:
            qa = a + t * d
            _, qb, _ = _closest_on_seg(qa, a2, b2)
            hits_obs.append(ObsContactHit(ht0, ht1, t, qb.copy(), sgn, d_req))
        return d_geom, tuple(hits_obs)

    ht0_b, ht1_b = float(hermite_t0_b), float(hermite_t1_b)
    hits_grp: list[GrpContactHit] = []
    for t in ts:
        qa = a + t * d
        chord_t_b, _, _ = _closest_on_seg(qa, a2, b2)
        hits_grp.append(
            GrpContactHit(ht0, ht1, t, ht0_b, ht1_b, float(chord_t_b), sgn, d_req)
        )
    return d_geom, tuple(hits_grp)


def capsule_aabb(
    a,
    b,
    r,
    mn,
    mx,
    margin=0.0,
    n_samp: int = 17,
    *,
    hermite_t0: float = 0.0,
    hermite_t1: float = 1.0,
    max_hits: int | None = 1,
) -> tuple[float, tuple[ObsContactHit, ...]]:
    """Distance from capsule (segment ab, radius r) to a solid AABB [mn, mx].

    Samples n_samp points on ab. d_geom = min gap. Hits = samples with gap < d_req,
    sorted by gap ascending (closest first); no extra mid/edge band.

    Args:
        a, b: segment endpoints, shape (3,)
        r: tube / capsule radius
        mn, mx: box corners, shape (3,) each
        margin: extra separation δ in d_req = r + margin
        n_samp: samples on ab (inclusive ends)
        hermite_t0, hermite_t1: parent Hermite params of chord ends a, b
        max_hits: if set, keep at most this many hits after sorting by gap;
            ``None`` keeps all active samples

    Returns:
        d_geom, hits (tuple of ObsContactHit; hits[0] is closest; d_req on each hit)
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    mn, mx = np.asarray(mn, float), np.asarray(mx, float)

    def box_qb(q):
        """Witness on AABB for q; signed gap (>0 out, <0 in) and force sign."""
        closest = np.clip(q, mn, mx)
        if np.any(q < mn) or np.any(q > mx):
            return float(np.linalg.norm(q - closest)), closest, 1.0
        to_lo, to_hi = q - mn, mx - q
        face = int(np.argmin(np.concatenate([to_lo, to_hi])))
        gap = -float(np.min(np.minimum(to_lo, to_hi)))
        qb = q.copy()
        if face < 3:
            qb[face] = mn[face]
        else:
            qb[face - 3] = mx[face - 3]
        return gap, qb, -1.0  # inside: sign=-1 so clearance pushes outward

    d = b - a
    d_req = float(r) + float(margin)
    n_samp = max(int(n_samp), 2)
    ts = np.linspace(0.0, 1.0, n_samp)
    gaps = np.empty(n_samp)
    qbs = []
    signs = []
    for i, t in enumerate(ts):
        gaps[i], qb, sgn = box_qb(a + t * d)
        qbs.append(qb)
        signs.append(sgn)

    ibest = int(np.argmin(gaps))
    d_geom = float(gaps[ibest])
    # hits: samples with gap < d_req only; sorted by gap (empty if all >= d_req)
    if d_geom >= d_req:
        return d_geom, ()

    order = sorted((i for i in range(n_samp) if gaps[i] < d_req), key=lambda i: gaps[i])
    if max_hits is not None:
        order = order[: max(int(max_hits), 0)]
    ht0, ht1 = float(hermite_t0), float(hermite_t1)
    hits = tuple(
        ObsContactHit(ht0, ht1, float(ts[i]), qbs[i].copy(), float(signs[i]), d_req)
        for i in order
    )
    return d_geom, hits


# --------------------------------
# Batched versions
# --------------------------------

def capsule_ball_batch(
    a,
    b,
    r,
    c,
    r_obs=0.0,
    margin=0.0,
):
    """Batched ``capsule_ball`` (closest hit only).

    Args:
        a, b: segment endpoints, shape (B, 3) or (3,)
        r: tube / capsule radius, scalar or (B,)
        c: obstacle center, shape (B, 3) or (3,)
        r_obs: obstacle ball radius, scalar or (B,)
        margin: extra separation δ in d_req = r + r_obs + margin, scalar or (B,)

    Returns:
        d_geom (B,), active (B,), chord_t (B,), qb (B, 3), d_req (B,), sign (B,)
    """
    a = np.asarray(a, float).reshape(-1, 3)
    b = np.asarray(b, float).reshape(-1, 3)
    c = np.asarray(c, float).reshape(-1, 3)
    B = int(a.shape[0])
    r = np.broadcast_to(np.asarray(r, float).reshape(-1), (B,))
    r_obs = np.broadcast_to(np.asarray(r_obs, float).reshape(-1), (B,))
    margin = np.broadcast_to(np.asarray(margin, float).reshape(-1), (B,))

    d = b - a
    L2 = np.sum(d * d, axis=-1) + _EPS
    t0 = np.clip(np.sum((c - a) * d, axis=-1) / L2, 0.0, 1.0)
    qa = a + t0[:, None] * d
    qb = np.array(c, copy=True)
    d_geom = np.linalg.norm(qa - qb, axis=-1)
    d_req = r + r_obs + margin
    active = d_geom < d_req
    sign = np.ones(B, dtype=float)
    return d_geom, active, t0, qb, d_req, sign


def capsule_capsule_batch(
    a,
    b,
    r,
    a2,
    b2,
    r2,
    margin=0.0,
    *,
    other_is_obs: bool = True,
):
    """Batched ``capsule_capsule`` (closest hit only).

    Args:
        a, b: first segment endpoints, shape (B, 3) or (3,)
        r: first capsule radius, scalar or (B,)
        a2, b2: second segment endpoints, shape (B, 3) or (3,)
        r2: second capsule radius, scalar or (B,)
        margin: extra separation δ, scalar or (B,)
        other_is_obs: True if (a2,b2,r2) is a static CapsuleObs; False if Hermite peer

    Returns:
        d_geom (B,), active (B,), then hit params:
        - obs: chord_t (B,), qb (B, 3), d_req (B,), sign (B,)
        - grp: chord_t_a (B,), chord_t_b (B,), d_req (B,), sign (B,)
    """
    a = np.asarray(a, float).reshape(-1, 3)
    b = np.asarray(b, float).reshape(-1, 3)
    a2 = np.asarray(a2, float).reshape(-1, 3)
    b2 = np.asarray(b2, float).reshape(-1, 3)
    B = int(a.shape[0])
    r = np.broadcast_to(np.asarray(r, float).reshape(-1), (B,))
    r2 = np.broadcast_to(np.asarray(r2, float).reshape(-1), (B,))
    margin = np.broadcast_to(np.asarray(margin, float).reshape(-1), (B,))

    d, d2, w = b - a, b2 - a2, a - a2
    A = np.sum(d * d, axis=-1)
    Bdot = np.sum(d * d2, axis=-1)
    C = np.sum(d2 * d2, axis=-1)
    D = np.sum(d * w, axis=-1)
    E = np.sum(d2 * w, axis=-1)
    denom = A * C - Bdot * Bdot

    def closest_t_q_d2(p, sa, sb):
        dd = sb - sa
        den = np.sum(dd * dd, axis=-1) + _EPS
        t = np.clip(np.sum((p - sa) * dd, axis=-1) / den, 0.0, 1.0)
        q = sa + t[:, None] * dd
        e = p - q
        return t, q, np.sum(e * e, axis=-1)

    # endpoint fallback candidates
    t_on_2_a, _, d2_0 = closest_t_q_d2(a, a2, b2)
    t_on_2_b, _, d2_1 = closest_t_q_d2(b, a2, b2)
    t_on_1_a2, _, d2_2 = closest_t_q_d2(a2, a, b)
    t_on_1_b2, _, d2_3 = closest_t_q_d2(b2, a, b)
    d2_ep = np.stack([d2_0, d2_1, d2_2, d2_3], axis=0)
    i_ep = np.argmin(d2_ep, axis=0)
    ar = np.arange(B)
    alpha_ep = np.stack(
        [np.zeros(B), np.ones(B), t_on_1_a2, t_on_1_b2], axis=0
    )[i_ep, ar]
    beta_ep = np.stack(
        [t_on_2_a, t_on_2_b, np.zeros(B), np.ones(B)], axis=0
    )[i_ep, ar]
    d2_ep_min = d2_ep[i_ep, ar]

    parallel = denom <= _EPS
    denom_safe = np.where(parallel, 1.0, denom)
    a0 = (Bdot * E - C * D) / denom_safe
    b0 = (A * E - Bdot * D) / denom_safe
    a_in = (0.0 <= a0) & (a0 <= 1.0)
    b_in = (0.0 <= b0) & (b0 <= 1.0)

    alpha = alpha_ep.copy()
    beta = beta_ep.copy()
    d2min = d2_ep_min.copy()

    both = (~parallel) & a_in & b_in
    alpha = np.where(both, a0, alpha)
    beta = np.where(both, b0, beta)
    e = (a + alpha[:, None] * d) - (a2 + beta[:, None] * d2)
    d2min = np.where(both, np.sum(e * e, axis=-1), d2min)

    a_out_b_in = (~parallel) & (~a_in) & b_in
    alpha_c = np.clip(a0, 0.0, 1.0)
    p = a + alpha_c[:, None] * d
    beta_c = np.clip(np.sum((p - a2) * d2, axis=-1) / (C + _EPS), 0.0, 1.0)
    e = p - (a2 + beta_c[:, None] * d2)
    alpha = np.where(a_out_b_in, alpha_c, alpha)
    beta = np.where(a_out_b_in, beta_c, beta)
    d2min = np.where(a_out_b_in, np.sum(e * e, axis=-1), d2min)

    a_in_b_out = (~parallel) & a_in & (~b_in)
    beta_c = np.clip(b0, 0.0, 1.0)
    p2 = a2 + beta_c[:, None] * d2
    alpha_c = np.clip(np.sum((p2 - a) * d, axis=-1) / (A + _EPS), 0.0, 1.0)
    e = (a + alpha_c[:, None] * d) - p2
    alpha = np.where(a_in_b_out, alpha_c, alpha)
    beta = np.where(a_in_b_out, beta_c, beta)
    d2min = np.where(a_in_b_out, np.sum(e * e, axis=-1), d2min)

    # both out or parallel: already endpoint
    d_geom = np.sqrt(d2min + _EPS)
    d_req = r + r2 + margin
    active = d_geom < d_req
    sign = np.ones(B, dtype=float)
    chord_t = alpha
    qa = a + chord_t[:, None] * d
    t_b, qb, _ = closest_t_q_d2(qa, a2, b2)

    if other_is_obs:
        return d_geom, active, chord_t, qb, d_req, sign
    return d_geom, active, chord_t, t_b, d_req, sign


def capsule_aabb_batch(
    a,
    b,
    r,
    mn,
    mx,
    margin=0.0,
    n_samp: int = 17,
    *,
    max_hits: int | None = 1,
):
    """Batched ``capsule_aabb``.

    Args:
        a, b: segment endpoints, shape (B, 3) or (3,)
        r: tube / capsule radius, scalar or (B,)
        mn, mx: box corners, shape (B, 3) or (3,)
        margin: extra separation δ in d_req = r + margin, scalar or (B,)
        n_samp: samples on ab (inclusive ends)
        max_hits: keep at most this many hits after sorting by gap;
            None keeps all active samples

    Returns:
        d_geom (B,), active (B,), chord_t (B, K), qb (B, K, 3),
        sign (B, K), hit_mask (B, K), d_req (B,)
        with K = max_hits (or n_samp if max_hits is None)
    """
    a = np.asarray(a, float).reshape(-1, 3)
    b = np.asarray(b, float).reshape(-1, 3)
    mn = np.asarray(mn, float).reshape(-1, 3)
    mx = np.asarray(mx, float).reshape(-1, 3)
    B = int(a.shape[0])
    r = np.broadcast_to(np.asarray(r, float).reshape(-1), (B,))
    margin = np.broadcast_to(np.asarray(margin, float).reshape(-1), (B,))
    n_samp = max(int(n_samp), 2)
    K = n_samp if max_hits is None else max(int(max_hits), 0)

    d = b - a
    d_req = r + margin
    ts = np.linspace(0.0, 1.0, n_samp)
    q = a[:, None, :] + ts[None, :, None] * d[:, None, :]  # (B, S, 3)
    closest = np.clip(q, mn[:, None, :], mx[:, None, :])
    outside = np.any(q < mn[:, None, :], axis=-1) | np.any(q > mx[:, None, :], axis=-1)
    gap_out = np.linalg.norm(q - closest, axis=-1)
    to_lo = q - mn[:, None, :]
    to_hi = mx[:, None, :] - q
    face = np.argmin(np.concatenate([to_lo, to_hi], axis=-1), axis=-1)  # (B, S)
    gap_in = -np.min(np.minimum(to_lo, to_hi), axis=-1)
    qb_in = np.array(q, copy=True)
    for f in range(3):
        qb_in[:, :, f] = np.where(face == f, mn[:, f][:, None], qb_in[:, :, f])
        qb_in[:, :, f] = np.where(face == f + 3, mx[:, f][:, None], qb_in[:, :, f])
    qb = np.where(outside[..., None], closest, qb_in)
    gaps = np.where(outside, gap_out, gap_in)
    signs = np.where(outside, 1.0, -1.0)

    ibest = np.argmin(gaps, axis=-1)
    d_geom = gaps[np.arange(B), ibest]
    active = d_geom < d_req

    K_out = max(K, 1)
    chord_t = np.zeros((B, K_out))
    qb_out = np.zeros((B, K_out, 3))
    sign_out = np.zeros((B, K_out))
    hit_mask = np.zeros((B, K_out))
    if K == 0:
        return d_geom, active, chord_t, qb_out, sign_out, hit_mask, d_req

    for i in range(B):
        if not active[i]:
            continue
        order = np.argsort(gaps[i])
        order = order[gaps[i, order] < d_req[i]][:K]
        n = int(order.shape[0])
        chord_t[i, :n] = ts[order]
        qb_out[i, :n] = qb[i, order]
        sign_out[i, :n] = signs[i, order]
        hit_mask[i, :n] = 1.0
    return d_geom, active, chord_t, qb_out, sign_out, hit_mask, d_req


# --------------------------------
# Narrowphase
# --------------------------------

def find_narrowphase_pairs_loop(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle],
    broad_obs: Sequence[BroadObsPair] = (),
    broad_grp: Sequence[BroadGrpPair] = (),
    margin: float = 0.0,
    *,
    with_neighbors: bool = False,
    max_hits_aabb: int | None = 1,
) -> tuple[list[NarrowObsPair], list[NarrowGrpPair]]:
    """Scalar-loop narrowphase; keep pairs with non-empty hits.

    Builds chords only for segments in ``broad_*``. Chord AABB midphase runs
    before exact ``capsule_*`` tests.
    ``with_neighbors`` → ball/capsule; ``max_hits_aabb`` → aabb kernel.
    """
    by_obs = {obs.name: obs for obs in obstacles}
    by_g = {g.id: g for g in graphs}
    narrow_obs: list[NarrowObsPair] = []
    narrow_grp: list[NarrowGrpPair] = []
    m = float(margin)

    need: dict[str, set[int]] = {}
    for p in broad_obs:
        need.setdefault(p.group_id, set()).add(p.seg_id)
    for p in broad_grp:
        need.setdefault(p.group_a, set()).add(p.seg_a)
        need.setdefault(p.group_b, set()).add(p.seg_b)

    # (group_id, seg_id) → list of (a, b, r, ht0, ht1, mn, mx); mn/mx padded by r+margin
    chords_by: dict[tuple[str, int], list[tuple]] = {}
    for gid, sids in need.items():
        g = by_g[gid]
        if g.seg_xyz is None or g.seg_radius is None:
            g.refresh(backend="numpy", store=True)
        xyz = np.asarray(g.seg_xyz, dtype=float)
        radius = np.asarray(g.seg_radius, dtype=float)
        T = int(xyz.shape[1])
        ts = np.linspace(0.0, 1.0, T)
        for sid in sids:
            chords = []
            for k in range(T - 1):
                a, b = xyz[sid, k], xyz[sid, k + 1]
                r = float(max(float(radius[sid, k]), float(radius[sid, k + 1])))
                pad = r + m
                chords.append(
                    (
                        a,
                        b,
                        r,
                        float(ts[k]),
                        float(ts[k + 1]),
                        np.minimum(a, b) - pad,
                        np.maximum(a, b) + pad,
                    )
                )
            chords_by[(gid, sid)] = chords

    obs_box = {o.name: o.aabb(margin) for o in obstacles}

    for p in broad_obs:
        obs = by_obs[p.obs_name]
        omn, omx = obs_box[p.obs_name]
        for a, b, r, ht0, ht1, cmn, cmx in chords_by.get((p.group_id, p.seg_id), ()):
            if (
                cmn[0] > omx[0]
                or omn[0] > cmx[0]
                or cmn[1] > omx[1]
                or omn[1] > cmx[1]
                or cmn[2] > omx[2]
                or omn[2] > cmx[2]
            ):
                continue
            if obs.kind == "sphere":
                o = cast(SphereObs, obs)
                d_geom, hits = capsule_ball(
                    a, b, r, o.c, o.r, margin,
                    hermite_t0=ht0, hermite_t1=ht1, with_neighbors=with_neighbors,
                )
            elif obs.kind == "capsule":
                o = cast(CapsuleObs, obs)
                d_geom, hits = capsule_capsule(
                    a, b, r, o.a, o.b, o.r, margin,
                    hermite_t0=ht0, hermite_t1=ht1, with_neighbors=with_neighbors,
                )
            elif obs.kind == "aabb":
                o = cast(AabbObs, obs)
                d_geom, hits = capsule_aabb(
                    a, b, r, o.mn, o.mx, margin,
                    hermite_t0=ht0, hermite_t1=ht1, max_hits=max_hits_aabb,
                )
            else:
                raise ValueError(f"unknown obstacle kind: {obs.kind!r}")
            if hits:
                narrow_obs.append(
                    NarrowObsPair(
                        p.group_id, p.seg_id, p.obs_name, d_geom,
                        cast(tuple[ObsContactHit, ...], hits),
                    )
                )

    for p in broad_grp:
        chords_a = chords_by.get((p.group_a, p.seg_a), ())
        chords_b = chords_by.get((p.group_b, p.seg_b), ())
        for a, b, ra, ht0_a, ht1_a, mn_a, mx_a in chords_a:
            for a2, b2, rb, ht0_b, ht1_b, mn_b, mx_b in chords_b:
                if (
                    mn_a[0] > mx_b[0]
                    or mn_b[0] > mx_a[0]
                    or mn_a[1] > mx_b[1]
                    or mn_b[1] > mx_a[1]
                    or mn_a[2] > mx_b[2]
                    or mn_b[2] > mx_a[2]
                ):
                    continue
                d_geom, hits = capsule_capsule(
                    a, b, ra, a2, b2, rb, margin,
                    other_is_obs=False,
                    hermite_t0=ht0_a, hermite_t1=ht1_a,
                    hermite_t0_b=ht0_b, hermite_t1_b=ht1_b,
                    with_neighbors=with_neighbors,
                )
                if hits:
                    narrow_grp.append(
                        NarrowGrpPair(
                            p.group_a, p.seg_a, p.group_b, p.seg_b, d_geom,
                            cast(tuple[GrpContactHit, ...], hits),
                        )
                    )

    return narrow_obs, narrow_grp


def find_narrowphase_pairs_batch(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle],
    broad_obs: Sequence[BroadObsPair] = (),
    broad_grp: Sequence[BroadGrpPair] = (),
    margin: float = 0.0,
    *,
    max_hits_aabb: int | None = 1,
) -> tuple[list[NarrowObsPair], list[NarrowGrpPair]]:
    """Batched narrowphase (closest hit only; same midphase as ``find_narrowphase_pairs_loop``).

    Args:
        graphs: OptimizeGraph sequence (needs ``seg_xyz`` / ``seg_radius``)
        obstacles: SphereObs | CapsuleObs | AabbObs
        broad_obs, broad_grp: broadphase pairs
        margin: clearance pad
        max_hits_aabb: forwarded to ``capsule_aabb_batch``

    Returns:
        narrow_obs, narrow_grp (same types as ``find_narrowphase_pairs_loop``)
    """
    by_obs = {obs.name: obs for obs in obstacles}
    by_g = {g.id: g for g in graphs}
    m = float(margin)

    need: dict[str, set[int]] = {}
    for p in broad_obs:
        need.setdefault(p.group_id, set()).add(p.seg_id)
    for p in broad_grp:
        need.setdefault(p.group_a, set()).add(p.seg_a)
        need.setdefault(p.group_b, set()).add(p.seg_b)

    chords_by: dict[tuple[str, int], list[tuple]] = {}
    for gid, sids in need.items():
        g = by_g[gid]
        if g.seg_xyz is None or g.seg_radius is None:
            g.refresh(backend="numpy", store=True)
        xyz = np.asarray(g.seg_xyz, dtype=float)
        radius = np.asarray(g.seg_radius, dtype=float)
        T = int(xyz.shape[1])
        ts = np.linspace(0.0, 1.0, T)
        for sid in sids:
            chords = []
            for k in range(T - 1):
                a, b = xyz[sid, k], xyz[sid, k + 1]
                r = float(max(float(radius[sid, k]), float(radius[sid, k + 1])))
                pad = r + m
                chords.append(
                    (
                        a,
                        b,
                        r,
                        float(ts[k]),
                        float(ts[k + 1]),
                        np.minimum(a, b) - pad,
                        np.maximum(a, b) + pad,
                    )
                )
            chords_by[(gid, sid)] = chords

    obs_box = {o.name: o.aabb(margin) for o in obstacles}

    # Accumulate AABB-passing candidates, then one batch call per kernel kind.
    ball_a, ball_b, ball_r, ball_c, ball_ro, ball_meta = [], [], [], [], [], []
    cap_a, cap_b, cap_r, cap_a2, cap_b2, cap_r2, cap_meta = [], [], [], [], [], [], []
    box_a, box_b, box_r, box_mn, box_mx, box_meta = [], [], [], [], [], []

    for p in broad_obs:
        obs = by_obs[p.obs_name]
        omn, omx = obs_box[p.obs_name]
        for a, b, r, ht0, ht1, cmn, cmx in chords_by.get((p.group_id, p.seg_id), ()):
            if (
                cmn[0] > omx[0]
                or omn[0] > cmx[0]
                or cmn[1] > omx[1]
                or omn[1] > cmx[1]
                or cmn[2] > omx[2]
                or omn[2] > cmx[2]
            ):
                continue
            meta = (p.group_id, p.seg_id, p.obs_name, ht0, ht1)
            if obs.kind == "sphere":
                o = cast(SphereObs, obs)
                ball_a.append(a); ball_b.append(b); ball_r.append(r)
                ball_c.append(o.c); ball_ro.append(o.r); ball_meta.append(meta)
            elif obs.kind == "capsule":
                o = cast(CapsuleObs, obs)
                cap_a.append(a); cap_b.append(b); cap_r.append(r)
                cap_a2.append(o.a); cap_b2.append(o.b); cap_r2.append(o.r)
                cap_meta.append(meta)
            elif obs.kind == "aabb":
                o = cast(AabbObs, obs)
                box_a.append(a); box_b.append(b); box_r.append(r)
                box_mn.append(o.mn); box_mx.append(o.mx); box_meta.append(meta)
            else:
                raise ValueError(f"unknown obstacle kind: {obs.kind!r}")

    narrow_obs: list[NarrowObsPair] = []
    if ball_a:
        d_geom, active, ct, qb, d_req, sign = capsule_ball_batch(
            np.stack(ball_a), np.stack(ball_b), ball_r, np.stack(ball_c), ball_ro, margin  # type: ignore
        )
        for i in np.flatnonzero(active):
            gid, sid, name, ht0, ht1 = ball_meta[i]
            hit = ObsContactHit(
                ht0, ht1, float(ct[i]), np.array(qb[i], copy=True), float(sign[i]), float(d_req[i])
            )
            narrow_obs.append(NarrowObsPair(gid, sid, name, float(d_geom[i]), (hit,)))
    if cap_a:
        d_geom, active, ct, qb, d_req, sign = capsule_capsule_batch(
            np.stack(cap_a), np.stack(cap_b), cap_r,
            np.stack(cap_a2), np.stack(cap_b2), cap_r2, margin, other_is_obs=True,
        )
        for i in np.flatnonzero(active):
            gid, sid, name, ht0, ht1 = cap_meta[i]
            hit = ObsContactHit(
                ht0, ht1, float(ct[i]), np.array(qb[i], copy=True), float(sign[i]), float(d_req[i])
            )
            narrow_obs.append(NarrowObsPair(gid, sid, name, float(d_geom[i]), (hit,)))
    if box_a:
        d_geom, active, ct, qb, sgn, mask, d_req = capsule_aabb_batch(
            np.stack(box_a), np.stack(box_b), box_r,
            np.stack(box_mn), np.stack(box_mx), margin, max_hits=max_hits_aabb,
        )
        for i in np.flatnonzero(active):
            gid, sid, name, ht0, ht1 = box_meta[i]
            hits = tuple(
                ObsContactHit(
                    ht0, ht1, float(ct[i, j]), np.array(qb[i, j], copy=True),
                    float(sgn[i, j]), float(d_req[i]),
                )
                for j in range(mask.shape[1])
                if mask[i, j] > 0.5
            )
            if hits:
                narrow_obs.append(NarrowObsPair(gid, sid, name, float(d_geom[i]), hits))

    ga, gb, ra, ga2, gb2, rb, gmeta = [], [], [], [], [], [], []
    for p in broad_grp:
        chords_a = chords_by.get((p.group_a, p.seg_a), ())
        chords_b = chords_by.get((p.group_b, p.seg_b), ())
        for a, b, r_a, ht0_a, ht1_a, mn_a, mx_a in chords_a:
            for a2, b2, r_b, ht0_b, ht1_b, mn_b, mx_b in chords_b:
                if (
                    mn_a[0] > mx_b[0]
                    or mn_b[0] > mx_a[0]
                    or mn_a[1] > mx_b[1]
                    or mn_b[1] > mx_a[1]
                    or mn_a[2] > mx_b[2]
                    or mn_b[2] > mx_a[2]
                ):
                    continue
                ga.append(a); gb.append(b); ra.append(r_a)
                ga2.append(a2); gb2.append(b2); rb.append(r_b)
                gmeta.append(
                    (p.group_a, p.seg_a, p.group_b, p.seg_b, ht0_a, ht1_a, ht0_b, ht1_b)
                )

    narrow_grp: list[NarrowGrpPair] = []
    if ga:
        d_geom, active, cta, ctb, d_req, sign = capsule_capsule_batch(
            np.stack(ga), np.stack(gb), ra,
            np.stack(ga2), np.stack(gb2), rb, margin, other_is_obs=False,
        )
        for i in np.flatnonzero(active):
            ga_id, sa, gb_id, sb, ht0_a, ht1_a, ht0_b, ht1_b = gmeta[i]
            hit = GrpContactHit(
                ht0_a, ht1_a, float(cta[i]), ht0_b, ht1_b, float(ctb[i]),
                float(sign[i]), float(d_req[i]),
            )
            narrow_grp.append(
                NarrowGrpPair(ga_id, sa, gb_id, sb, float(d_geom[i]), (hit,))
            )

    return narrow_obs, narrow_grp


def find_narrowphase_pairs(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle],
    broad_obs: Sequence[BroadObsPair] = (),
    broad_grp: Sequence[BroadGrpPair] = (),
    margin: float = 0.0,
    *,
    batch: bool = True,
    with_neighbors: bool = False,
    max_hits_aabb: int | None = 1,
) -> tuple[list[NarrowObsPair], list[NarrowGrpPair]]:
    """Narrowphase entry: ``batch=True`` → batched kernels; else scalar loop.

    Args:
        graphs: OptimizeGraph sequence
        obstacles: SphereObs | CapsuleObs | AabbObs
        broad_obs, broad_grp: broadphase pairs
        margin: clearance pad
        batch: if True, use ``find_narrowphase_pairs_batch`` (closest-only)
        with_neighbors: loop path only; incompatible with ``batch=True``
        max_hits_aabb: forwarded to aabb kernels

    Returns:
        narrow_obs, narrow_grp
    """
    if batch:
        if with_neighbors:
            raise ValueError("batch=True does not support with_neighbors=True; set batch=False")
        return find_narrowphase_pairs_batch(
            graphs, obstacles, broad_obs, broad_grp, margin, max_hits_aabb=max_hits_aabb
        )
    return find_narrowphase_pairs_loop(
        graphs,
        obstacles,
        broad_obs,
        broad_grp,
        margin,
        with_neighbors=with_neighbors,
        max_hits_aabb=max_hits_aabb,
    )
