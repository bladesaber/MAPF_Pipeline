"""§7.2.1 — polyline capsule narrow-phase kernels (numpy).

Capsule = line segment ⊕ ball(r). Returns (d_geom, hits); d_req lives on each hit.
ObsContactHit: Hermite ends (hermite_t0/t1), local chord_t, qb on the obstacle.
GrpContactHit: same for both groups (…_a / …_b).
Neighbor chord_t's stabilize AD clearance. Detection outside AD (§7.2).
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
    a, b, r, c, r_obs=0.0, margin=0.0, *, hermite_t0: float = 0.0, hermite_t1: float = 1.0
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
) -> tuple[float, tuple[ObsContactHit, ...]]:
    """Distance from capsule (segment ab, radius r) to a solid AABB [mn, mx].

    Samples n_samp points on ab. d_geom = min gap. Hits = samples with gap < d_req
    (closest first); no extra mid/edge band (n_samp already covers the chord).

    Args:
        a, b: segment endpoints, shape (3,)
        r: tube / capsule radius
        mn, mx: box corners, shape (3,) each
        margin: extra separation δ in d_req = r + margin
        n_samp: samples on ab (inclusive ends)
        hermite_t0, hermite_t1: parent Hermite params of chord ends a, b

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
    # hits: samples with gap < d_req only; closest first (empty if all >= d_req)
    if d_geom >= d_req:
        return d_geom, ()

    ht0, ht1 = float(hermite_t0), float(hermite_t1)
    order = [ibest] + [i for i in range(n_samp) if i != ibest and gaps[i] < d_req]
    hits = tuple(
        ObsContactHit(ht0, ht1, float(ts[i]), qbs[i].copy(), float(signs[i]), d_req)
        for i in order
    )
    return d_geom, hits


def find_narrowphase_pairs(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle],
    broad_obs: Sequence[BroadObsPair] = (),
    broad_grp: Sequence[BroadGrpPair] = (),
    margin: float = 0.0,
) -> tuple[list[NarrowObsPair], list[NarrowGrpPair]]:
    """Filter broadphase pairs with capsule kernels; keep pairs with non-empty hits.

    Uses ``seg_xyz`` / ``seg_radius`` (``refresh(..., store=True)`` first).
    """
    by_obs = {obs.name: obs for obs in obstacles}
    narrow_obs: list[NarrowObsPair] = []
    narrow_grp: list[NarrowGrpPair] = []

    # (group_id, seg_id) → chords (a, b, r, hermite_t0, hermite_t1)
    chords_by: dict[tuple[str, int], list[tuple]] = {}
    for g in graphs:
        if g.seg_xyz is None or g.seg_radius is None:
            g.refresh(backend="numpy", store=True)
        xyz = np.asarray(g.seg_xyz, dtype=float)
        radius = np.asarray(g.seg_radius, dtype=float)
        T = int(xyz.shape[1])
        ts = np.linspace(0.0, 1.0, T)
        for sid in range(len(g.segments)):
            chords_by[(g.id, sid)] = [
                (
                    xyz[sid, k],
                    xyz[sid, k + 1],
                    float(max(float(radius[sid, k]), float(radius[sid, k + 1]))),
                    float(ts[k]),
                    float(ts[k + 1]),
                )
                for k in range(T - 1)
            ]

    for p in broad_obs:
        obs = by_obs[p.obs_name]
        for a, b, r, ht0, ht1 in chords_by.get((p.group_id, p.seg_id), ()):
            if obs.kind == "sphere":
                o = cast(SphereObs, obs)
                d_geom, hits = capsule_ball(
                    a, b, r, o.c, o.r, margin, hermite_t0=ht0, hermite_t1=ht1
                )
            elif obs.kind == "capsule":
                o = cast(CapsuleObs, obs)
                d_geom, hits = capsule_capsule(
                    a, b, r, o.a, o.b, o.r, margin, hermite_t0=ht0, hermite_t1=ht1
                )
            elif obs.kind == "aabb":
                o = cast(AabbObs, obs)
                d_geom, hits = capsule_aabb(
                    a, b, r, o.mn, o.mx, margin, hermite_t0=ht0, hermite_t1=ht1
                )
            else:
                raise ValueError(f"unknown obstacle kind: {obs.kind!r}")
            if hits:
                narrow_obs.append(
                    NarrowObsPair(
                        p.group_id,
                        p.seg_id,
                        p.obs_name,
                        d_geom,
                        cast(tuple[ObsContactHit, ...], hits),
                    )
                )

    for p in broad_grp:
        chords_a = chords_by.get((p.group_a, p.seg_a), ())
        chords_b = chords_by.get((p.group_b, p.seg_b), ())
        for a, b, ra, ht0_a, ht1_a in chords_a:
            for a2, b2, rb, ht0_b, ht1_b in chords_b:
                d_geom, hits = capsule_capsule(
                    a,
                    b,
                    ra,
                    a2,
                    b2,
                    rb,
                    margin,
                    other_is_obs=False,
                    hermite_t0=ht0_a,
                    hermite_t1=ht1_a,
                    hermite_t0_b=ht0_b,
                    hermite_t1_b=ht1_b,
                )
                if hits:
                    narrow_grp.append(
                        NarrowGrpPair(
                            p.group_a,
                            p.seg_a,
                            p.group_b,
                            p.seg_b,
                            d_geom,
                            cast(tuple[GrpContactHit, ...], hits),
                        )
                    )

    return narrow_obs, narrow_grp
