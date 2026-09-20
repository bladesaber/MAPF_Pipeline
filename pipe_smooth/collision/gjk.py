"""§7.2.2 — convex distance via coal GJK/EPA (detection outside AD).

Any pair with to_coal() -> (shape, Transform3s) uses signed_distance.
Witness points lie on coal shape surfaces (hull / box / sphere / capsule).

coal’s Boost.Python stubs are incomplete → basedpyright false positives on ctors.
"""

# pyright: reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Sequence

import numpy as np
import coal

from pipe_smooth.geometry.assemble import OptimizeGraph
from pipe_smooth.geometry.hermite import bezier_on_interval, hermite_bezier_controls
from pipe_smooth.collision.obstacles import AabbObs, CapsuleObs, SphereObs
from pipe_smooth.collision.utils import BroadGrpPair, BroadObsPair

_EPS = 1e-10

Obstacle = SphereObs | CapsuleObs | AabbObs


@dataclass(frozen=True)
class NarrowHullObsPair:
    """One active Hull4↔obstacle contact after GJK (one sub-hull; flat, no nested hit).

    qa ≈ weights @ (C_true + shift) on the sub-cubic; qb on the obstacle — freeze for AD/SQP.
    ``sign``: +1 separated / outside; -1 penetrating (d_geom < 0).
    """

    group_id: str
    seg_id: str
    obs_name: str
    d_geom: float
    d_req: float
    hermite_t0: float  # parent Hermite param at sub-hull start, [0, 1]
    hermite_t1: float  # parent Hermite param at sub-hull end, [0, 1]
    weights: np.ndarray  # (4,)
    qb: np.ndarray  # (3,)
    sign: float  # +1 outside / separated; -1 overlap
    shift: np.ndarray  # (4, 3); freeze with weights for qa = λ @ (C + shift)


@dataclass(frozen=True)
class NarrowHullGrpPair:
    """One active Hull4↔Hull4 cross-group contact after GJK (flat, no nested hit)."""

    group_a: str
    seg_a: str
    group_b: str
    seg_b: str
    d_geom: float
    d_req: float
    hermite_t0_a: float
    hermite_t1_a: float
    weights_a: np.ndarray  # (4,)
    hermite_t0_b: float
    hermite_t1_b: float
    weights_b: np.ndarray  # (4,)
    sign: float  # +1 separated; -1 overlap
    shift_a: np.ndarray  # (4, 3)
    shift_b: np.ndarray  # (4, 3)


@dataclass
class Hull4:
    """Four-point convex hull (e.g. Hermite → Bézier controls). Optional r for d_req.

    ``pts`` are the verts used by GJK/weights (= true controls + ``shift``).
    ``shift`` is zero unless near-colinear/coplanar (tiny thicken for coal).
    Freeze AD as qa ≈ λ @ (C_true + shift) with shift held constant.
    """

    kind: ClassVar[str] = "hull4"
    pts: np.ndarray  # (4, 3) — coal / weights verts
    r: float = 0.0
    name: str = ""
    shift: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))

    @classmethod
    def from_pts(cls, pts, r: float = 0.0, name: str = "", eps: float = 1e-5):
        """Build Hull4; if tetra volume ~0, record a tiny shift and thicken ``pts``.

        ``eps`` is relative to chord length (shift ~ eps·L). Keep ≳1e-5 so 6V
        stays above ``_EPS`` and ``to_coal`` uses a tetra (not a flat fan).
        """
        P = np.asarray(pts, float).reshape(4, 3)
        shift = np.zeros((4, 3))
        d = P[3] - P[0]
        L = float(np.linalg.norm(d)) + _EPS
        vol = abs(float(np.dot(np.cross(P[1] - P[0], P[2] - P[0]), P[3] - P[0])))
        if vol <= 1e-12 * (L ** 3):
            # Orthonormal frame on the chord: u along P0→P3; n,b ⊥ u.
            u = d / L
            a = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            n = np.cross(u, a)
            n = n / (np.linalg.norm(n) + _EPS)
            b = np.cross(u, n)
            b = b / (np.linalg.norm(b) + _EPS)
            # Nudge only mid controls. Pure ±n can stay coplanar (area>0, vol=0);
            # add a bit of b on C2 so the tetra has non-zero volume.
            shift[1] = eps * L * n
            shift[2] = -eps * L * n + eps * L * b
        return cls(pts=P + shift, r=r, name=name, shift=shift)

    def aabb(self, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        P = np.asarray(self.pts, float).reshape(4, 3)
        e = float(self.r) + float(margin)
        return P.min(axis=0) - e, P.max(axis=0) + e

    def to_coal(self):
        """coal.Convex tetra / fan from four vertices (identity pose)."""
        pts = np.asarray(self.pts, dtype=float).reshape(-1, 3)
        assert len(pts) == 4, "pts must be a list of 4 points"
        _, idx = np.unique(np.round(pts / _EPS), axis=0, return_index=True)
        if len(idx) != len(pts):
            raise ValueError(f"duplicate points in pts: {len(idx)} != {len(pts)}")
        pts = pts[np.sort(idx)]
        n = len(pts)
        vecs, tris = coal.StdVec_Vec3s(), coal.StdVec_Triangle()
        for p in pts:
            vecs.append(p)
        vol = abs(
            float(np.dot(np.cross(pts[1] - pts[0], pts[2] - pts[0]), pts[3] - pts[0]))
        )
        if vol > _EPS:
            for ijk in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                tris.append(coal.Triangle(*ijk))
        else:
            for k in range(1, n - 1):
                tris.append(coal.Triangle(0, k, k + 1))
        return coal.Convex(vecs, tris), coal.Transform3s.Identity()

    def weights(self, q) -> np.ndarray:
        """Convex combo λ (4,) with q ≈ Σ λ_i pts[i], λ≥0, Σλ=1 (freeze for AD/SQP).

        Tries tetra faces (0,1,2),(0,1,3),(0,2,3),(1,2,3); picks best barycentric fit.
        """
        C = np.asarray(self.pts, float).reshape(4, 3)
        q = np.asarray(q, float).reshape(3)
        best_w, best_err = None, np.inf
        for i, j, k in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            a, b, c = C[i], C[j], C[k]
            n = np.cross(b - a, c - a)
            n2 = float(n @ n) + _EPS
            # project q onto plane; signed areas → barycentric
            w_a = float(np.dot(np.cross(b - q, c - q), n) / n2)
            w_b = float(np.dot(np.cross(c - q, a - q), n) / n2)
            w_c = float(np.dot(np.cross(a - q, b - q), n) / n2)
            if min(w_a, w_b, w_c) < -1e-8:
                continue
            w = np.zeros(4)
            w[i], w[j], w[k] = w_a, w_b, w_c
            s = float(w.sum())
            if s < _EPS:
                continue
            w /= s
            err = float(np.linalg.norm(w @ C - q))
            if err < best_err:
                best_err, best_w = err, w
        if best_w is None:
            raise ValueError(f"Hull4.weights: point {q} not on any face of pts")
        return best_w


def signed_distance(s1, T1, s2, T2):
    """Signed GJK/EPA distance and closest points (world frames T1, T2).

    Args:
        s1, s2: coal collision geometries
        T1, T2: Transform3s for each shape

    Returns:
        d_geom: >0 separated, <0 overlapping (penetration as negative gap)
        q_a, q_b: nearest points on shape1 / shape2, shape (3,)
    """
    req = coal.DistanceRequest()
    req.enable_signed_distance = True  # type: ignore
    res = coal.DistanceResult()
    coal.distance(s1, T1, s2, T2, req, res)
    d = res.min_distance
    d_geom = float(0.0 if d is None else d)
    q_a = np.asarray(res.getNearestPoint1(), dtype=float).reshape(3)
    q_b = np.asarray(res.getNearestPoint2(), dtype=float).reshape(3)
    return d_geom, q_a, q_b


def find_narrowphase_pairs(
    graphs: Sequence[OptimizeGraph],
    obstacles: Sequence[Obstacle],
    broad_obs: Sequence[BroadObsPair] = (),
    broad_grp: Sequence[BroadGrpPair] = (),
    margin: float = 0.0,
) -> tuple[list[NarrowHullObsPair], list[NarrowHullGrpPair]]:
    """Filter broadphase pairs with Hull4 GJK; keep active pairs (d_geom < d_req).

    Each Hermite segment → subdivided Bézier Hull4s (``seg.ts_count`` samples ⇒
    n-1 sub-hulls). Obstacles use ``to_coal()`` (native convex Sphere/Capsule/Box).
    """
    obs_coal = {obs.name: obs.to_coal() for obs in obstacles}
    obs_r = {
        obs.name: float(getattr(obs, "r", 0.0)) if obs.kind != "aabb" else 0.0
        for obs in obstacles
    }

    # (group_id, seg_id) → list of (Hull4, hermite_t0, hermite_t1)
    hulls_by: dict[tuple[str, str], list[tuple[Hull4, float, float]]] = {}
    for g in graphs:
        seen: set[str] = set()
        for path in g.usage_paths:
            for i, sid in enumerate(path.segments):
                if sid in seen:
                    continue
                seen.add(sid)
                seg = g.segments[sid]
                nu, nv = g.nodes[seg.u], g.nodes[seg.v]
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
                C = hermite_bezier_controls(p0, v0, s0, p1, v1, s1, backend="numpy")
                r = max(float(nu.radius), float(nv.radius))
                n = max(2, int(seg.ts_count))
                ts = np.linspace(0.0, 1.0, n)
                hulls_by[(g.id, sid)] = [
                    (
                        Hull4.from_pts(
                            bezier_on_interval(C, float(ts[k]), float(ts[k + 1])),
                            r=r,
                        ),
                        float(ts[k]),
                        float(ts[k + 1]),
                    )
                    for k in range(n - 1)
                ]

    narrow_obs: list[NarrowHullObsPair] = []
    narrow_grp: list[NarrowHullGrpPair] = []

    for p in broad_obs:
        s_obs, T_obs = obs_coal[p.obs_name]
        r_o = obs_r[p.obs_name]
        for hull, ht0, ht1 in hulls_by.get((p.group_id, p.seg_id), ()):
            d_geom, qa, qb = signed_distance(*hull.to_coal(), s_obs, T_obs)
            d_req = float(hull.r) + r_o + float(margin)
            if not np.isfinite(d_geom) or d_geom >= d_req:
                continue
            sign = 1.0 if d_geom >= 0.0 else -1.0
            narrow_obs.append(
                NarrowHullObsPair(
                    p.group_id,
                    p.seg_id,
                    p.obs_name,
                    d_geom,
                    d_req,
                    ht0,
                    ht1,
                    hull.weights(qa),
                    qb.copy(),
                    sign,
                    hull.shift.copy(),
                )
            )

    for p in broad_grp:
        for ha, ht0_a, ht1_a in hulls_by.get((p.group_a, p.seg_a), ()):
            for hb, ht0_b, ht1_b in hulls_by.get((p.group_b, p.seg_b), ()):
                d_geom, qa, qb = signed_distance(*ha.to_coal(), *hb.to_coal())
                d_req = float(ha.r) + float(hb.r) + float(margin)
                if not np.isfinite(d_geom) or d_geom >= d_req:
                    continue
                sign = 1.0 if d_geom >= 0.0 else -1.0
                narrow_grp.append(
                    NarrowHullGrpPair(
                        p.group_a,
                        p.seg_a,
                        p.group_b,
                        p.seg_b,
                        d_geom,
                        d_req,
                        ht0_a,
                        ht1_a,
                        ha.weights(qa),
                        ht0_b,
                        ht1_b,
                        hb.weights(qb),
                        sign,
                        ha.shift.copy(),
                        hb.shift.copy(),
                    )
                )

    return narrow_obs, narrow_grp
