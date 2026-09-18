"""§7.2.2 — convex distance via coal GJK/EPA (detection outside AD).

Hulls bound centerlines; radii only in d_req. Overlap ⇒ d_geom < 0.

coal’s Boost.Python stubs are incomplete → basedpyright false positives on ctors.
"""

# pyright: reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

import numpy as np
import coal

_EPS = 1e-10
_ID = coal.Transform3s.Identity()


def _convex(pts):
    """Build a coal.Convex from exactly four vertices (e.g. Bézier controls).

    Rejects duplicates. Uses a tetrahedron if the four points have volume,
    otherwise a triangle fan (coplanar CH).

    Args:
        pts: vertices, shape (4, 3)

    Returns:
        coal.Convex mesh suitable for GJK/EPA distance queries
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    assert len(pts) == 4, "pts must be a list of 4 points"

    # near-duplicate drop: bin by EPS, keep first of each bin (vectorized)
    _, idx = np.unique(np.round(pts / _EPS), axis=0, return_index=True)
    if len(idx) != len(pts):
        raise ValueError(f"duplicate points in pts: {len(idx)} != {len(pts)}")

    pts = pts[np.sort(idx)]
    n = len(pts)

    vecs, tris = coal.StdVec_Vec3s(), coal.StdVec_Triangle()
    for p in pts:
        vecs.append(p)

    vol = abs(float(np.dot(np.cross(pts[1] - pts[0], pts[2] - pts[0]), pts[3] - pts[0])))
    if vol > _EPS:
        for ijk in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            tris.append(coal.Triangle(*ijk))
    else:
        for k in range(1, n - 1):
            tris.append(coal.Triangle(0, k, k + 1))
    return coal.Convex(vecs, tris)


def _aabb(mn, mx):
    """Axis-aligned box as coal.Box plus world transform at the box center.

    Args:
        mn, mx: AABB corners, shape (3,), mn <= mx per axis

    Returns:
        box: coal.Box with side lengths (mx - mn)
        T: Transform3s translating the box from the origin to 0.5 * (mn + mx)
    """
    mn = np.asarray(mn, dtype=float).reshape(3)
    mx = np.asarray(mx, dtype=float).reshape(3)
    size = np.maximum(mx - mn, _EPS)
    box = coal.Box(float(size[0]), float(size[1]), float(size[2]))
    T = coal.Transform3s.Identity()
    T.setTranslation(0.5 * (mn + mx))
    return box, T


def _dist(s1, T1, s2, T2):
    """Signed GJK/EPA distance between two coal shapes (world frames T1, T2).

    Args:
        s1, s2: coal collision geometries
        T1, T2: Transform3s for each shape

    Returns:
        d_geom: >0 if separated, <0 if overlapping (penetration depth as negative gap)
    """
    res = coal.DistanceResult()
    coal.distance(s1, T1, s2, T2, coal.DistanceRequest(), res)
    d = res.min_distance
    return float(0.0 if d is None else d)


def hull_hull(pts_a, pts_b, r_a=0.0, r_b=0.0, margin=0.0):
    """Distance between two convex hulls (e.g. Hermite/Bézier control CH).

    Hulls bound centerlines only; tube/obstacle radii go into d_req (§7.2.2).
    Clearance holds when d_geom >= d_req.

    Args:
        pts_a, pts_b: hull vertices, shape (n, 3) (typically 4 Bézier controls)
        r_a, r_b: radii associated with each hull (pipe–pipe or pipe–shell)
        margin: extra separation δ

    Returns:
        d_geom: signed hull–hull gap (negative if overlapping)
        d_req: r_a + r_b + margin
    """
    return _dist(_convex(pts_a), _ID, _convex(pts_b), _ID), float(r_a) + float(r_b) + float(margin)


def hull_aabb(pts, mn, mx, r_hull=0.0, r_obs=0.0, margin=0.0):
    """Distance from a convex hull to an AABB obstacle (as coal.Box).

    Args:
        pts: hull vertices, shape (n, 3)
        mn, mx: AABB corners, shape (3,)
        r_hull: pipe radius on the hull side
        r_obs: obstacle shell / padding on the box (0 if the AABB is already solid)
        margin: extra separation δ

    Returns:
        d_geom: signed hull–box gap (negative if overlapping)
        d_req: r_hull + r_obs + margin
    """
    box, Tb = _aabb(mn, mx)
    return _dist(_convex(pts), _ID, box, Tb), float(r_hull) + float(r_obs) + float(margin)


def aabb_aabb(mn_a, mx_a, mn_b, mx_b, r_a=0.0, r_b=0.0, margin=0.0):
    """Distance between two AABBs (same coal path as hull queries).

    Typical use: subdivided arc AABBs vs obstacles or other pipes (§7.2.2).

    Args:
        mn_a, mx_a: first AABB corners, shape (3,)
        mn_b, mx_b: second AABB corners, shape (3,)
        r_a, r_b: radii / shell padding for each box
        margin: extra separation δ

    Returns:
        d_geom: signed box–box gap (negative if overlapping)
        d_req: r_a + r_b + margin
    """
    ba, Ta = _aabb(mn_a, mx_a)
    bb, Tb = _aabb(mn_b, mx_b)
    return _dist(ba, Ta, bb, Tb), float(r_a) + float(r_b) + float(margin)
