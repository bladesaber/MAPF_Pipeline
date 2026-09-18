"""§7.2.1 — polyline capsule narrow-phase kernels (numpy).

Capsule = line segment ⊕ ball(r). Returns (d_geom, d_req); hinge uses [d_req - d_geom]₊.
Detection / active-set stay outside AD (§7.2).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _seg_dist2(p, a, b):
    """Squared distance from point p to segment ab."""
    d = b - a
    denom = float(d @ d) + _EPS
    t = float(np.clip(((p - a) @ d) / denom, 0.0, 1.0))
    q = a + t * d
    e = p - q
    return float(e @ e)


def capsule_ball(a, b, r, c, r_obs=0.0, margin=0.0):
    """Distance from capsule (segment ab, radius r) to a ball (center c, radius r_obs).

    Projects c onto ab, then takes Euclidean distance from that point to c (centerline
    gap). Point obstacle: set r_obs=0. Clearance holds when d_geom >= d_req.

    Args:
        a, b: segment endpoints, shape (3,)
        r: tube / capsule radius
        c: obstacle center, shape (3,)
        r_obs: obstacle ball radius (0 for a point)
        margin: extra separation δ

    Returns:
        d_geom: centerline–center distance (>= 0)
        d_req: r + r_obs + margin
    """
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    d = b - a
    t = float(np.clip(((c - a) @ d) / (float(d @ d) + _EPS), 0.0, 1.0))
    d_geom = float(np.linalg.norm(a + t * d - c))
    return d_geom, float(r) + float(r_obs) + float(margin)


def capsule_capsule(a, b, r, a2, b2, r2, margin=0.0):
    """Distance between two capsules (segments ab / a2b2 with radii r / r2).

    Narrow-phase §7.2.1 Steps A–C: closest approach of the two centerline segments
    (infinite-line solve, then clamp to [0,1]; parallel / both-out → endpoint–
    segment fallback). d_geom is that segment–segment distance (>= 0). Clearance
    when d_geom >= d_req.

    Args:
        a, b: first segment endpoints, shape (3,)
        r: first capsule radius
        a2, b2: second segment endpoints, shape (3,)
        r2: second capsule radius
        margin: extra separation δ

    Returns:
        d_geom: centerline segment–segment distance
        d_req: r + r2 + margin
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a2 = np.asarray(a2, float)
    b2 = np.asarray(b2, float)
    d, d2, w = b - a, b2 - a2, a - a2
    A, B, C = float(d @ d), float(d @ d2), float(d2 @ d2)
    D, E = float(d @ w), float(d2 @ w)
    denom = A * C - B * B

    if denom <= _EPS:
        # parallel: min of four point–segment distances
        d2min = min(
            _seg_dist2(a, a2, b2),
            _seg_dist2(b, a2, b2),
            _seg_dist2(a2, a, b),
            _seg_dist2(b2, a, b),
        )
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
            d2min = min(
                _seg_dist2(a, a2, b2),
                _seg_dist2(b, a2, b2),
                _seg_dist2(a2, a, b),
                _seg_dist2(b2, a, b),
            )

    d_geom = float(np.sqrt(d2min + _EPS))
    return d_geom, float(r) + float(r2) + float(margin)


def capsule_aabb(a, b, r, mn, mx, margin=0.0, n_samp: int = 17):
    """Distance from capsule (segment ab, radius r) to a solid axis-aligned box [mn, mx].

    Samples n_samp points along ab and takes the minimum signed gap from each sample
    to the box: >0 outside, <0 if the centerline point is inside. Approximate
    min_α d□(p(α)) from §7.2.1; denser n_samp → tighter. Clearance when d_geom >= d_req.

    Args:
        a, b: segment endpoints, shape (3,)
        r: tube / capsule radius
        mn, mx: box corners, shape (3,) each (componentwise min / max)
        margin: extra separation δ
        n_samp: number of samples on ab (inclusive ends)

    Returns:
        d_geom: min signed centerline–box gap (may be negative if segment enters the box)
        d_req: r + margin
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    mn, mx = np.asarray(mn, float), np.asarray(mx, float)

    def signed_gap(q):
        q = np.asarray(q, float)
        closest = np.clip(q, mn, mx)
        if np.any(q < mn) or np.any(q > mx):
            return float(np.linalg.norm(q - closest))
        return -float(np.min(np.minimum(q - mn, mx - q)))

    ts = np.linspace(0.0, 1.0, max(int(n_samp), 2))
    d_geom = min(signed_gap(a + t * (b - a)) for t in ts)
    return d_geom, float(r) + float(margin)
