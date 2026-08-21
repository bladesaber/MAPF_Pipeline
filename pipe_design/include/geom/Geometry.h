#pragma once

#include "geom/Types.h"

#include <algorithm>
#include <cmath>

namespace pipe_design::geom {

inline float distance(const Vec3& a, const Vec3& b) {
    return (a - b).norm();
}

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// Signed clearance from a point to the obstacle surface.
/// Positive = outside, zero = on surface, negative = penetrating.
inline float signed_clearance(const Vec3& point, const SphereObstacle& obs) {
    return distance(point, obs.center) - obs.radius;
}

inline float point_to_segment_distance(const Vec3& point, const Vec3& a, const Vec3& b) {
    const Vec3 ab = b - a;
    const float ab_len_sq = ab.norm_sq();
    if (ab_len_sq < 1e-12f) {
        return distance(point, a);
    }
    float t = dot(point - a, ab) / ab_len_sq;
    if (t < 0.f) {
        t = 0.f;
    }
    if (t > 1.f) {
        t = 1.f;
    }
    const Vec3 closest{a.x + ab.x * t, a.y + ab.y * t, a.z + ab.z * t};
    return distance(point, closest);
}

inline float signed_clearance(const Vec3& point, const CapsuleObstacle& obs) {
    return point_to_segment_distance(point, obs.a, obs.b) - obs.radius;
}

/// Signed clearance from a point to an AABB surface.
inline float signed_clearance(const Vec3& point, const AabbObstacle& obs) {
    const float cx = std::clamp(point.x, obs.mn.x, obs.mx.x);
    const float cy = std::clamp(point.y, obs.mn.y, obs.mx.y);
    const float cz = std::clamp(point.z, obs.mn.z, obs.mx.z);
    const float ox = point.x - cx;
    const float oy = point.y - cy;
    const float oz = point.z - cz;
    const float d2 = ox * ox + oy * oy + oz * oz;
    if (d2 > 0.f) {
        return std::sqrt(d2);
    }
    const float ix = std::min(point.x - obs.mn.x, obs.mx.x - point.x);
    const float iy = std::min(point.y - obs.mn.y, obs.mx.y - point.y);
    const float iz = std::min(point.z - obs.mn.z, obs.mx.z - point.z);
    return -std::min(ix, std::min(iy, iz));
}

/// Closest distance between segments p0→p1 and q0→q1 (3D).
/// Degenerate segments (near-zero length) fall back to point–segment / point–point.
inline float segment_to_segment_distance(
    const Vec3& p0,
    const Vec3& p1,
    const Vec3& q0,
    const Vec3& q1) {
    constexpr float kEps = 1e-12f;

    const Vec3 u = p1 - p0;
    const Vec3 v = q1 - q0;
    const float a = u.norm_sq();
    const float c = v.norm_sq();

    if (a < kEps && c < kEps) {
        return distance(p0, q0);
    }
    if (a < kEps) {
        return point_to_segment_distance(p0, q0, q1);
    }
    if (c < kEps) {
        return point_to_segment_distance(q0, p0, p1);
    }

    const Vec3 w = p0 - q0;
    const float b = dot(u, v);
    const float d = dot(u, w);
    const float e = dot(v, w);
    const float D = a * c - b * b;

    float sN = 0.f;
    float sD = D;
    float tN = 0.f;
    float tD = D;

    if (D < kEps) {
        // Nearly parallel: pick s = 0 and project onto q.
        sN = 0.f;
        sD = 1.f;
        tN = e;
        tD = c;
    } else {
        sN = b * e - c * d;
        tN = a * e - b * d;
        if (sN < 0.f) {
            sN = 0.f;
            tN = e;
            tD = c;
        } else if (sN > sD) {
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }

    if (tN < 0.f) {
        tN = 0.f;
        if (-d < 0.f) {
            sN = 0.f;
        } else if (-d > a) {
            sN = sD;
        } else {
            sN = -d;
            sD = a;
        }
    } else if (tN > tD) {
        tN = tD;
        if ((-d + b) < 0.f) {
            sN = 0.f;
        } else if ((-d + b) > a) {
            sN = sD;
        } else {
            sN = -d + b;
            sD = a;
        }
    }

    const float sc = (std::fabs(sN) < kEps) ? 0.f : (sN / sD);
    const float tc = (std::fabs(tN) < kEps) ? 0.f : (tN / tD);
    const Vec3 diff = w + u * sc - v * tc;
    return diff.norm();
}

/// Signed clearance from segment `from`→`to` to a sphere surface.
inline float signed_clearance(
    const Vec3& from,
    const Vec3& to,
    const SphereObstacle& obs) {
    return point_to_segment_distance(obs.center, from, to) - obs.radius;
}

/// Signed clearance from segment `from`→`to` to a capsule surface.
/// Equivalent to segment–segment distance minus capsule radius.
inline float signed_clearance(
    const Vec3& from,
    const Vec3& to,
    const CapsuleObstacle& obs) {
    return segment_to_segment_distance(from, to, obs.a, obs.b) - obs.radius;
}

/// Signed clearance from segment `from`→`to` to an AABB.
/// Minimizes point–AABB clearance along the segment (convex → ternary search).
inline float signed_clearance(
    const Vec3& from,
    const Vec3& to,
    const AabbObstacle& obs) {
    auto at = [&](float t) {
        return signed_clearance(
            Vec3{
                from.x + (to.x - from.x) * t,
                from.y + (to.y - from.y) * t,
                from.z + (to.z - from.z) * t},
            obs);
    };
    float lo = 0.f;
    float hi = 1.f;
    constexpr float kTEps = 1e-5f;
    while (true) {
        const float m1 = lo + (hi - lo) * (1.f / 3.f);
        const float m2 = lo + (hi - lo) * (2.f / 3.f);
        if (std::fabs(m1 - m2) < kTEps) {
            break;
        }
        if (at(m1) > at(m2)) {
            lo = m1;
        } else {
            hi = m2;
        }
    }
    return at(0.5f * (lo + hi));
}

}  // namespace pipe_design::geom
