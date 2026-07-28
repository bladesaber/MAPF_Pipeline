#pragma once

#include "geom/Types.h"

namespace pipe::geom {

inline float distance(const Vec3& a, const Vec3& b) {
    return (a - b).norm();
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
    float t = ((point.x - a.x) * ab.x + (point.y - a.y) * ab.y + (point.z - a.z) * ab.z) / ab_len_sq;
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

}  // namespace pipe::geom
