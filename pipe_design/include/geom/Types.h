#pragma once

#include <cmath>
#include <cstdint>
#include <ostream>

namespace pipe_design::geom {

/// Continuous world-space point / vector.
struct Vec3 {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }

    float norm_sq() const { return x * x + y * y + z * z; }
    float norm() const { return std::sqrt(norm_sq()); }

    friend std::ostream& operator<<(std::ostream& os, const Vec3& v) {
        return os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
    }
};

/// Discriminator for solid obstacle types (BVH refs, loaders, …).
enum class ObstacleKind : std::uint8_t { Sphere = 0, Capsule = 1, Aabb = 2 };

/// Spherical obstacle: any shape can be approximated by a set of these.
struct SphereObstacle {
    Vec3 center;
    float radius = 0.f;

    SphereObstacle() = default;
    SphereObstacle(Vec3 center_, float radius_) : center(center_), radius(radius_) {}
};

/// Capsule obstacle: sphere swept along a segment (pipe segment / elongated body).
struct CapsuleObstacle {
    Vec3 a;
    Vec3 b;
    float radius = 0.f;

    CapsuleObstacle() = default;
    CapsuleObstacle(Vec3 a_, Vec3 b_, float radius_) : a(a_), b(b_), radius(radius_) {}
};

/// Axis-aligned box obstacle: closed world-space slab \([mn, mx]\).
/// Distinct from BVH query `Aabb` in `Bvh.h` (bounding volume, not a solid).
struct AabbObstacle {
    Vec3 mn;
    Vec3 mx;

    AabbObstacle() = default;
    AabbObstacle(Vec3 mn_, Vec3 mx_) : mn(mn_), mx(mx_) {}
};

}  // namespace pipe_design::geom
