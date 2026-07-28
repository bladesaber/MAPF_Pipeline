#pragma once

#include <cmath>
#include <ostream>

namespace pipe::geom {

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

}  // namespace pipe::geom
