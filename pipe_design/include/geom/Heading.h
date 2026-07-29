#pragma once

#include "geom/Types.h"

#include <algorithm>
#include <cmath>

namespace pipe_design::geom {

namespace heading_detail {
constexpr float kPi = 3.14159265358979323846f;
}  // namespace heading_detail

/// Unit direction in world space (source of truth for orientation).
///
/// Invariant: `forward()` is either the zero vector (none) or a unit vector.
/// Construction only via factories / default; there is no public `Heading(Vec3)`
/// that could skip normalization. Discrete bins are views — use bin()/from_bin().
///
/// (0,0,0) means "no heading" (e.g. search start before the first step).
class Heading {
public:
    /// None heading.
    Heading() = default;

    const Vec3& forward() const { return forward_; }

    bool is_none() const { return forward_.norm_sq() <= 0.f; }

    Heading negated() const {
        if (is_none()) {
            return Heading{};
        }
        return unit_unchecked(Vec3{-forward_.x, -forward_.y, -forward_.z});
    }

    /// Cosine of angle between unit headings in [-1, 1]. Returns 1 if either is none.
    float alignment(const Heading& other) const {
        if (is_none() || other.is_none()) {
            return 1.f;
        }
        const float d = forward_.x * other.forward_.x + forward_.y * other.forward_.y +
                        forward_.z * other.forward_.z;
        return std::clamp(d, -1.f, 1.f);
    }

    /// Angle in radians between headings. Returns 0 if either is none.
    float angle_to(const Heading& other) const {
        return std::acos(alignment(other));
    }

    bool nearly_equal(const Heading& other, float cos_eps = 1e-3f) const {
        if (is_none() && other.is_none()) {
            return true;
        }
        return alignment(other) >= 1.f - cos_eps;
    }

    bool is_opposite(const Heading& other, float cos_eps = 1e-3f) const {
        if (is_none() || other.is_none()) {
            return false;
        }
        return alignment(other) <= -1.f + cos_eps;
    }

    // --- factories (always normalize or produce an exact unit / none) ---

    /// Normalize `v`; zero / near-zero → none.
    static Heading from_vector(Vec3 v) {
        const float n2 = v.norm_sq();
        if (n2 <= 0.f) {
            return Heading{};
        }
        const float inv = 1.f / std::sqrt(n2);
        return unit_unchecked(Vec3{v.x * inv, v.y * inv, v.z * inv});
    }

    /// Yaw about +Z, pitch from XY-plane (radians). Already unit by construction.
    /// forward = (cos(pitch)·cos(yaw), cos(pitch)·sin(yaw), sin(pitch)).
    static Heading from_yaw_pitch(float yaw, float pitch) {
        const float cp = std::cos(pitch);
        return unit_unchecked(Vec3{cp * std::cos(yaw), cp * std::sin(yaw), std::sin(pitch)});
    }

    /// From a grid neighbor offset (need not be unit; e.g. (±1,±1,0)).
    static Heading from_grid_step(int dx, int dy, int dz) {
        return from_vector(
            Vec3{static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dz)});
    }

    // --- discrete views (for Hybrid-A* state hashing) ---

    /// Azimuth bin in [0, num_bins) from atan2(y, x). None → 0. Requires num_bins > 0.
    int yaw_bin(int num_bins) const {
        if (num_bins <= 0 || is_none()) {
            return 0;
        }
        float yaw = std::atan2(forward_.y, forward_.x);  // (-π, π]
        if (yaw < 0.f) {
            yaw += 2.f * heading_detail::kPi;
        }
        int b = static_cast<int>(
            yaw / (2.f * heading_detail::kPi) * static_cast<float>(num_bins));
        if (b >= num_bins) {
            b = num_bins - 1;
        }
        if (b < 0) {
            b = 0;
        }
        return b;
    }

    /// Elevation bin in [0, num_bins) from asin(z). None → mid bin. Requires num_bins > 0.
    int pitch_bin(int num_bins) const {
        if (num_bins <= 0) {
            return 0;
        }
        if (is_none()) {
            return num_bins / 2;
        }
        const float pitch = std::asin(std::clamp(forward_.z, -1.f, 1.f));  // [-π/2, π/2]
        const float u = (pitch + 0.5f * heading_detail::kPi) / heading_detail::kPi;  // [0, 1]
        int b = static_cast<int>(u * static_cast<float>(num_bins));
        if (b >= num_bins) {
            b = num_bins - 1;
        }
        if (b < 0) {
            b = 0;
        }
        return b;
    }

    /// Linear index: yaw + pitch * num_yaw_bins (default pitch bins = 1 → yaw-only).
    int bin(int num_yaw_bins, int num_pitch_bins = 1) const {
        return yaw_bin(num_yaw_bins) + pitch_bin(num_pitch_bins) * num_yaw_bins;
    }

    /// Inverse of bin(): heading at bin centers.
    static Heading from_bin(
        int yaw_b,
        int num_yaw_bins,
        int pitch_b = 0,
        int num_pitch_bins = 1) {
        if (num_yaw_bins <= 0) {
            return Heading{};
        }
        if (num_pitch_bins <= 0) {
            num_pitch_bins = 1;
        }
        yaw_b = ((yaw_b % num_yaw_bins) + num_yaw_bins) % num_yaw_bins;
        pitch_b = std::clamp(pitch_b, 0, num_pitch_bins - 1);

        const float yaw = (static_cast<float>(yaw_b) + 0.5f) *
                          (2.f * heading_detail::kPi / static_cast<float>(num_yaw_bins));
        const float pitch = -0.5f * heading_detail::kPi +
                            (static_cast<float>(pitch_b) + 0.5f) *
                                (heading_detail::kPi / static_cast<float>(num_pitch_bins));
        return from_yaw_pitch(yaw, pitch);
    }

    static Heading from_linear_bin(int linear, int num_yaw_bins, int num_pitch_bins = 1) {
        if (num_yaw_bins <= 0) {
            return Heading{};
        }
        if (num_pitch_bins <= 0) {
            num_pitch_bins = 1;
        }
        const int pitch_b = linear / num_yaw_bins;
        const int yaw_b = linear - pitch_b * num_yaw_bins;
        return from_bin(yaw_b, num_yaw_bins, pitch_b, num_pitch_bins);
    }

private:
    struct UnitTag {};

    /// Trusts that `u` is already unit or zero. Not for public use.
    explicit Heading(UnitTag, Vec3 u) : forward_(u) {}

    static Heading unit_unchecked(Vec3 u) { return Heading{UnitTag{}, u}; }

    Vec3 forward_{};
};

}  // namespace pipe_design::geom
