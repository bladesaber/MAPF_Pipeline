#pragma once

#include "geom/Bvh.h"
#include "geom/Types.h"
#include "grid/Types.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace pipe_design::grid {

/// Dense truncated clearance field for static obstacles, plus retained bake geometry.
///
/// Each cell stores clearance to the nearest static obstacle surface,
/// clamped to [0, clearance_cap]. Cells far from all obstacles keep
/// clearance_cap, so bake only needs to touch a local neighborhood of
/// each obstacle (radius ≈ obstacle_r + clearance_cap).
/// Set clearance_cap >= max pipe radius (usually equal to it).
///
/// `bake` also keeps the source spheres/capsules/aabbs for analytic world-point
/// queries (LOS / continuous samples) — cell lookups stay on the dense field.
///
/// Shared read-only across the entire CBS tree.
class StaticClearanceGrid {
public:
    StaticClearanceGrid(
        int size_x,
        int size_y,
        int size_z,
        float resolution,
        float clearance_cap);

    int size_x() const { return size_x_; }
    int size_y() const { return size_y_; }
    int size_z() const { return size_z_; }
    float resolution() const { return resolution_; }
    float clearance_cap() const { return clearance_cap_; }

    const std::vector<geom::SphereObstacle>& spheres() const { return spheres_; }
    const std::vector<geom::CapsuleObstacle>& capsules() const { return capsules_; }
    const std::vector<geom::AabbObstacle>& aabbs() const { return aabbs_; }

    bool in_bounds(const CellCoord& cell) const;

    /// True if `point` lies in the continuous domain
    /// `[0, size_x*res) × [0, size_y*res) × [0, size_z*res)`.
    bool in_bounds(const geom::Vec3& point) const;

    /// World-space center of a cell: `((i+0.5)*res, …)`.
    geom::Vec3 cell_center(const CellCoord& cell) const;

    /// Inverse of `cell_center` indexing: `floor(x / resolution)` (may be out of bounds).
    CellCoord world_to_cell(const geom::Vec3& point) const {
        const float inv = 1.f / resolution_;
        return {
            static_cast<int32_t>(std::floor(point.x * inv)),
            static_cast<int32_t>(std::floor(point.y * inv)),
            static_cast<int32_t>(std::floor(point.z * inv)),
        };
    }

    /// O(1) field lookup. Out-of-bounds returns 0 (treated as blocked).
    float clearance(const CellCoord& cell) const;

    /// Analytic clearance at a world point from retained bake geometry.
    /// Upper-bounded by clearance_cap; may be negative inside obstacles.
    /// Out-of-domain points return 0.
    float clearance(const geom::Vec3& point) const;

    bool is_free(const CellCoord& cell, float pipe_radius) const {
        return clearance(cell) >= pipe_radius;
    }

    /// Analytic freeness at a world point (retained bake geometry).
    bool is_free(const geom::Vec3& point, float pipe_radius) const;

    /// Edge freeness between cells:
    ///   Chebyshev == 1 → discrete AABB via `is_free`
    ///   Chebyshev  > 1 → analytic `is_move_free` between cell centers
    /// Throws if `from == to`.
    bool is_move_free(
        const CellCoord& from,
        const CellCoord& to,
        float pipe_radius) const;

    /// Continuous edge freeness via analytic segment–obstacle signed clearance.
    /// Does not check domain bounds — caller must ensure endpoints are valid.
    bool is_move_free(
        const geom::Vec3& from,
        const geom::Vec3& to,
        float pipe_radius) const;

    /// Row-major index: x + y*size_x + z*size_x*size_y. Cell must be in_bounds.
    std::size_t flat_index(const CellCoord& cell) const;

    /// Total cell count (size_x * size_y * size_z).
    std::size_t cell_count() const {
        return field_.size();
    }

    /// Rebuild the dense field, retain geometry, and rebuild the obstacle BVH.
    /// `clearance_cap` should be >= max pipe radius (see `clearance_cap_`).
    void bake(
        const std::vector<geom::SphereObstacle>& spheres,
        const std::vector<geom::CapsuleObstacle>& capsules = {},
        const std::vector<geom::AabbObstacle>& aabbs = {});

    /// Reset field to clearance_cap and drop retained geometry / BVH.
    void clear();

private:
    void stamp_sphere(const geom::SphereObstacle& sphere);
    void stamp_capsule(const geom::CapsuleObstacle& capsule);
    void stamp_aabb(const geom::AabbObstacle& aabb);
    void lower_clearance(const CellCoord& cell, float value);

    int size_x_ = 0;
    int size_y_ = 0;
    int size_z_ = 0;
    float resolution_ = 1.f;
    /// Truncation upper bound for the dense clearance field and analytic
    /// `clearance(point)`. Cells farther from obstacles than this keep this
    /// value; bake only stamps a neighborhood of radius ≈ obstacle_r + cap.
    ///
    /// NOTE: Must be >= the maximum pipe radius used with this grid
    /// (`is_free(..., pipe_radius)` / planning). Typical choice: set equal to
    /// that max pipe radius. It is NOT obstacle radius — field stores distance
    /// to obstacle surface, then compares against pipe_radius.
    float clearance_cap_ = 1.f;
    int slice_size_ = 0;
    std::vector<float> field_;
    std::vector<geom::SphereObstacle> spheres_;
    std::vector<geom::CapsuleObstacle> capsules_;
    std::vector<geom::AabbObstacle> aabbs_;
    geom::Bvh bvh_;
};

}  // namespace pipe_design::grid
