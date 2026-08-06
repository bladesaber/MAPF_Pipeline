#pragma once

#include "geom/Types.h"
#include "grid/Types.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace pipe_design::grid {

/// Dense truncated clearance field for static obstacles.
///
/// Each cell stores clearance to the nearest static obstacle surface,
/// clamped to [0, clearance_cap]. Cells far from all obstacles keep
/// clearance_cap, so bake only needs to touch a local neighborhood of
/// each obstacle (radius ≈ obstacle_r + clearance_cap).
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

    bool in_bounds(const CellCoord& cell) const;

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

    /// O(1) lookup. Out-of-bounds returns 0 (treated as blocked).
    float clearance(const CellCoord& cell) const;

    bool is_free(const CellCoord& cell, float pipe_radius) const {
        return clearance(cell) >= pipe_radius;
    }

    /// Row-major index: x + y*size_x + z*size_x*size_y. Cell must be in_bounds.
    std::size_t flat_index(const CellCoord& cell) const;

    /// Total cell count (size_x * size_y * size_z).
    std::size_t cell_count() const {
        return field_.size();
    }

    /// Rebuild the whole field from static geometry.
    /// clearance_cap should be >= max_obstacle_radius + max_pipe_radius.
    void bake(
        const std::vector<geom::SphereObstacle>& spheres,
        const std::vector<geom::CapsuleObstacle>& capsules = {});

    /// Reset every cell to clearance_cap (empty domain).
    void clear();

private:
    void stamp_sphere(const geom::SphereObstacle& sphere);
    void stamp_capsule(const geom::CapsuleObstacle& capsule);
    void lower_clearance(const CellCoord& cell, float value);

    int size_x_ = 0;
    int size_y_ = 0;
    int size_z_ = 0;
    float resolution_ = 1.f;
    float clearance_cap_ = 1.f;
    int slice_size_ = 0;
    std::vector<float> field_;
};

}  // namespace pipe_design::grid
