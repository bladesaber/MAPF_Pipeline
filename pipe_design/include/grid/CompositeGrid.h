#pragma once

#include "geom/Types.h"
#include "grid/ObstacleChain.h"
#include "grid/StaticClearanceGrid.h"

#include <memory>

namespace pipe_design::grid {

/// Occupancy query that combines:
///   - StaticClearanceGrid  (dense field + retained static geometry)
///   - ObstacleChain        (CBS-node dynamic / constraint obstacles)
///
/// Cell queries use the dense static field; world-point queries use analytic
/// static geometry (and analytic chain clearance).
///
/// clearance(cell) = min(static.clearance(cell), chain.clearance(cell_center))
class CompositeGrid {
public:
    CompositeGrid(
        const StaticClearanceGrid& static_grid,
        std::shared_ptr<const ObstacleChain> chain = nullptr);

    const StaticClearanceGrid& static_grid() const { return *static_grid_; }
    const std::shared_ptr<const ObstacleChain>& chain() const { return chain_; }

    void set_chain(std::shared_ptr<const ObstacleChain> chain) {
        chain_ = std::move(chain);
    }

    bool in_bounds(const CellCoord& cell) const {
        return static_grid_->in_bounds(cell);
    }

    bool in_bounds(const geom::Vec3& point) const {
        return static_grid_->in_bounds(point);
    }

    geom::Vec3 cell_center(const CellCoord& cell) const {
        return static_grid_->cell_center(cell);
    }

    CellCoord world_to_cell(const geom::Vec3& point) const {
        return static_grid_->world_to_cell(point);
    }

    /// Combined clearance at a grid cell.
    float clearance(const CellCoord& cell) const;

    /// Combined clearance at an arbitrary world point
    /// (analytic static geometry + analytic chain).
    float clearance(const geom::Vec3& point) const;

    /// True if a pipe of `pipe_radius` fits at the cell / point.
    /// Short-circuits: static check first, then chain.collides (early exit).
    bool is_free(const CellCoord& cell, float pipe_radius) const;
    bool is_free(const geom::Vec3& point, float pipe_radius) const;

    /// Edge freeness between cells:
    ///   Chebyshev == 1 → discrete AABB via `is_free` (dense field + chain)
    ///   Chebyshev  > 1 → analytic `is_move_free` between cell centers
    ///   Chebyshev == 0 → `is_free(from)`
    bool is_move_free(
        const CellCoord& from,
        const CellCoord& to,
        float pipe_radius) const;

    /// Continuous edge freeness: analytic static + chain segment checks.
    bool is_move_free(
        const geom::Vec3& from,
        const geom::Vec3& to,
        float pipe_radius) const;

private:
    const StaticClearanceGrid* static_grid_ = nullptr;
    std::shared_ptr<const ObstacleChain> chain_;
};

}  // namespace pipe_design::grid
