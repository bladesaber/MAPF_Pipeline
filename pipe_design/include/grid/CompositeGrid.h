#pragma once

#include "geom/Types.h"
#include "grid/ObstacleChain.h"
#include "grid/StaticClearanceGrid.h"

#include <memory>

namespace pipe::grid {

/// Occupancy query that combines:
///   - StaticClearanceGrid  (shared dense field)
///   - ObstacleChain        (CBS-node dynamic / constraint obstacles)
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

    geom::Vec3 cell_center(const CellCoord& cell) const {
        return static_grid_->cell_center(cell);
    }

    /// Combined clearance at a grid cell.
    float clearance(const CellCoord& cell) const;

    /// Combined clearance at an arbitrary world point
    /// (static field sampled at nearest cell + chain analytic query).
    float clearance(const geom::Vec3& point) const;

    /// True if a pipe of `pipe_radius` fits at the cell / point.
    /// Short-circuits: static check first, then chain.collides (early exit).
    bool is_free(const CellCoord& cell, float pipe_radius) const;
    bool is_free(const geom::Vec3& point, float pipe_radius) const;

private:
    const StaticClearanceGrid* static_grid_ = nullptr;
    std::shared_ptr<const ObstacleChain> chain_;
};

}  // namespace pipe::grid
