#include "grid/CompositeGrid.h"

#include <algorithm>
#include <cmath>

namespace pipe_design::grid {

CompositeGrid::CompositeGrid(
    const StaticClearanceGrid& static_grid,
    std::shared_ptr<const ObstacleChain> chain)
    : static_grid_(&static_grid), chain_(std::move(chain)) {}

float CompositeGrid::clearance(const CellCoord& cell) const {
    const float static_clr = static_grid_->clearance(cell);
    if (!chain_ || chain_->total_count() == 0) {
        return static_clr;
    }
    const float dynamic_clr = chain_->clearance(static_grid_->cell_center(cell));
    const float dynamic_stored = std::max(0.f, dynamic_clr);
    return std::min(static_clr, dynamic_stored);
}

float CompositeGrid::clearance(const geom::Vec3& point) const {
    const float inv = 1.f / static_grid_->resolution();
    const CellCoord cell{
        static_cast<int>(std::floor(point.x * inv)),
        static_cast<int>(std::floor(point.y * inv)),
        static_cast<int>(std::floor(point.z * inv)),
    };
    const float static_clr = static_grid_->clearance(cell);

    if (!chain_ || chain_->total_count() == 0) {
        return static_clr;
    }
    const float dynamic_clr = std::max(0.f, chain_->clearance(point));
    return std::min(static_clr, dynamic_clr);
}

bool CompositeGrid::is_free(const CellCoord& cell, float pipe_radius) const {
    if (static_grid_->clearance(cell) < pipe_radius) {
        return false;
    }
    if (chain_ && chain_->collides(static_grid_->cell_center(cell), pipe_radius)) {
        return false;
    }
    return true;
}

bool CompositeGrid::is_free(const geom::Vec3& point, float pipe_radius) const {
    const float inv = 1.f / static_grid_->resolution();
    const CellCoord cell{
        static_cast<int>(std::floor(point.x * inv)),
        static_cast<int>(std::floor(point.y * inv)),
        static_cast<int>(std::floor(point.z * inv)),
    };
    if (static_grid_->clearance(cell) < pipe_radius) {
        return false;
    }
    if (chain_ && chain_->collides(point, pipe_radius)) {
        return false;
    }
    return true;
}

}  // namespace pipe_design::grid
