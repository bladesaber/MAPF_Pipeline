#include "grid/CompositeGrid.h"

#include <algorithm>

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
    const float static_clr = static_grid_->clearance(point);

    if (!chain_ || chain_->total_count() == 0) {
        return static_clr;
    }
    const float dynamic_clr = chain_->clearance(point);
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
    if (!static_grid_->is_free(point, pipe_radius)) {
        return false;
    }
    if (chain_ && chain_->collides(point, pipe_radius)) {
        return false;
    }
    return true;
}

bool CompositeGrid::is_move_free(
    const CellCoord& from,
    const CellCoord& to,
    float pipe_radius) const {
    const int32_t adx = std::abs(to.x - from.x);
    const int32_t ady = std::abs(to.y - from.y);
    const int32_t adz = std::abs(to.z - from.z);
    const int32_t cheb = std::max(adx, std::max(ady, adz));

    if (cheb == 0) {
        throw std::invalid_argument("CompositeGrid::is_move_free: from == to");
    }
    if (cheb > 1) {
        return is_move_free(cell_center(from), cell_center(to), pipe_radius);
    }

    // Unit neighbor (incl. diagonal): discrete corner-cut via composite is_free.
    const int32_t x0 = std::min(from.x, to.x);
    const int32_t x1 = std::max(from.x, to.x);
    const int32_t y0 = std::min(from.y, to.y);
    const int32_t y1 = std::max(from.y, to.y);
    const int32_t z0 = std::min(from.z, to.z);
    const int32_t z1 = std::max(from.z, to.z);

    for (int32_t z = z0; z <= z1; ++z) {
        for (int32_t y = y0; y <= y1; ++y) {
            for (int32_t x = x0; x <= x1; ++x) {
                if (x == from.x && y == from.y && z == from.z) {
                    continue;
                }
                if (!is_free(CellCoord{x, y, z}, pipe_radius)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool CompositeGrid::is_move_free(
    const geom::Vec3& from,
    const geom::Vec3& to,
    float pipe_radius) const {
    if (!static_grid_->is_move_free(from, to, pipe_radius)) {
        return false;
    }
    if (chain_ && chain_->collides(from, to, pipe_radius)) {
        return false;
    }
    return true;
}

}  // namespace pipe_design::grid
