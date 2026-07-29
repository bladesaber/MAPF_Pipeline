#include "heuristic/Dijkstra.h"

#include <queue>
#include <utility>

namespace pipe_design::heuristic {

Dijkstra::Dijkstra(
    const grid::StaticClearanceGrid& grid,
    double step_cost,
    float bake_radius)
    : grid_(&grid),
      step_cost_(step_cost > 0.0 ? step_cost : static_cast<double>(grid.resolution())),
      bake_radius_(bake_radius) {}

void Dijkstra::clear() {
    prepared_ = false;
    goals_.clear();
    dist_.clear();
}

void Dijkstra::prepare(grid::CellCoord goal) {
    prepare(std::vector<grid::CellCoord>{goal});
}

void Dijkstra::prepare(std::initializer_list<grid::CellCoord> goals) {
    prepare(std::vector<grid::CellCoord>(goals));
}

void Dijkstra::prepare(const std::vector<grid::CellCoord>& goals) {
    clear();
    goals_ = goals;
    run_dijkstra();
}

bool Dijkstra::traversable(const grid::CellCoord& cell) const {
    if (!grid_->in_bounds(cell)) {
        return false;
    }
    if (bake_radius_ > 0.f) {
        return grid_->is_free(cell, bake_radius_);
    }
    // Relaxed: not tied to query pipe radius; exclude non-positive clearance
    // (stamped solid / surface after clamp to 0).
    return grid_->clearance(cell) > 0.f;
}

void Dijkstra::run_dijkstra() {
    dist_.assign(grid_->cell_count(), kUnreachable);

    using Node = std::pair<double, grid::CellCoord>;  // (g, cell)
    auto cmp = [](const Node& a, const Node& b) { return a.first > b.first; };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> open(cmp);

    for (const grid::CellCoord& gcell : goals_) {
        if (!traversable(gcell)) {
            continue;
        }
        const std::size_t idx = grid_->flat_index(gcell);
        if (dist_[idx] > 0.0) {
            dist_[idx] = 0.0;
            open.push({0.0, gcell});
        }
    }

    if (open.empty()) {
        prepared_ = false;
        return;
    }

    while (!open.empty()) {
        const auto [g, cell] = open.top();
        open.pop();

        if (g > dist_[grid_->flat_index(cell)]) {
            continue;  // stale
        }

        for (int i = 0; i < kNumNeighbors; ++i) {
            grid::CellCoord nb{
                cell.x + kOffsets[i][0],
                cell.y + kOffsets[i][1],
                cell.z + kOffsets[i][2],
            };
            if (!traversable(nb)) {
                continue;
            }
            const double ng = g + step_cost_;
            const std::size_t idx = grid_->flat_index(nb);
            if (ng < dist_[idx]) {
                dist_[idx] = ng;
                open.push({ng, nb});
            }
        }
    }

    prepared_ = true;
}

double Dijkstra::estimate(const grid::CellCoord& from, const grid::CellCoord& to) const {
    (void)to;  // multi-source: distance-to-set; `to` unused when prepared
    if (!prepared_) {
        return 0.0;
    }
    return distance(from);
}

double Dijkstra::distance(const grid::CellCoord& cell) const {
    if (!prepared_ || !grid_->in_bounds(cell)) {
        return kUnreachable;
    }
    return dist_[grid_->flat_index(cell)];
}

}  // namespace pipe_design::heuristic
