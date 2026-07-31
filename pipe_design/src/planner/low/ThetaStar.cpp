#include "planner/low/ThetaStar.h"

#include "geom/Heading.h"
#include "geom/Types.h"

#include <algorithm>
#include <cmath>

namespace pipe_design::planner {

void ThetaStar::set_step_cost(double step_cost) {
    step_cost_ = step_cost > 0.0 ? step_cost : 1.0;
}

bool ThetaStar::line_of_sight(
    const grid::CompositeGrid& grid,
    grid::CellCoord a,
    grid::CellCoord b,
    float pipe_radius) {
    if (!grid.in_bounds(a) || !grid.is_free(a, pipe_radius)) {
        return false;
    }
    if (a == b) {
        return true;
    }
    if (!grid.in_bounds(b) || !grid.is_free(b, pipe_radius)) {
        return false;
    }

    const geom::Vec3 pa = grid.cell_center(a);
    const geom::Vec3 pb = grid.cell_center(b);
    const geom::Vec3 delta = pb - pa;
    const float len = delta.norm();
    if (len < 1e-9f) {
        return true;
    }

    const float res = grid.static_grid().resolution();
    const float step = std::max(1e-6f, 0.25f * res);
    const int n = std::max(1, static_cast<int>(std::ceil(static_cast<double>(len / step))));
    for (int i = 1; i < n; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(n);
        if (!grid.is_free(pa + delta * t, pipe_radius)) {
            return false;
        }
    }
    return true;
}

double ThetaStar::euclidean_cell_cost(
    const grid::CellCoord& a,
    const grid::CellCoord& b,
    double step_cost) {
    const double dx = static_cast<double>(a.x - b.x);
    const double dy = static_cast<double>(a.y - b.y);
    const double dz = static_cast<double>(a.z - b.z);
    return step_cost * std::sqrt(dx * dx + dy * dy + dz * dz);
}

void ThetaStar::update_vertex(
    SearchNode* current,
    SearchNode* neighbor,
    double grid_edge_cost) {
    auto* parent = static_cast<SearchNode*>(current->parent);
    double tentative_g = current->g + grid_edge_cost;
    SearchNode* tentative_parent = current;

    if (parent != nullptr &&
        line_of_sight(*grid_, parent->coord, neighbor->coord, pipe_radius_)) {
        const double via_parent =
            parent->g + euclidean_cell_cost(parent->coord, neighbor->coord, step_cost_);
        if (via_parent < tentative_g) {
            tentative_g = via_parent;
            tentative_parent = parent;
        }
    }

    const bool discovered = neighbor->status == NodeStatus::New;
    if (!discovered && tentative_g >= neighbor->g) {
        return;
    }

    neighbor->parent = tentative_parent;
    set_node_costs(neighbor, tentative_g, heuristic_at(neighbor->coord));
    neighbor->status = NodeStatus::Open;
    if (discovered) {
        open_->push(neighbor);
    } else if (!open_->decrease_key(neighbor)) {
        open_->push(neighbor);
    }
}

void ThetaStar::expand(SearchNode* current) {
    successor_buf_.clear();
    actions_->successors(current->coord, geom::Heading{}, successor_buf_);

    for (const motion::Successor& succ : successor_buf_) {
        if (!grid_->in_bounds(succ.cell) || !grid_->is_free(succ.cell, pipe_radius_)) {
            continue;
        }

        SearchNode* neighbor = ensure_node(SearchNode::make_key(succ.cell));
        if (neighbor->status == NodeStatus::Closed) {
            continue;
        }

        update_vertex(current, neighbor, succ.cost);
    }
}

}  // namespace pipe_design::planner
