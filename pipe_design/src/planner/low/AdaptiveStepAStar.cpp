#include "planner/low/AdaptiveStepAStar.h"

#include "geom/Heading.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pipe_design::planner {

int AdaptiveStepAStar::step_size_for(float clearance, float pipe_radius) const {
    if (shell_ == nullptr || grid_ == nullptr) {
        return 1;
    }
    const float res = grid_->static_grid().resolution();
    const float slack = clearance - pipe_radius;
    const int s = static_cast<int>(std::floor(static_cast<double>(slack) / static_cast<double>(res)));
    return std::clamp(s, 1, shell_->max_step());
}

void AdaptiveStepAStar::reset(
    const grid::CompositeGrid& grid,
    grid::CellCoord start,
    grid::CellCoord goal,
    float pipe_radius,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    motion::AdaptiveShell& shell) {
    reset(
        grid,
        std::vector<grid::CellCoord>{start},
        std::vector<grid::CellCoord>{goal},
        pipe_radius,
        heuristic,
        open,
        shell);
}

void AdaptiveStepAStar::reset(
    const grid::CompositeGrid& grid,
    const std::vector<grid::CellCoord>& starts,
    const std::vector<grid::CellCoord>& goals,
    float pipe_radius,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    motion::AdaptiveShell& shell) {
    shell_ = &shell;
    covered_.clear();
    AStar<SearchNode>::reset(grid, starts, goals, pipe_radius, heuristic, open, shell);
}

void AdaptiveStepAStar::expand(SearchNode* current) {
    const int s = step_size_for(grid_->clearance(current->coord), pipe_radius_);
    shell_->set_step_size(s);

    successor_buf_.clear();
    shell_->successors(current->coord, geom::Heading{}, successor_buf_);
    for (const grid::CellCoord& g : goals_) {
        shell_->append_interior_goal(current->coord, g, geom::Heading{}, successor_buf_);
    }

    for (const motion::Successor& succ : successor_buf_) {
        if (!grid_->in_bounds(succ.cell) || !grid_->is_free(succ.cell, pipe_radius_)) {
            continue;
        }
        if (covered_.find(succ.cell) != covered_.end()) {
            continue;
        }

        SearchNode* neighbor = ensure_node(SearchNode::make_key(succ.cell));
        if (neighbor->status == NodeStatus::Closed) {
            continue;
        }

        const double tentative_g = current->g + succ.cost;
        const bool discovered = neighbor->status == NodeStatus::New;
        if (!discovered && tentative_g >= neighbor->g) {
            continue;
        }

        neighbor->parent = current;
        set_node_costs(neighbor, tentative_g, heuristic_at(succ.cell));
        neighbor->status = NodeStatus::Open;
        if (discovered) {
            open_->push(neighbor);
        } else if (!open_->decrease_key(neighbor)) {
            open_->push(neighbor);
        }
    }

    // Cover Chebyshev ball interior so later shells cannot jump back in.
    const std::vector<motion::AdaptiveShell::Offset>& interior = shell_->interior_offsets_for(s);
    covered_.reserve(covered_.size() + interior.size());
    for (const motion::AdaptiveShell::Offset& o : interior) {
        covered_.insert(grid::CellCoord{
            current->coord.x + o.dx,
            current->coord.y + o.dy,
            current->coord.z + o.dz,
        });
    }
}

}  // namespace pipe_design::planner
