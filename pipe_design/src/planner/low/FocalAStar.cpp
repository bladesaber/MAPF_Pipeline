#include "planner/low/FocalAStar.h"

#include "geom/Heading.h"

#include <vector>

namespace pipe_design::planner {

void FocalAStar::set_focal_weight(double weight) {
    focal_weight_ = weight < 1.0 ? 1.0 : weight;
}

double FocalAStar::combine_key(double g, double h) const {
    return g + h;
}

void FocalAStar::reset(
    const grid::CompositeGrid& grid,
    grid::CellCoord start,
    grid::CellCoord goal,
    float pipe_radius,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions) {
    reset(
        grid,
        std::vector<grid::CellCoord>{start},
        std::vector<grid::CellCoord>{goal},
        pipe_radius,
        heuristic,
        open,
        actions);
}

void FocalAStar::reset(
    const grid::CompositeGrid& grid,
    const std::vector<grid::CellCoord>& starts,
    const std::vector<grid::CellCoord>& goals,
    float pipe_radius,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions) {
    focal_.clear();
    open_nodes_.clear();
    focal_bound_ = 0.0;

    AStar<SearchNode>::reset(grid, starts, goals, pipe_radius, heuristic, open, actions);

    if (status_ != SearchStatus::Running || focal_weight_ <= 1.0) {
        return;
    }
    // Seed open_nodes_ from starts (only Open nodes exist right after reset).
    for (const auto& kv : nodes_) {
        SearchNode* n = kv.second.get();
        if (n->status == NodeStatus::Open) {
            open_nodes_.insert(n);
        }
    }
    focal_bound_ = focal_weight_ * open_->top()->f;
    for (SearchNode* n : open_nodes_) {
        if (n->f <= focal_bound_) {
            focal_.push(n);
        }
    }
}

void FocalAStar::sync_focal_bound() {
    const double bound = focal_weight_ * open_->top()->f;
    if (bound <= focal_bound_) {  // in most situations, the bound will not change, so we can return early
        focal_bound_ = bound;
        return;
    }
    for (SearchNode* n : open_nodes_) {
        if (n->f <= bound && n->f > focal_bound_) {
            push_focal(n);
        }
    }
    focal_bound_ = bound;
}

void FocalAStar::push_focal(SearchNode* node) {
    if (node->f > focal_bound_) {
        return;
    }
    if (focal_.contains(node)) {
        focal_.decrease_key(node);
    } else {
        focal_.push(node);
    }
}

SearchStatus FocalAStar::step() {
    if (focal_weight_ <= 1.0) {
        return AStar<SearchNode>::step();
    }
    if (status_ != SearchStatus::Running) {
        return status_;
    }
    if (open_ == nullptr || open_->empty()) {
        status_ = SearchStatus::Failure;
        return status_;
    }

    sync_focal_bound();

    while (!focal_.empty() &&
           (focal_.top()->status != NodeStatus::Open || focal_.top()->f > focal_bound_)) {
        focal_.pop();
    }
    if (focal_.empty()) {
        status_ = SearchStatus::Failure;
        return status_;
    }

    SearchNode* best = focal_.pop();
    open_->remove(best);
    open_nodes_.erase(best);
    best->status = NodeStatus::Closed;
    last_expanded_ = best;
    ++expansions_;

    before_expand(best);

    if (is_goal(best->coord)) {
        goal_node_ = best;
        status_ = SearchStatus::Success;
        return status_;
    }

    expand(best);

    if (open_->empty()) {
        status_ = SearchStatus::Failure;
    }
    return status_;
}

void FocalAStar::expand(SearchNode* current) {
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
        open_nodes_.insert(neighbor);
        push_focal(neighbor);
    }
}

}  // namespace pipe_design::planner
