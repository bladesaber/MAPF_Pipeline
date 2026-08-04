#include "planner/middle/BidirectionalAStar.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace pipe_design::planner {

void BidirectionalAStar::reset(
    const grid::CompositeGrid& grid,
    grid::CellCoord start,
    grid::CellCoord goal,
    float pipe_radius,
    const heuristic::Heuristic& heuristic,
    OpenSet& open_forward,
    OpenSet& open_backward,
    const motion::ActionModel& actions) {
    reset(
        grid,
        std::vector<grid::CellCoord>{start},
        std::vector<grid::CellCoord>{goal},
        pipe_radius,
        heuristic,
        open_forward,
        open_backward,
        actions);
}

void BidirectionalAStar::reset(
    const grid::CompositeGrid& grid,
    const std::vector<grid::CellCoord>& starts,
    const std::vector<grid::CellCoord>& goals,
    float pipe_radius,
    const heuristic::Heuristic& heuristic,
    OpenSet& open_forward,
    OpenSet& open_backward,
    const motion::ActionModel& actions) {
    expansions_ = 0;
    prefer_forward_ = true;
    path_mode_ = PathMode::None;
    meet_fwd_ = nullptr;
    meet_bwd_ = nullptr;
    meet_cost_ = 0.0;

    fwd_.reset(grid, starts, goals, pipe_radius, heuristic, open_forward, actions);
    bwd_.reset(grid, goals, starts, pipe_radius, heuristic, open_backward, actions);

    if (fwd_.status() == SearchStatus::Success) {
        path_mode_ = PathMode::ForwardOnly;
        status_ = SearchStatus::Success;
        return;
    }
    if (bwd_.status() == SearchStatus::Success) {
        path_mode_ = PathMode::BackwardOnly;
        status_ = SearchStatus::Success;
        return;
    }
    if (fwd_.status() != SearchStatus::Running ||
        bwd_.status() != SearchStatus::Running) {
        status_ = SearchStatus::Failure;
        return;
    }
    status_ = SearchStatus::Running;
}

void BidirectionalAStar::set_heuristic_weight(double weight) {
    fwd_.set_heuristic_weight(weight);
    bwd_.set_heuristic_weight(weight);
}

SearchStatus BidirectionalAStar::step() {
    if (status_ != SearchStatus::Running) {
        return status_;
    }

    const bool fwd_ok = fwd_.status() == SearchStatus::Running;
    const bool bwd_ok = bwd_.status() == SearchStatus::Running;
    if (!fwd_ok && !bwd_ok) {
        status_ = SearchStatus::Failure;
        return status_;
    }

    AStar<SearchNode>* side = nullptr;
    if (fwd_ok && bwd_ok) {
        side = prefer_forward_ ? &fwd_ : &bwd_;
        prefer_forward_ = !prefer_forward_;
    } else {
        side = fwd_ok ? &fwd_ : &bwd_;
    }

    const bool from_fwd = side == &fwd_;
    side->step();
    expansions_ = fwd_.expansions() + bwd_.expansions();

    if (side->status() == SearchStatus::Success) {
        path_mode_ = from_fwd ? PathMode::ForwardOnly : PathMode::BackwardOnly;
        status_ = SearchStatus::Success;
        return status_;
    }

    const SearchNode* cur = side->last_expanded();
    if (cur != nullptr) {
        const AStar<SearchNode>& other = from_fwd ? bwd_ : fwd_;
        const SearchNode* o = other.find_node(SearchNode::make_key(cur->coord));
        if (o != nullptr && o->status == NodeStatus::Closed) {
            meet_fwd_ = from_fwd ? cur : o;
            meet_bwd_ = from_fwd ? o : cur;
            meet_cost_ = meet_fwd_->g + meet_bwd_->g;
            path_mode_ = PathMode::Meet;
            status_ = SearchStatus::Success;
            return status_;
        }
    }

    if (fwd_.status() != SearchStatus::Running &&
        bwd_.status() != SearchStatus::Running) {
        status_ = SearchStatus::Failure;
    }
    return status_;
}

Path BidirectionalAStar::path() const {
    if (status_ != SearchStatus::Success) {
        return {};
    }

    if (path_mode_ == PathMode::ForwardOnly) {
        return fwd_.path();
    }

    if (path_mode_ == PathMode::BackwardOnly) {
        Path p = bwd_.path();
        std::reverse(p.cells.begin(), p.cells.end());
        return p;
    }

    if (path_mode_ != PathMode::Meet || meet_fwd_ == nullptr || meet_bwd_ == nullptr) {
        return {};
    }

    Path out;
    for (const SearchNode* n = meet_fwd_; n != nullptr; n = n->parent) {
        out.cells.push_back(n->coord);
    }
    std::reverse(out.cells.begin(), out.cells.end());

    for (const SearchNode* n = meet_bwd_->parent; n != nullptr; n = n->parent) {
        out.cells.push_back(n->coord);
    }
    out.cost = meet_cost_;
    return out;
}

}  // namespace pipe_design::planner
