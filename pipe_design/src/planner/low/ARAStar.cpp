#include "planner/low/ARAStar.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pipe_design::planner {

void ARAStar::set_eps_schedule(double start, double target, double step) {
    eps_ = std::max(start, target);
    eps_target_ = target;
    eps_step_ = std::max(step, 1e-9);
}

double ARAStar::combine_key(double g, double h) const {
    return g + eps_ * h;
}

double ARAStar::suboptimality_bound() const {
    if (goal_node_ == nullptr) {
        return std::numeric_limits<double>::infinity();
    }
    double min_admissible = std::numeric_limits<double>::infinity();
    for (const auto& kv : nodes_) {
        const SearchNode* n = kv.second.get();
        if (n->status == NodeStatus::Open) {
            min_admissible = std::min(min_admissible, n->g + n->h);
        }
    }
    for (const SearchNode* n : incons_) {
        min_admissible = std::min(min_admissible, n->g + n->h);
    }
    if (!std::isfinite(min_admissible) || min_admissible <= 0.0) {
        return eps_;
    }
    return std::min(eps_, goal_node_->g / min_admissible);
}

Path ARAStar::path() const {
    if (goal_node_ == nullptr) {
        return {};
    }
    return reconstruct(goal_node_);
}

void ARAStar::expand(SearchNode* current) {
    successor_buf_.clear();
    actions_->successors(current->coord, geom::Heading{}, successor_buf_);

    for (const motion::Successor& succ : successor_buf_) {
        if (!grid_->in_bounds(succ.cell) || !grid_->is_free(succ.cell, pipe_radius_)) {
            continue;
        }

        SearchNode* neighbor = ensure_node(SearchNode::make_key(succ.cell));
        const double tentative_g = current->g + succ.cost;

        if (neighbor->status == NodeStatus::Closed) {
            if (tentative_g < neighbor->g) {
                neighbor->parent = current;
                set_node_costs(neighbor, tentative_g, heuristic_at(succ.cell));
                incons_.insert(neighbor);
                if (is_goal(succ.cell) &&
                    (goal_node_ == nullptr || tentative_g < goal_node_->g)) {
                    goal_node_ = neighbor;
                }
            }
            continue;
        }
        
        /// TODO: need to check whether this part is correct.
        // New + never costed (parent unset, f/h still 0) ⇒ unvisited.
        const bool unvisited = neighbor->status == NodeStatus::New &&
            neighbor->parent == nullptr && neighbor->h == 0.0 && neighbor->f == 0.0;
        if (!unvisited && tentative_g >= neighbor->g) {
            continue;
        }

        const bool in_open = neighbor->status == NodeStatus::Open;
        neighbor->parent = current;
        set_node_costs(neighbor, tentative_g, heuristic_at(succ.cell));
        neighbor->status = NodeStatus::Open;
        if (is_goal(succ.cell) &&
            (goal_node_ == nullptr || tentative_g < goal_node_->g)) {
            goal_node_ = neighbor;
        }
        if (!in_open) {
            open_->push(neighbor);
        } else if (!open_->decrease_key(neighbor)) {
            open_->push(neighbor);
        }
    }
}

SearchStatus ARAStar::step() {
    if (status_ != SearchStatus::Running) {
        return status_;
    }
    if (open_ == nullptr) {
        status_ = SearchStatus::Failure;
        return status_;
    }
    // reset() is non-virtual; clear stale INCONS pointers after a fresh reset.
    if (expansions_ == 0) {
        incons_.clear();
    }

    // ImprovePath done when open empty or min f_ε ≥ g(goal).
    const bool improve_done =
        open_->empty() ||
        (goal_node_ != nullptr && open_->top()->f >= goal_node_->g);

    if (improve_done) {
        if (goal_node_ == nullptr) {
            status_ = SearchStatus::Failure;
            return status_;
        }
        // Repair under a smaller ε, or finish when target reached.
        if (eps_ <= eps_target_ + 1e-12) {
            status_ = SearchStatus::Success;
            return status_;
        }
        eps_ = std::max(eps_target_, eps_ - eps_step_);
        /// NOTE: Why we don't need to update goal_node_? because goal_node's heuristic is 0.
        open_->clear();
        for (auto& kv : nodes_) {
            if (kv.second->status == NodeStatus::Closed) {
                kv.second->status = NodeStatus::New;
            }
        }
        for (SearchNode* n : incons_) {
            n->status = NodeStatus::Open;
        }
        incons_.clear();
        for (auto& kv : nodes_) {
            SearchNode* n = kv.second.get();
            if (n->status != NodeStatus::Open) {
                continue;
            }
            set_node_costs(n, n->g, heuristic_at(n->coord));
            open_->push(n);
        }
        return status_;
    }

    auto* current = static_cast<SearchNode*>(open_->pop());
    current->status = NodeStatus::Closed;
    last_expanded_ = current;
    ++expansions_;
    before_expand(current);
    expand(current);
    return status_;
}

}  // namespace pipe_design::planner
