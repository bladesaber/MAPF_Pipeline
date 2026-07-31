#include "planner/low/ANAStar.h"

#include <cmath>
#include <limits>

namespace pipe_design::planner {

double ANAStar::incumbent_cost() const {
    return goal_node_ == nullptr ? std::numeric_limits<double>::infinity()
                                 : goal_node_->g;
}

double ANAStar::combine_key(double g, double h) const {
    const double G = incumbent_cost();
    if (!std::isfinite(G)) {
        // Before first solution: greedy on h (same as maximizing e with G=∞).
        return h;
    }
    if (h <= 1e-15) {
        return -std::numeric_limits<double>::infinity();
    }
    // Max e = (G − g) / h  →  min-heap key −e.
    return -(G - g) / h;
}

Path ANAStar::path() const {
    if (goal_node_ == nullptr) {
        return {};
    }
    return reconstruct(goal_node_);
}

void ANAStar::expand(SearchNode* current) {
    successor_buf_.clear();
    actions_->successors(current->coord, geom::Heading{}, successor_buf_);

    const double G = incumbent_cost();

    for (const motion::Successor& succ : successor_buf_) {
        if (!grid_->in_bounds(succ.cell) || !grid_->is_free(succ.cell, pipe_radius_)) {
            continue;
        }

        SearchNode* neighbor = ensure_node(SearchNode::make_key(succ.cell));
        const double tentative_g = current->g + succ.cost;
        const double h = heuristic_at(succ.cell);

        // Prune dominated by incumbent.
        if (std::isfinite(G) && tentative_g + h >= G) {
            continue;
        }

        if (neighbor->status == NodeStatus::Closed) {
            if (tentative_g >= neighbor->g) {
                continue;
            }
            // Reopen with better g.
        } else {
            /// TODO: need to check whether this part is correct.
            const bool unvisited = neighbor->status == NodeStatus::New &&
                neighbor->parent == nullptr && neighbor->h == 0.0 && neighbor->f == 0.0;
            if (!unvisited && tentative_g >= neighbor->g) {
                continue;
            }
        }

        const bool in_open = neighbor->status == NodeStatus::Open;
        neighbor->parent = current;
        set_node_costs(neighbor, tentative_g, h);
        neighbor->status = NodeStatus::Open;
        if (!in_open) {
            open_->push(neighbor);
        } else if (!open_->decrease_key(neighbor)) {
            open_->push(neighbor);
        }
    }
}

SearchStatus ANAStar::step() {
    if (status_ != SearchStatus::Running) {
        return status_;
    }
    if (open_ == nullptr || open_->empty()) {
        status_ = goal_node_ != nullptr ? SearchStatus::Success : SearchStatus::Failure;
        return status_;
    }

    auto* current = static_cast<SearchNode*>(open_->pop());

    current->status = NodeStatus::Closed;
    last_expanded_ = current;
    ++expansions_;
    before_expand(current);

    if (is_goal(current->coord)) {
        if (goal_node_ == nullptr || current->g < goal_node_->g) {
            goal_node_ = current;
        }
        open_->clear();
        for (auto& kv : nodes_) {
            SearchNode* n = kv.second.get();
            if (n->status != NodeStatus::Open) {
                continue;
            }
            if (n->g + n->h >= goal_node_->g) {
                n->status = NodeStatus::New;
                continue;
            }
            set_node_costs(n, n->g, n->h);
            open_->push(n);
        }
        return status_;
    }

    expand(current);
    return status_;
}

SearchStatus ANAStar::search(std::size_t max_expansions) {
    while (status_ == SearchStatus::Running && expansions_ < max_expansions) {
        step();
    }
    return status_;
}

}  // namespace pipe_design::planner
