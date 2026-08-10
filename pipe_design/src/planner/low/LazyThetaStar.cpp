#include "planner/low/LazyThetaStar.h"

#include "geom/Heading.h"

#include <limits>

namespace pipe_design::planner {

void LazyThetaStar::set_vertex(SearchNode* current) {
    auto* parent = static_cast<SearchNode*>(current->parent);
    if (parent == nullptr) {
        return;
    }
    if (line_of_sight(*grid_, parent->coord, current->coord, pipe_radius_)) {
        return;
    }

    double best_g = std::numeric_limits<double>::infinity();
    SearchNode* best_parent = nullptr;

    successor_buf_.clear();
    actions_->successors(current->coord, geom::Heading{}, successor_buf_);
    for (const motion::Successor& succ : successor_buf_) {
        if (!grid_->in_bounds(succ.cell) ||
            !grid_->is_move_free(current->coord, succ.cell, pipe_radius_)) {
            continue;
        }
        SearchNode* nbr = find_node(SearchNode::make_key(succ.cell));
        if (nbr == nullptr || nbr->status != NodeStatus::Closed) {
            continue;
        }
        // Undirected grid edge: cost(nbr → current) == succ.cost (current → nbr).
        const double g = nbr->g + succ.cost;
        if (g < best_g) {
            best_g = g;
            best_parent = nbr;
        }
    }

    if (best_parent == nullptr) {
        return;
    }
    current->parent = best_parent;
    set_node_costs(current, best_g, heuristic_at(current->coord));
}

void LazyThetaStar::before_expand(SearchNode* current) {
    set_vertex(current);
}

void LazyThetaStar::expand(SearchNode* current) {
    successor_buf_.clear();
    actions_->successors(current->coord, geom::Heading{}, successor_buf_);

    for (const motion::Successor& succ : successor_buf_) {
        if (!grid_->in_bounds(succ.cell) ||
            !grid_->is_move_free(current->coord, succ.cell, pipe_radius_)) {
            continue;
        }

        SearchNode* neighbor = ensure_node(SearchNode::make_key(succ.cell));
        if (neighbor->status == NodeStatus::Closed) {
            continue;
        }

        // Assume path-2 via parent(current) without LOS (validated in set_vertex).
        auto* parent = static_cast<SearchNode*>(current->parent);
        double tentative_g = current->g + succ.cost;
        SearchNode* tentative_parent = current;

        if (parent != nullptr) {
            const double via_parent =
                parent->g + euclidean_cell_cost(parent->coord, neighbor->coord, step_cost_);
            if (via_parent < tentative_g) {
                tentative_g = via_parent;
                tentative_parent = parent;
            }
        }

        const bool discovered = neighbor->status == NodeStatus::New;
        if (!discovered && tentative_g >= neighbor->g) {
            continue;
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
}

}  // namespace pipe_design::planner
