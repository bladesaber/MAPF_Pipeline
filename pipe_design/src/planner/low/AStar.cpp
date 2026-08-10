#include "planner/low/AStar.h"

#include "geom/Heading.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace pipe_design::planner {

template <typename Node>
void AStar<Node>::reset(
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

template <typename Node>
void AStar<Node>::reset(
    const grid::CompositeGrid& grid,
    const std::vector<grid::CellCoord>& starts,
    const std::vector<grid::CellCoord>& goals,
    float pipe_radius,
    const heuristic::Heuristic& heuristic,
    OpenSet& open,
    const motion::ActionModel& actions) {
    grid_ = &grid;
    pipe_radius_ = pipe_radius;
    heuristic_ = &heuristic;
    open_ = &open;
    actions_ = &actions;
    expansions_ = 0;
    last_expanded_ = nullptr;
    goal_node_ = nullptr;
    nodes_.clear();
    goals_.clear();
    open_->clear();

    for (const grid::CellCoord& g : goals) {
        if (grid_->in_bounds(g) && grid_->is_free(g, pipe_radius_)) {
            goals_.insert(g);
        }
    }

    std::vector<grid::CellCoord> valid_starts;
    valid_starts.reserve(starts.size());
    for (const grid::CellCoord& s : starts) {
        if (grid_->in_bounds(s) && grid_->is_free(s, pipe_radius_)) {
            valid_starts.push_back(s);
        }
    }

    if (valid_starts.empty() || goals_.empty()) {
        status_ = SearchStatus::Failure;
        return;
    }

    for (const grid::CellCoord& s : valid_starts) {
        if (is_goal(s)) {
            Node* node = ensure_node(Node::make_key(s));
            set_node_costs(node, 0.0, 0.0);
            node->parent = nullptr;
            node->status = NodeStatus::Closed;
            goal_node_ = node;
            last_expanded_ = node;
            status_ = SearchStatus::Success;
            return;
        }
    }

    for (const grid::CellCoord& s : valid_starts) {
        Node* node = ensure_node(Node::make_key(s));
        set_node_costs(node, 0.0, heuristic_at(s));
        node->parent = nullptr;
        node->status = NodeStatus::Open;
        open_->push(node);
    }
    status_ = SearchStatus::Running;
}

template <typename Node>
SearchStatus AStar<Node>::step() {
    if (status_ != SearchStatus::Running) {
        return status_;
    }
    if (open_ == nullptr || open_->empty()) {
        status_ = SearchStatus::Failure;
        return status_;
    }

    auto* current = static_cast<Node*>(open_->pop());
    current->status = NodeStatus::Closed;
    last_expanded_ = current;
    ++expansions_;

    before_expand(current);

    if (is_goal(current->coord)) {
        goal_node_ = current;
        status_ = SearchStatus::Success;
        return status_;
    }

    expand(current);

    if (open_->empty()) {
        status_ = SearchStatus::Failure;
    }
    return status_;
}

template <typename Node>
Path AStar<Node>::path() const {
    if (status_ != SearchStatus::Success || goal_node_ == nullptr) {
        return {};
    }
    return reconstruct(goal_node_);
}

template <typename Node>
bool AStar<Node>::is_goal(const grid::CellCoord& cell) const {
    return goals_.find(cell) != goals_.end();
}

template <typename Node>
Node* AStar<Node>::find_node(const Key& key) {
    const auto it = nodes_.find(key);
    return it == nodes_.end() ? nullptr : it->second.get();
}

template <typename Node>
const Node* AStar<Node>::find_node(const Key& key) const {
    const auto it = nodes_.find(key);
    return it == nodes_.end() ? nullptr : it->second.get();
}

template <typename Node>
void AStar<Node>::set_heuristic_weight(double weight) {
    heuristic_weight_ = weight;
}

template <typename Node>
double AStar<Node>::combine_key(double g, double h) const {
    return g + heuristic_weight_ * h;
}

template <typename Node>
Node* AStar<Node>::ensure_node(const Key& key) {
    const auto it = nodes_.find(key);
    if (it != nodes_.end()) {
        return it->second.get();
    }
    auto node = std::make_unique<Node>(Node::from_key(key));
    Node* raw = node.get();
    nodes_.emplace(key, std::move(node));
    return raw;
}

template <typename Node>
double AStar<Node>::heuristic_at(const grid::CellCoord& from) const {
    double best = std::numeric_limits<double>::infinity();
    for (const grid::CellCoord& g : goals_) {
        best = std::min(best, heuristic_->estimate(from, g));
    }
    return best;
}

template <typename Node>
void AStar<Node>::set_node_costs(Node* node, double g, double h) const {
    node->g = g;
    node->h = h;
    node->f = combine_key(g, h);
}

template <typename Node>
Path AStar<Node>::reconstruct(const Node* goal) const {
    Path out;
    for (const SearchNode* n = goal; n != nullptr; n = n->parent) {
        out.cells.push_back(n->coord);
    }
    std::reverse(out.cells.begin(), out.cells.end());
    out.cost = goal->g;
    return out;
}

template <typename Node>
void AStar<Node>::expand(Node* current) {
    successor_buf_.clear();
    // Cell A*: no incoming heading → ActionModel skips turn penalty.
    actions_->successors(current->coord, geom::Heading{}, successor_buf_);

    for (const motion::Successor& succ : successor_buf_) {
        if (!grid_->in_bounds(succ.cell) ||
            !grid_->is_move_free(current->coord, succ.cell, pipe_radius_)) {
            continue;
        }

        Node* neighbor = ensure_node(Node::make_key(succ.cell));
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
}

template class AStar<SearchNode>;
template class AStar<DirectedSearchNode>;

}  // namespace pipe_design::planner
