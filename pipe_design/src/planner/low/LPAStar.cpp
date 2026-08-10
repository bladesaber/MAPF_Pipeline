#include "planner/low/LPAStar.h"

#include "geom/Heading.h"

#include <algorithm>
#include <cmath>

namespace pipe_design::planner {

void LPAStar::reset(
    const grid::CompositeGrid& grid,
    grid::CellCoord start,
    grid::CellCoord goal,
    float pipe_radius,
    const heuristic::Heuristic& heuristic,
    Open& open,
    const motion::ActionModel& actions) {
    grid_ = &grid;
    pipe_radius_ = pipe_radius;
    heuristic_ = &heuristic;
    open_ = &open;
    actions_ = &actions;
    expansions_ = 0;
    nodes_.clear();
    open_->clear();
    start_node_ = nullptr;
    goal_node_ = nullptr;
    start_coord_ = start;
    goal_coord_ = goal;

    if (!grid_->in_bounds(start) || !grid_->is_free(start, pipe_radius_) ||
        !grid_->in_bounds(goal) || !grid_->is_free(goal, pipe_radius_)) {
        status_ = SearchStatus::Failure;
        return;
    }

    start_node_ = ensure_node(LpaSearchNode::make_key(start));
    goal_node_ = ensure_node(LpaSearchNode::make_key(goal));
    start_node_->rhs = 0.0;
    start_node_->g = LpaSearchNode::kInf;
    start_node_->parent = nullptr;
    update_vertex(start_node_);
    status_ = SearchStatus::Running;
}

void LPAStar::update_vertex(LpaSearchNode* u) {
    const bool in_open = open_->contains(u);
    if (u->g != u->rhs) {
        if (in_open) {
            open_->remove(u);
        }
        open_->push(u);
    } else if (in_open) {
        open_->remove(u);
    }
}

double LPAStar::edge_cost(const grid::CellCoord& from, const grid::CellCoord& to) const {
    if (!grid_->in_bounds(to) ||
        !grid_->is_move_free(from, to, pipe_radius_)) {
        return LpaSearchNode::kInf;
    }
    std::vector<motion::Successor> local;
    actions_->successors(from, geom::Heading{}, local);
    for (const motion::Successor& s : local) {
        if (s.cell == to) {
            return s.cost;
        }
    }
    return LpaSearchNode::kInf;
}

LpaSearchNode* LPAStar::ensure_node(const Key& key) {
    const auto it = nodes_.find(key);
    if (it != nodes_.end()) {
        return it->second.get();
    }
    auto node = std::make_unique<LpaSearchNode>(LpaSearchNode::from_key(key));
    node->h = heuristic_->estimate(key.coord, goal_coord_);
    LpaSearchNode* raw = node.get();
    nodes_.emplace(key, std::move(node));
    return raw;
}

LpaSearchNode* LPAStar::find_node(const Key& key) {
    const auto it = nodes_.find(key);
    return it == nodes_.end() ? nullptr : it->second.get();
}

const LpaSearchNode* LPAStar::find_node(const Key& key) const {
    const auto it = nodes_.find(key);
    return it == nodes_.end() ? nullptr : it->second.get();
}

SearchStatus LPAStar::step() {
    if (status_ != SearchStatus::Running) {
        return status_;
    }
    if (open_ == nullptr || goal_node_ == nullptr) {
        status_ = SearchStatus::Failure;
        return status_;
    }

    // Also keep going while g > rhs at goal (overconsistent); use != for stop.
    const LPAKey goal_key = lpa_key_of(goal_node_);
    if (!(open_->top_key() < goal_key) && goal_node_->g == goal_node_->rhs) {
        status_ = std::isfinite(goal_node_->g) ? SearchStatus::Success : SearchStatus::Failure;
        return status_;
    }
    if (open_->empty()) {
        status_ = SearchStatus::Failure;
        return status_;
    }

    LpaSearchNode* u = open_->pop();
    ++expansions_;

    if (u->g > u->rhs) {
        // Overconsistent: g ← rhs; improve successors via edge_cost
        u->g = u->rhs;
        neighbor_buf_.clear();
        actions_->successors(u->coord, geom::Heading{}, neighbor_buf_);
        for (const motion::Successor& s : neighbor_buf_) {
            if (!grid_->in_bounds(s.cell)) {
                continue;
            }
            LpaSearchNode* v = ensure_node(LpaSearchNode::make_key(s.cell));
            const double cand = u->g + edge_cost(u->coord, s.cell);
            if (v->rhs > cand) {
                v->parent = u;
                v->rhs = cand;
                update_vertex(v);
            }
        }
    } else if (u->g < u->rhs) {  // in original paper, assume node whose g == rhs will never happen in open set, so use else
        // Underconsistent: g ← ∞; raw-update states that had parent u
        u->g = LpaSearchNode::kInf;
        neighbor_buf_.clear();
        actions_->successors(u->coord, geom::Heading{}, neighbor_buf_);
        const std::vector<motion::Successor> succs = neighbor_buf_;

        for (const motion::Successor& s : succs) {
            if (!grid_->in_bounds(s.cell)) {
                continue;
            }
            LpaSearchNode* v = ensure_node(LpaSearchNode::make_key(s.cell));
            if (v != start_node_ && v->lpa_parent() == u) {
                v->rhs = LpaSearchNode::kInf;
                v->parent = nullptr;
                neighbor_buf_.clear();
                actions_->successors(v->coord, geom::Heading{}, neighbor_buf_);
                for (const motion::Successor& nb : neighbor_buf_) {
                    if (!grid_->in_bounds(nb.cell)) {
                        continue;
                    }
                    LpaSearchNode* pred = find_node(LpaSearchNode::make_key(nb.cell));
                    if (pred == nullptr) {
                        continue;
                    }
                    const double cand = pred->g + edge_cost(pred->coord, v->coord);
                    if (cand < v->rhs) {
                        v->rhs = cand;
                        v->parent = pred;
                    }
                }
            }
            update_vertex(v);
        }
        update_vertex(u);
    }
    return status_;
}

void LPAStar::update_obstacles(const std::vector<grid::CellCoord>& changed) {
    if (grid_ == nullptr || open_ == nullptr || start_node_ == nullptr) {
        return;
    }

    for (const grid::CellCoord& cell : changed) {
        if (!grid_->in_bounds(cell)) {
            continue;
        }
        LpaSearchNode* a = ensure_node(LpaSearchNode::make_key(cell));
        neighbor_buf_.clear();
        actions_->successors(cell, geom::Heading{}, neighbor_buf_);
        const std::vector<motion::Successor> succs = neighbor_buf_;

        for (const motion::Successor& s : succs) {
            if (!grid_->in_bounds(s.cell)) {
                continue;
            }
            LpaSearchNode* b = ensure_node(LpaSearchNode::make_key(s.cell));

            // Directed edges (a→b) and (b→a); Fig.7 Main edge-cost update.
            for (int dir = 0; dir < 2; ++dir) {
                LpaSearchNode* u = (dir == 0) ? a : b;
                LpaSearchNode* v = (dir == 0) ? b : a;
                const double c_new =
                    (dir == 0) ? edge_cost(cell, s.cell) : edge_cost(s.cell, cell);

                if (v->rhs > u->g + c_new) {
                    v->parent = u;
                    v->rhs = u->g + c_new;
                    update_vertex(v);
                } else if (v != start_node_ && v->lpa_parent() == u) {
                    v->rhs = LpaSearchNode::kInf;
                    v->parent = nullptr;
                    neighbor_buf_.clear();
                    actions_->successors(v->coord, geom::Heading{}, neighbor_buf_);
                    for (const motion::Successor& nb : neighbor_buf_) {
                        if (!grid_->in_bounds(nb.cell)) {
                            continue;
                        }
                        LpaSearchNode* pred = find_node(LpaSearchNode::make_key(nb.cell));
                        if (pred == nullptr) {
                            continue;
                        }
                        const double c = edge_cost(pred->coord, v->coord);
                        const double cand = pred->g + c;
                        if (cand < v->rhs) {
                            v->rhs = cand;
                            v->parent = pred;
                        }
                    }
                    update_vertex(v);
                }
            }
        }
    }
    status_ = SearchStatus::Running;
}

Path LPAStar::path() const {
    if (status_ != SearchStatus::Success || goal_node_ == nullptr ||
        !std::isfinite(goal_node_->g)) {
        return {};
    }
    Path out;
    for (const LpaSearchNode* n = goal_node_; n != nullptr; n = n->lpa_parent()) {
        out.cells.push_back(n->coord);
    }
    std::reverse(out.cells.begin(), out.cells.end());
    out.cost = goal_node_->g;
    return out;
}

}  // namespace pipe_design::planner
