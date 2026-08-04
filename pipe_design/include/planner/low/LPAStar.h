#pragma once

#include "grid/CompositeGrid.h"
#include "grid/Types.h"
#include "heuristic/Heuristic.h"
#include "planner/low/AStar.h"  // SearchStatus
#include "planner/low/LpaSearchNode.h"
#include "planner/low/Path.h"
#include "planner/motion/ActionModel.h"
#include "planner/open/LPAOpenSet.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace pipe_design::planner {

/// Lifelong Planning A* (Koenig & Likhachev, optimized). Single start / goal.
///
/// Does not own grid, heuristic, open set, or action model.
/// `update_vertex`: heap membership only (Fig.7).
/// `step`: one ComputeShortestPath iteration.
/// `update_obstacles`: edge-cost changes after grid occupancy updates.
class LPAStar {
public:
    using Key = LpaSearchNode::Key;
    using Open = LPAOpenSet<LpaSearchNode, LpaKeyCompare>;

    static constexpr std::size_t kUnlimited =
        std::numeric_limits<std::size_t>::max();

    LPAStar() = default;

    void reset(
        const grid::CompositeGrid& grid,
        grid::CellCoord start,
        grid::CellCoord goal,
        float pipe_radius,
        const heuristic::Heuristic& heuristic,
        Open& open,
        const motion::ActionModel& actions);

    SearchStatus step();
    SearchStatus search(std::size_t max_expansions = kUnlimited);

    /// Caller has already updated the grid; `changed` cells may have new edge costs.
    void update_obstacles(const std::vector<grid::CellCoord>& changed);

    SearchStatus status() const { return status_; }
    std::size_t expansions() const { return expansions_; }

    Path path() const;

    const LpaSearchNode* start_node() const { return start_node_; }
    const LpaSearchNode* goal_node() const { return goal_node_; }
    const std::unordered_map<Key, std::unique_ptr<LpaSearchNode>>& nodes() const {
        return nodes_;
    }

    LpaSearchNode* find_node(const Key& key);
    const LpaSearchNode* find_node(const Key& key) const;

protected:
    /// Fig.7 UpdateVertex: only open-set Insert / Update / Remove.
    void update_vertex(LpaSearchNode* u);

    LpaSearchNode* ensure_node(const Key& key);
    /// Occupancy-aware edge cost (∞ if either endpoint blocked for pipe_radius).
    double edge_cost(const grid::CellCoord& from, const grid::CellCoord& to) const;

    const grid::CompositeGrid* grid_ = nullptr;
    float pipe_radius_ = 0.f;
    const heuristic::Heuristic* heuristic_ = nullptr;
    Open* open_ = nullptr;
    const motion::ActionModel* actions_ = nullptr;

    grid::CellCoord start_coord_{};
    grid::CellCoord goal_coord_{};
    LpaSearchNode* start_node_ = nullptr;
    LpaSearchNode* goal_node_ = nullptr;

    std::unordered_map<Key, std::unique_ptr<LpaSearchNode>> nodes_;
    std::vector<motion::Successor> neighbor_buf_;

    SearchStatus status_ = SearchStatus::Ready;
    std::size_t expansions_ = 0;
};

}  // namespace pipe_design::planner
