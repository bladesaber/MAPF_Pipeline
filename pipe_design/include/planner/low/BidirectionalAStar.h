#pragma once

#include "grid/CompositeGrid.h"
#include "grid/Types.h"
#include "heuristic/Heuristic.h"
#include "planner/low/AStar.h"
#include "planner/low/LowLevelPlanner.h"
#include "planner/low/SearchNode.h"
#include "planner/motion/ActionModel.h"
#include "planner/open/OpenSet.h"

#include <vector>

namespace pipe_design::planner {

/// Bidirectional A* over plain cell states (`AStar<SearchNode>` only).
///
/// Holds two independent A* searches (start→goal and goal→start). Weight is
/// forwarded to both via `AStar::set_heuristic_weight`. Caller owns both open sets.
///
/// Assumes undirected / reversible grid edges (6/26-connect).
class BidirectionalAStar : public LowLevelPlanner {
public:
    BidirectionalAStar() = default;
    ~BidirectionalAStar() override = default;

    void reset(
        const grid::CompositeGrid& grid,
        const std::vector<grid::CellCoord>& starts,
        const std::vector<grid::CellCoord>& goals,
        float pipe_radius,
        const heuristic::Heuristic& heuristic,
        OpenSet& open_forward,
        OpenSet& open_backward,
        const motion::ActionModel& actions);

    void reset(
        const grid::CompositeGrid& grid,
        grid::CellCoord start,
        grid::CellCoord goal,
        float pipe_radius,
        const heuristic::Heuristic& heuristic,
        OpenSet& open_forward,
        OpenSet& open_backward,
        const motion::ActionModel& actions);

    SearchStatus step() override;
    Path path() const override;

    void set_heuristic_weight(double weight);
    double heuristic_weight() const { return fwd_.heuristic_weight(); }

    const AStar<SearchNode>& forward() const { return fwd_; }
    const AStar<SearchNode>& backward() const { return bwd_; }

private:
    enum class PathMode {
        None,
        ForwardOnly,   // forward reached a goal
        BackwardOnly,  // backward reached a start
        Meet,          // frontiers met on a closed cell
    };

    AStar<SearchNode> fwd_;
    AStar<SearchNode> bwd_;

    bool prefer_forward_ = true;
    PathMode path_mode_ = PathMode::None;
    const SearchNode* meet_fwd_ = nullptr;
    const SearchNode* meet_bwd_ = nullptr;
    double meet_cost_ = 0.0;
};

}  // namespace pipe_design::planner
