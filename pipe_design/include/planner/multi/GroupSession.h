#pragma once

#include "grid/CompositeGrid.h"
#include "heuristic/Heuristic.h"
#include "planner/low/AStar.h"
#include "planner/low/SearchNode.h"
#include "planner/motion/ActionModel.h"
#include "planner/multi/MultiQueryPlanner.h"
#include "planner/open/OpenSet.h"
#include "topology/Types.h"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace pipe_design::planner {

/// Single-group multi-query: fixed obstacles; same-group paths grow ConnectedSets.
class GroupSession : public MultiQueryPlanner {
public:
    GroupSession() = default;
    ~GroupSession() override = default;

    /// `grid` is read-only for the session (never mutated).
    void reset(
        const topology::GroupTopologyPlan& plan,
        const grid::CompositeGrid& grid,
        AStar<SearchNode>& low,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions);

    SearchStatus step_task() override;
    Path last_path() const override;

    /// Copy of current Task results (safe to keep after another `reset`).
    std::vector<topology::Task> tasks() const { return tasks_; }

private:
    const std::vector<grid::CellCoord>& set_of(const topology::TerminalId& id) const;
    void merge_path(const topology::TerminalId& u, const topology::TerminalId& v, const Path& path);

    const topology::GroupTopologyPlan* plan_ = nullptr;
    const grid::CompositeGrid* grid_ = nullptr;
    AStar<SearchNode>* low_ = nullptr;
    const heuristic::Heuristic* heuristic_ = nullptr;
    OpenSet* open_ = nullptr;
    const motion::ActionModel* actions_ = nullptr;

    std::vector<topology::Task> tasks_;
    std::size_t cursor_ = 0;
    std::size_t last_finished_ = static_cast<std::size_t>(-1);

    std::unordered_map<topology::TerminalId, std::size_t> term_comp_;
    std::vector<std::vector<grid::CellCoord>> comps_;
};

}  // namespace pipe_design::planner
