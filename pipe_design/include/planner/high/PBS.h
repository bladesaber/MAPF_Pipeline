#pragma once

#include "conflict/ConflictAnalyzer.h"
#include "grid/CompositeGrid.h"
#include "heuristic/Heuristic.h"
#include "planner/high/HighLevelPlanner.h"
#include "planner/high/SequenceReplanner.h"
#include "planner/low/AStar.h"
#include "planner/low/SearchNode.h"
#include "planner/motion/ActionModel.h"
#include "planner/multi/PrioritySequencer.h"
#include "planner/open/OpenSet.h"
#include "topology/Types.h"

#include <cstddef>
#include <string>
#include <vector>

namespace pipe_design::planner {

/// Iterative priority repair (task-level), not a CBS-style search tree.
///
/// Each `step()`: `[V]` on current Tasks → if conflicts, `SequenceReplanner` updates
/// the task order → `PrioritySequencer` re-routes. No `select_conflict` / PTNode / open set.
class PBS : public HighLevelPlanner {
public:
    PBS() = default;
    ~PBS() override = default;

    /// Initial order = `plans.ordered_tasks()` ids; routes once (`depth` = 0).
    void reset(
        const topology::GlobalTopologyPlan& plans,
        const grid::CompositeGrid& grid,
        AStar<SearchNode>& low,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions,
        PrioritySequencer& sequencer,
        conflict::ConflictAnalyzer& analyzer,
        SequenceReplanner& replanner);

    SearchStatus step() override;
    std::vector<topology::Task> solution() const override { return solution_; }

    const std::vector<std::string>& order() const { return order_; }
    const std::vector<topology::Task>& tasks() const { return tasks_; }

    /// Successful sequence repairs so far (root route = 0).
    std::size_t depth() const { return depth_; }
    /// Sum of Task path costs after the last successful `route()`.
    double cost() const { return cost_; }
    /// Conflict count from the last `analyze` in `step()` (0 before first step).
    std::size_t last_conflict_count() const { return last_conflict_count_; }

protected:
    /// Run PrioritySequencer under `order_`; fills `tasks_` / `cost_`. False on failure.
    bool route();

private:
    const topology::GlobalTopologyPlan* plans_ = nullptr;
    const grid::CompositeGrid* grid_ = nullptr;
    AStar<SearchNode>* low_ = nullptr;
    const heuristic::Heuristic* heuristic_ = nullptr;
    OpenSet* open_set_ = nullptr;
    const motion::ActionModel* actions_ = nullptr;
    PrioritySequencer* sequencer_ = nullptr;
    conflict::ConflictAnalyzer* analyzer_ = nullptr;
    SequenceReplanner* replanner_ = nullptr;

    std::vector<std::string> order_;
    std::vector<topology::Task> tasks_;
    std::vector<topology::Task> solution_;
    std::size_t depth_ = 0;
    double cost_ = 0.0;
    std::size_t last_conflict_count_ = 0;
};

}  // namespace pipe_design::planner
