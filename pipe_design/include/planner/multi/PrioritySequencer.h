#pragma once

#include "grid/CompositeGrid.h"
#include "grid/ObstacleChain.h"
#include "grid/StaticClearanceGrid.h"
#include "heuristic/Heuristic.h"
#include "planner/low/AStar.h"
#include "planner/low/SearchNode.h"
#include "planner/motion/ActionModel.h"
#include "planner/multi/MultiQueryPlanner.h"
#include "planner/open/OpenSet.h"
#include "topology/Types.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pipe_design::planner {

/// Cross-group sequential routing: same-group → ConnectedSets; other-group → chain obstacles.
class PrioritySequencer : public MultiQueryPlanner {
public:
    PrioritySequencer() = default;
    ~PrioritySequencer() override = default;

    /// Schedule from `plans.ordered_tasks()`.
    /// `grid` is read-only: only its static field + entry chain are used as base.
    void reset(
        const topology::GlobalTopologyPlan& plans,
        const grid::CompositeGrid& grid,
        AStar<SearchNode>& low,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions);

    /// Explicit task_id order (e.g. PBS priority list).
    void reset(
        const topology::GlobalTopologyPlan& plans,
        const std::vector<std::string>& order,
        const grid::CompositeGrid& grid,
        AStar<SearchNode>& low,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions);

    SearchStatus step_task() override;
    Path last_path() const override;

    /// Copy of current Task results (safe to keep after another `reset`).
    std::vector<topology::Task> tasks() const { return tasks_; }

    /// True if any Task finished with `SearchStatus::Failure` (session may still be Success).
    bool any_failure() const {
        for (const topology::Task& t : tasks_) {
            if (t.status == SearchStatus::Failure) {
                return true;
            }
        }
        return false;
    }

private:
    const topology::GroupTopologyPlan* group_of(const topology::GroupId& id) const;
    const std::vector<grid::CellCoord>& set_of(const topology::TerminalId& id) const;
    void merge_path(const topology::TerminalId& u, const topology::TerminalId& v, const Path& path);

    struct Committed {
        topology::GroupId group_id;
        std::size_t task_index = 0;
        float radius = 0.f;
    };

    const topology::GlobalTopologyPlan* plans_ = nullptr;
    const grid::StaticClearanceGrid* static_grid_ = nullptr;
    std::shared_ptr<const grid::ObstacleChain> base_chain_;
    AStar<SearchNode>* low_ = nullptr;
    const heuristic::Heuristic* heuristic_ = nullptr;
    OpenSet* open_ = nullptr;
    const motion::ActionModel* actions_ = nullptr;

    std::vector<topology::Task> tasks_;
    std::size_t cursor_ = 0;
    std::size_t last_finished_ = static_cast<std::size_t>(-1);

    std::unordered_map<topology::TerminalId, std::size_t> term_comp_;
    std::vector<std::vector<grid::CellCoord>> comps_;
    std::vector<Committed> committed_;
};

}  // namespace pipe_design::planner
