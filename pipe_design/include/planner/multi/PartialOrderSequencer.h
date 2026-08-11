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
#include <unordered_set>
#include <vector>

namespace pipe_design::planner {

/// One priority edge: `higher_id` must be planned / claim space before `lower_id`
/// (`higher ≺ lower`). Outsider may encode intra-group grow_order as such edges.
struct PriorityConstraint {
    std::string higher_id;
    std::string lower_id;

    PriorityConstraint() = default;
    PriorityConstraint(std::string higher_id_, std::string lower_id_)
        : higher_id(std::move(higher_id_))
        , lower_id(std::move(lower_id_)) {}
};

/// Per-task planning context + result for warm-start (`snapshot()`).
struct PartialOrderPastEntry {
    /// Transitive higher tasks from other groups (obstacle donors).
    std::unordered_set<std::string> foreign_higher;
    /// Same-group tasks earlier in the topo schedule (ConnectedSet donors).
    std::unordered_set<std::string> same_group_prev;
    topology::Task task;
};

/// Exported leaf state for partial-order path reuse (not the constraint list).
struct PartialOrderPast {
    std::unordered_map<std::string, PartialOrderPastEntry> by_id;
};

/// Like `PrioritySequencer`, but schedule from a partial order (constraint set).
///
/// `reset` runs `topological_sort` on `plans.enumerate_tasks()` + `constraints`.
/// Cycle → session Failure. Tie-break among ready nodes follows enumerate order.
class PartialOrderSequencer : public MultiQueryPlanner {
public:
    PartialOrderSequencer() = default;
    ~PartialOrderSequencer() override = default;

    /// `grid` is read-only (static + entry chain only).
    /// Low-level failure always ends the session as Failure (PBS leaf semantics).
    /// `past` enables path reuse in `step_task` via `can_reuse` (pass empty to disable).
    void reset(
        const topology::GlobalTopologyPlan& plans,
        const std::vector<PriorityConstraint>& constraints,
        const grid::CompositeGrid& grid,
        AStar<SearchNode>& low,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions,
        const PartialOrderPast& past);

    /// Like `reset` with an empty `past` (no reuse).
    void reset(
        const topology::GlobalTopologyPlan& plans,
        const std::vector<PriorityConstraint>& constraints,
        const grid::CompositeGrid& grid,
        AStar<SearchNode>& low,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions);

    SearchStatus step_task() override;
    Path last_path() const override;

    /// Freeze per-task foreign_higher / same_group_prev / Task for later reuse.
    PartialOrderPast snapshot() const;

    /// Reuse iff past succeeded, foreign_higher / same_group_prev unchanged, and no
    /// donor appears in `replanned_ids_` this session.
    bool can_reuse(const std::string& task_id, const PartialOrderPast& past) const;

    std::vector<topology::Task> tasks() const { return tasks_; }
    const std::vector<std::string>& order() const { return order_; }
    const std::vector<PriorityConstraint>& constraints() const { return constraints_; }
    const std::vector<std::string>& replanned_ids() const { return replanned_ids_; }

    bool any_failure() const {
        for (const topology::Task& t : tasks_) {
            if (t.status == SearchStatus::Failure) {
                return true;
            }
        }
        return false;
    }

    /// Kahn topo over `seed` nodes + `constraints`. Empty vector if cycle / unknown id.
    /// Ready-set tie-break follows `seed` order.
    static std::vector<std::string> topological_sort(
        const std::vector<std::string>& seed,
        const std::vector<PriorityConstraint>& constraints);

    /// Task ids that must precede `task_id` (transitive) under current constraints_.
    std::unordered_set<std::string> higher_of(const std::string& task_id) const;

private:
    const topology::GroupTopologyPlan* group_of(const topology::GroupId& id) const;
    const std::vector<grid::CellCoord>& set_of(const topology::TerminalId& id) const;
    void merge_path(const topology::TerminalId& u, const topology::TerminalId& v, const Path& path);
    void rebuild_higher_graph();

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

    std::vector<PriorityConstraint> constraints_;
    std::vector<std::string> order_;
    /// reverse adj: lower_id → higher_ids (for higher_of DFS).
    std::unordered_map<std::string, std::vector<std::string>> higher_adj_;

    std::vector<topology::Task> tasks_;
    std::size_t cursor_ = 0;
    std::size_t last_finished_ = static_cast<std::size_t>(-1);
    /// Task ids that ran low-level search this session (not reused).
    std::vector<std::string> replanned_ids_;
    PartialOrderPast past_;

    std::unordered_map<topology::TerminalId, std::size_t> term_comp_;
    std::vector<std::vector<grid::CellCoord>> comps_;
    std::vector<Committed> committed_;
};

}  // namespace pipe_design::planner
