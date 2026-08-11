#pragma once

#include "geom/Types.h"
#include "grid/CompositeGrid.h"
#include "grid/ObstacleChain.h"
#include "grid/StaticClearanceGrid.h"
#include "heuristic/Heuristic.h"
#include "planner/low/AStar.h"
#include "planner/low/Path.h"
#include "planner/low/SearchNode.h"
#include "planner/motion/ActionModel.h"
#include "planner/multi/MultiQueryPlanner.h"
#include "planner/open/OpenSet.h"
#include "topology/Types.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pipe_design::planner {

/// Bake a path into an ObstacleChain (sphere / capsule runs). Shared by sequencers.
inline std::shared_ptr<const grid::ObstacleChain> chain_with_path(
    std::shared_ptr<const grid::ObstacleChain> chain,
    const grid::StaticClearanceGrid& sg,
    const Path& path,
    float radius) {
    if (path.cells.empty()) {
        return chain;
    }
    if (!chain) {
        chain = grid::ObstacleChain::make_root();
    }

    std::vector<geom::SphereObstacle> spheres;
    std::vector<geom::CapsuleObstacle> capsules;

    if (path.cells.size() == 1) {
        spheres.push_back({sg.cell_center(path.cells[0]), radius});
        return chain->extend(spheres, capsules);
    }

    // Merge maximal constant-direction runs, then:
    //   Chebyshev-adjacent endpoints → sphere (grid neighborhood; any-angle safe)
    //   otherwise → one capsule (covers long / any-angle hops)
    std::size_t i = 0;
    while (i + 1 < path.cells.size()) {
        const grid::CellCoord dir = path.cells[i + 1] - path.cells[i];
        std::size_t j = i + 1;
        while (j + 1 < path.cells.size() &&
               (path.cells[j + 1] - path.cells[j]) == dir) {
            ++j;
        }
        const geom::Vec3 a = sg.cell_center(path.cells[i]);
        const geom::Vec3 b = sg.cell_center(path.cells[j]);
        const grid::CellCoord d = path.cells[j] - path.cells[i];
        const int cheb = std::max({std::abs(d.x), std::abs(d.y), std::abs(d.z)});
        if (cheb == 1) {
            spheres.push_back({b, radius});
        } else {
            capsules.push_back({a, b, radius});
        }
        i = j;
    }

    return chain->extend(spheres, capsules);
}

/// Cross-group sequential routing: same-group → ConnectedSets; other-group → chain obstacles.
class PrioritySequencer : public MultiQueryPlanner {
public:
    PrioritySequencer() = default;
    ~PrioritySequencer() override = default;

    /// Schedule from `plans.ordered_tasks()`.
    /// `grid` is read-only: only its static field + entry chain are used as base.
    /// `stop_on_failure`: if true, pathfinding failure sets session Failure and stops;
    /// otherwise continue remaining Tasks (PP default; inspect via `any_failure()`).
    void reset(
        const topology::GlobalTopologyPlan& plans,
        const grid::CompositeGrid& grid,
        AStar<SearchNode>& low,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions,
        bool stop_on_failure = false);

    /// Like plans-order `reset`, with successful-prefix reuse from `past`.
    void reset(
        const topology::GlobalTopologyPlan& plans,
        const grid::CompositeGrid& grid,
        AStar<SearchNode>& low,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions,
        const std::vector<topology::Task>& past,
        bool stop_on_failure = false);

    /// Explicit task_id order (e.g. PP priority list).
    void reset(
        const topology::GlobalTopologyPlan& plans,
        const std::vector<std::string>& order,
        const grid::CompositeGrid& grid,
        AStar<SearchNode>& low,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions,
        bool stop_on_failure = false);

    /// Like order `reset`, then reuse a common successful prefix from `past`
    /// (same task_id sequence + `succeeded()`): replay merge/commit and set `cursor_`
    /// so `search` skips those Tasks.
    void reset(
        const topology::GlobalTopologyPlan& plans,
        const std::vector<std::string>& order,
        const grid::CompositeGrid& grid,
        AStar<SearchNode>& low,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions,
        const std::vector<topology::Task>& past,
        bool stop_on_failure = false);

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
    bool stop_on_failure_ = false;

    std::unordered_map<topology::TerminalId, std::size_t> term_comp_;
    std::vector<std::vector<grid::CellCoord>> comps_;
    std::vector<Committed> committed_;
};

}  // namespace pipe_design::planner
