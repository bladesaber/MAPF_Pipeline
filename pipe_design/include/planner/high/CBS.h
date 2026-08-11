#pragma once

#include "conflict/ConflictAnalyzer.h"
#include "grid/CompositeGrid.h"
#include "heuristic/Heuristic.h"
#include "planner/high/HighLevelPlanner.h"
#include "planner/low/AStar.h"
#include "planner/low/SearchNode.h"
#include "planner/motion/ActionModel.h"
#include "planner/multi/GroupSession.h"
#include "planner/open/OpenSet.h"
#include "topology/Types.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace pipe_design::planner {

/// Constraint-tree high-level search (geometric CBS).
///
/// Topology frozen. Each CT node holds per-group `ObstacleChain` constraints + Tasks.
/// `[V]` runs at **generation** (`evaluate`): caches `h` + one selected `conflict`.
/// Expansion uses the cached conflict; OPEN orders by `f = cost + h`.
///
/// Other groups are **not** baked as low-level obstacles (constraint-centric CBS).
/// Override `constrain` for disjoint split / stronger branching later.
class CBS : public HighLevelPlanner {
public:
    CBS() = default;
    ~CBS() override = default;

    /// Bind plans + leaf driver + analyzer. Builds root CT node (all groups routed).
    void reset(
        const topology::GlobalTopologyPlan& plans,
        const grid::CompositeGrid& grid,
        AStar<SearchNode>& low,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions,
        GroupSession& session,
        conflict::ConflictAnalyzer& analyzer);

    SearchStatus step() override;
    std::vector<topology::Task> solution() const override { return solution_; }

protected:
    struct CTNode {
        /// Per-group constraint chains (absent → entry `grid.chain()` / empty root).
        std::unordered_map<topology::GroupId, std::shared_ptr<const grid::ObstacleChain>> chains;
        /// Per-Task COW: children share unchanged Tasks; `route_group` replaces one group.
        std::vector<std::shared_ptr<const topology::Task>> tasks;
        double cost = 0.0;  // g = sum of path costs
        double h = 0.0;     // analyzer heuristic (filled at generation)
        /// Selected conflict at generation; nullopt = conflict-free.
        std::optional<conflict::Conflict> conflict;
        std::size_t depth = 0;  // root = 0; child = parent + 1
    };

    /// Add a constraint so `group` must avoid `c` on its side.
    /// Default: extend with the **other** witness sphere. Override for variants.
    virtual std::shared_ptr<const grid::ObstacleChain> constrain(
        std::shared_ptr<const grid::ObstacleChain> chain,
        const conflict::Conflict& c,
        const topology::GroupId& group) const;

    std::shared_ptr<const grid::ObstacleChain> chain_for(
        const CTNode& node,
        const topology::GroupId& group) const;

    /// Re-route one group into `node` (COW-replace `tasks`). False on failure.
    bool route_group(CTNode& node, const topology::GroupId& group);

    /// After paths are set: write `cost` / `h` / `conflict` via analyzer (generation-time).
    void evaluate(CTNode& node);

    static double path_cost(const CTNode& node);

private:
    struct CTNodeGreater {
        bool operator()(const CTNode& a, const CTNode& b) const {
            const double fa = a.cost + a.h;
            const double fb = b.cost + b.h;
            if (fa != fb) {
                return fa > fb;
            }
            return a.h > b.h;  // equal f → prefer smaller h
        }
    };

    const topology::GlobalTopologyPlan* plans_ = nullptr;
    const grid::CompositeGrid* grid_ = nullptr;
    AStar<SearchNode>* low_ = nullptr;
    const heuristic::Heuristic* heuristic_ = nullptr;
    OpenSet* open_set_ = nullptr;
    const motion::ActionModel* actions_ = nullptr;
    GroupSession* session_ = nullptr;
    conflict::ConflictAnalyzer* analyzer_ = nullptr;

    std::priority_queue<CTNode, std::vector<CTNode>, CTNodeGreater> open_;
    std::vector<topology::Task> solution_;
};

}  // namespace pipe_design::planner
