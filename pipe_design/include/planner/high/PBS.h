#pragma once

#include "conflict/ConflictAnalyzer.h"
#include "grid/CompositeGrid.h"
#include "heuristic/Heuristic.h"
#include "planner/high/HighLevelPlanner.h"
#include "planner/low/AStar.h"
#include "planner/low/SearchNode.h"
#include "planner/motion/ActionModel.h"
#include "planner/multi/PartialOrderSequencer.h"
#include "planner/open/OpenSet.h"
#include "topology/Types.h"

#include <cstddef>
#include <cstdint>
#include <queue>
#include <vector>

namespace pipe_design::planner {

/// Open-set policy for the PBS priority tree.
enum class PBSSearchMode : std::uint8_t {
    /// Prefer lower `cost`, then shallower `depth` (CBS-like).
    BestFirst = 0,
    /// Prefer deeper `depth`, then lower `cost` (classic PBS-style).
    DepthFirst = 1,
};

/// Priority-tree high-level search (Ma-style PBS).
///
/// Topology frozen. Each PT node holds a `PriorityConstraint` set + leaf Tasks.
/// Leaf routing via `PartialOrderSequencer` (warm-start from parent `snapshot`).
/// `[V]` via injected `ConflictAnalyzer`; branch adds `task_a ≺ task_b` / reverse.
class PBS : public HighLevelPlanner {
public:
    PBS() : open_(PTNodeGreater{PBSSearchMode::DepthFirst}) {}
    ~PBS() override = default;

    void set_search_mode(PBSSearchMode mode) { search_mode_ = mode; }
    PBSSearchMode search_mode() const { return search_mode_; }

    void reset(
        const topology::GlobalTopologyPlan& plans,
        const grid::CompositeGrid& grid,
        AStar<SearchNode>& low,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions,
        PartialOrderSequencer& sequencer,
        conflict::ConflictAnalyzer& analyzer);

    SearchStatus step() override;
    std::vector<topology::Task> solution() const override { return solution_; }

protected:
    struct PTNode {
        std::vector<PriorityConstraint> constraints;
        PartialOrderPast past;
        std::vector<topology::Task> tasks;
        double cost = 0.0;
        std::size_t depth = 0;
    };

    /// Route `node` with `PartialOrderSequencer` (+ `node.past` reuse). False on failure.
    bool route(PTNode& node);

private:
    struct PTNodeGreater {
        PBSSearchMode mode = PBSSearchMode::DepthFirst;

        bool operator()(const PTNode& a, const PTNode& b) const {
            if (mode == PBSSearchMode::DepthFirst) {
                if (a.depth != b.depth) {
                    return a.depth < b.depth;  // deeper first
                }
                return a.cost > b.cost;
            }
            if (a.cost != b.cost) {
                return a.cost > b.cost;  // cheaper first
            }
            return a.depth > b.depth;  // shallower first
        }
    };

    static bool has_constraint(
        const std::vector<PriorityConstraint>& cons,
        const std::string& higher_id,
        const std::string& lower_id);

    const topology::GlobalTopologyPlan* plans_ = nullptr;
    const grid::CompositeGrid* grid_ = nullptr;
    AStar<SearchNode>* low_ = nullptr;
    const heuristic::Heuristic* heuristic_ = nullptr;
    OpenSet* open_set_ = nullptr;
    const motion::ActionModel* actions_ = nullptr;
    PartialOrderSequencer* sequencer_ = nullptr;
    conflict::ConflictAnalyzer* analyzer_ = nullptr;

    PBSSearchMode search_mode_ = PBSSearchMode::DepthFirst;
    std::priority_queue<PTNode, std::vector<PTNode>, PTNodeGreater> open_;
    std::vector<topology::Task> solution_;
};

}  // namespace pipe_design::planner
