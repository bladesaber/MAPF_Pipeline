#pragma once

#include "planner/low/AStar.h"
#include "planner/low/SearchNode.h"
#include "planner/open/OpenSet.h"
#include "planner/open/PairingHeap.h"

#include <unordered_set>
#include <vector>

namespace pipe_design::planner {

/// FOCAL ordering: min h; on tie prefer larger g.
struct MinHgCompare {
    bool operator()(const SearchNode* a, const SearchNode* b) const {
        if (a->h != b->h) {
            return a->h > b->h;
        }
        return a->g < b->g;
    }
};

/// Focal A* (A*_ε): OPEN by `f = g + h`; FOCAL by `h` among
/// { n ∈ OPEN | f(n) ≤ w · f_min }.
///
/// Caller owns OPEN (`OpenSet&`, typically `PairingHeap`). FOCAL is an owned
/// `PairingHeapBase` with `MinHgCompare`. Expand removes from both.
/// `open_nodes_` mirrors OPEN membership for O(|OPEN|) bound sync.
/// `w == 1` falls back to plain A* (FOCAL unused).
class FocalAStar : public AStar<SearchNode> {
public:
    void set_focal_weight(double weight);
    double focal_weight() const { return focal_weight_; }

    void reset(
        const grid::CompositeGrid& grid,
        const std::vector<grid::CellCoord>& starts,
        const std::vector<grid::CellCoord>& goals,
        float pipe_radius,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions);

    void reset(
        const grid::CompositeGrid& grid,
        grid::CellCoord start,
        grid::CellCoord goal,
        float pipe_radius,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        const motion::ActionModel& actions);

    SearchStatus step() override;

protected:
    double combine_key(double g, double h) const override;
    void expand(SearchNode* current) override;

private:
    void sync_focal_bound();
    void push_focal(SearchNode* node);

    double focal_weight_ = 1.0;
    double focal_bound_ = 0.0;
    PairingHeapBase<SearchNode, MinHgCompare> focal_;
    std::unordered_set<SearchNode*> open_nodes_;
};

extern template class PairingHeapBase<SearchNode, MinHgCompare>;

}  // namespace pipe_design::planner
