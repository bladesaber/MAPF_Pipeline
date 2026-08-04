#pragma once

#include "planner/low/AStar.h"
#include "planner/motion/AdaptiveShell.h"

#include <unordered_set>
#include <vector>

namespace pipe_design::planner {

/// A* with clearance-adaptive Chebyshev shell steps (`AdaptiveShell`).
///
/// Before each expand, `step_size_for(clearance, pipe_radius)` sets the shell
/// radius. After expand, cells with Chebyshev distance `< s` are marked covered
/// so later jumps cannot land back inside that ball (anti jump-back).
/// Goals inside the ball are still linked as direct successors.
class AdaptiveStepAStar : public AStar<SearchNode> {
public:
    /// Map clearance / pipe radius → shell step in `[1, shell.max_step]`.
    int step_size_for(float clearance, float pipe_radius) const;

    void reset(
        const grid::CompositeGrid& grid,
        const std::vector<grid::CellCoord>& starts,
        const std::vector<grid::CellCoord>& goals,
        float pipe_radius,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        motion::AdaptiveShell& shell);

    void reset(
        const grid::CompositeGrid& grid,
        grid::CellCoord start,
        grid::CellCoord goal,
        float pipe_radius,
        const heuristic::Heuristic& heuristic,
        OpenSet& open,
        motion::AdaptiveShell& shell);

protected:
    void expand(SearchNode* current) override;

private:
    motion::AdaptiveShell* shell_ = nullptr;
    std::unordered_set<grid::CellCoord> covered_;
};

}  // namespace pipe_design::planner
