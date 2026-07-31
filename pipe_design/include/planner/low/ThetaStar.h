#pragma once

#include "grid/CompositeGrid.h"
#include "grid/Types.h"
#include "planner/low/AStar.h"

namespace pipe_design::planner {

/// Basic Theta*: any-angle paths via parent line-of-sight on expand.
///
/// Neighbors still come from `ActionModel` (6/8/26); when parent(s) can see
/// s', parent is set to parent(s) with Euclidean edge cost. Path cells are
/// waypoints (may skip grid steps). Prefer a Euclidean heuristic.
///
/// `step_cost` scales Euclidean parent edges (default 1; usually = resolution).
class ThetaStar : public AStar<SearchNode> {
public:
    void set_step_cost(double step_cost);
    double step_cost() const { return step_cost_; }

    /// Line-of-sight between cell centers (samples segment with `is_free`).
    static bool line_of_sight(
        const grid::CompositeGrid& grid,
        grid::CellCoord a,
        grid::CellCoord b,
        float pipe_radius);

protected:
    void expand(SearchNode* current) override;

    /// Path-2 update: parent(current) → neighbor when LOS holds; else path-1.
    void update_vertex(SearchNode* current, SearchNode* neighbor, double grid_edge_cost);

    /// Euclidean edge cost in the same units as ActionModel step_cost.
    static double euclidean_cell_cost(
        const grid::CellCoord& a,
        const grid::CellCoord& b,
        double step_cost);

    double step_cost_ = 1.0;
};

}  // namespace pipe_design::planner
