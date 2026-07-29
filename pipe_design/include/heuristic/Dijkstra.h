#pragma once

#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"
#include "heuristic/Heuristic.h"

#include <initializer_list>
#include <limits>
#include <vector>

namespace pipe_design::heuristic {

/// True-distance heuristic: multi-source Dijkstra on the static grid.
///
/// Admissible w.r.t. static obstacles only (ignores ObstacleChain). Distance
/// is to the **nearest** prepared source (single goal or attach set).
///
/// Occupancy while baking is **relaxed**: does not use the query pipe radius
/// (start/end radii may differ). Default: traverse cells with clearance > 0
/// (outside stamped solids). Optional bake_radius > 0 tightens with is_free.
///
/// Call prepare(...) before A*; call clear() or re-prepare when sources /
/// static bake change. Connectivity: 6-connect; edge cost = step_cost
/// (default: grid resolution).
class Dijkstra final : public Heuristic {
public:
    static constexpr double kUnreachable = std::numeric_limits<double>::infinity();

    /// @param grid         Static clearance field (non-owning; must outlive this).
    /// @param step_cost    Uniform edge cost; <= 0 means use grid.resolution().
    /// @param bake_radius  If > 0, use is_free(cell, bake_radius); if <= 0, clearance > 0.
    explicit Dijkstra(
        const grid::StaticClearanceGrid& grid,
        double step_cost = 0.0,
        float bake_radius = 0.f);

    const grid::StaticClearanceGrid& grid() const { return *grid_; }
    double step_cost() const { return step_cost_; }
    float bake_radius() const { return bake_radius_; }
    bool prepared() const { return prepared_; }
    const std::vector<grid::CellCoord>& goals() const { return goals_; }

    /// Drop prepared field (keeps grid binding).
    void clear();

    /// Multi-source Dijkstra from all valid sources in `goals`.
    void prepare(const std::vector<grid::CellCoord>& goals);
    void prepare(std::initializer_list<grid::CellCoord> goals);
    /// Single-source convenience.
    void prepare(grid::CellCoord goal);

    /// Distance to nearest source. `to` is ignored when prepared (multi-source).
    double estimate(const grid::CellCoord& from, const grid::CellCoord& to) const override;

    /// Distance from cell to nearest prepared source.
    double distance(const grid::CellCoord& cell) const;

private:
    bool traversable(const grid::CellCoord& cell) const;
    void run_dijkstra();

    const grid::StaticClearanceGrid* grid_ = nullptr;
    double step_cost_ = 1.0;
    float bake_radius_ = 0.f;

    bool prepared_ = false;
    std::vector<grid::CellCoord> goals_;
    std::vector<double> dist_;

    static constexpr int kNumNeighbors = 6;
    static constexpr int kOffsets[kNumNeighbors][3] = {
        {1, 0, 0},
        {-1, 0, 0},
        {0, 1, 0},
        {0, -1, 0},
        {0, 0, 1},
        {0, 0, -1},
    };
};

}  // namespace pipe_design::heuristic
