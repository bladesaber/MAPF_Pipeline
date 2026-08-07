#pragma once

#include "heuristic/Dijkstra.h"
#include "topology/Mst.h"
#include "topology/TopologyBuilder.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pipe_design::topology {

/// How to set `TopologyEdge::radius` from the two endpoint terminal radii.
/// Aligns with Python `merge_leafs(..., merge_radius_fun=…)`.
enum class RadiusMerge : std::uint8_t {
    Min = 0,     // np.min (default)
    Average,     // 0.5 * (r_u + r_v)
};

/// Pairwise edge weight for the terminal cost map.
enum class CostType : std::uint8_t {
    Euclidean = 0,  // ||p_i - p_j||
    Dijkstra,       // true distance on Problem::static_grid() (6-connect)
    Volume,         // ||p_i - p_j|| * max(r_i, r_j)^2
};

/// Baseline `[T]` builder: one MST per `PipeGroup` (all terminals, no radius bins).
class PureMST : public TopologyBuilder {
public:
    explicit PureMST(RadiusMerge radius_merge = RadiusMerge::Min,
                     CostType cost_type = CostType::Euclidean)
        : radius_merge_(radius_merge)
        , cost_type_(cost_type) {}

    RadiusMerge radius_merge() const { return radius_merge_; }
    void set_radius_merge(RadiusMerge merge) { radius_merge_ = merge; }

    CostType cost_type() const { return cost_type_; }
    void set_cost_type(CostType type) { cost_type_ = type; }

    void build(const problem::Problem& problem) override {
        problem_ = &problem;
        plan_.groups.clear();
        plan_.groups.reserve(problem.groups().size());

        for (const problem::PipeGroup& g : problem.groups()) {
            GroupTopologyPlan gp(g.id);
            gp.terminals = g.terminals;

            const std::size_t n = gp.terminals.size();
            if (n < 2) {
                throw std::logic_error("PureMST: group has less than 2 terminals");
            }

            const std::vector<float> cost = cost_map(gp.terminals);
            const std::vector<MstEdge> tree = minimum_spanning_tree(n, cost);
            gp.edges.reserve(tree.size());
            gp.grow_order.reserve(tree.size());
            for (const MstEdge& e : tree) {
                const Terminal& tu = gp.terminals[e.u];
                const Terminal& tv = gp.terminals[e.v];
                TopologyEdge edge(
                    tu.id + "-" + tv.id,
                    tu.id,
                    tv.id,
                    merge_radius(tu.radius, tv.radius));
                gp.grow_order.push_back(edge.id);
                gp.edges.push_back(std::move(edge));
            }
            plan_.groups.push_back(std::move(gp));
        }
    }

    /// Row-major `n × n` pairwise costs (selected by `cost_type_`).
    virtual std::vector<float> cost_map(const std::vector<Terminal>& terminals) const {
        switch (cost_type_) {
        case CostType::Dijkstra:
            return cost_map_dijkstra(terminals);
        case CostType::Volume:
            return cost_map_volume(terminals);
        case CostType::Euclidean:
        default:
            return cost_map_euclidean(terminals);
        }
    }

protected:
    float merge_radius(float a, float b) const {
        if (radius_merge_ == RadiusMerge::Average) {
            return 0.5f * (a + b);
        }
        return std::min(a, b);
    }

    std::vector<float> cost_map_euclidean(const std::vector<Terminal>& terminals) const {
        const std::size_t n = terminals.size();
        std::vector<float> cost(n * n, 0.f);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const float w = (terminals[i].pos - terminals[j].pos).norm();
                cost[i * n + j] = w;
                cost[j * n + i] = w;
            }
        }
        return cost;
    }

    /// Euclidean * max(r_i, r_j)^2 (Python MinimumVolumeTree-style).
    std::vector<float> cost_map_volume(const std::vector<Terminal>& terminals) const {
        const std::size_t n = terminals.size();
        std::vector<float> cost(n * n, 0.f);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const float dist = (terminals[i].pos - terminals[j].pos).norm();
                const float r = merge_radius(terminals[i].radius, terminals[j].radius);
                const float w = dist * r * r;
                cost[i * n + j] = w;
                cost[j * n + i] = w;
            }
        }
        return cost;
    }

    /// True distance on `Problem::static_grid()` via `heuristic::Dijkstra` (6-connect).
    std::vector<float> cost_map_dijkstra(const std::vector<Terminal>& terminals) const {
        if (problem_ == nullptr) {
            throw std::logic_error("PureMST: Dijkstra cost requires build(problem) first");
        }
        const grid::StaticClearanceGrid& grid = problem_->static_grid();
        heuristic::Dijkstra dijk(grid);

        const std::size_t n = terminals.size();
        std::vector<float> cost(n * n, 0.f);

        for (std::size_t i = 0; i < n; ++i) {
            const grid::CellCoord src = grid.world_to_cell(terminals[i].pos);
            dijk.prepare(src);
            for (std::size_t j = i + 1; j < n; ++j) {
                const grid::CellCoord dst = grid.world_to_cell(terminals[j].pos);
                const double d = dijk.distance(dst);
                if (!(d < heuristic::Dijkstra::kUnreachable)) {
                    throw std::logic_error(
                        "PureMST: unreachable terminals on static grid (" +
                        terminals[i].id + " -> " + terminals[j].id + ")");
                }
                const float w = static_cast<float>(d);
                cost[i * n + j] = w;
                cost[j * n + i] = w;
            }
        }
        return cost;
    }

    RadiusMerge radius_merge_ = RadiusMerge::Min;
    CostType cost_type_ = CostType::Euclidean;
};

}  // namespace pipe_design::topology
