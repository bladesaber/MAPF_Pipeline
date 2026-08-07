#pragma once

#include "topology/PureMST.h"

#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pipe_design::topology {

/// Hybrid MST: bin terminals by radius (thick → thin), MST within each bin,
/// bridge each new bin to the **previous** bin (nearest terminal pair) **before**
/// within-bin edges so GroupSession ConnectedSets see the prior layer first.
class RadiusSpineMSTBuilder : public PureMST {
public:
    using PureMST::PureMST;

    void build(const problem::Problem& problem) override {
        problem_ = &problem;
        plan_.groups.clear();
        plan_.groups.reserve(problem.groups().size());

        for (const problem::PipeGroup& g : problem.groups()) {
            GroupTopologyPlan gp(g.id);
            gp.terminals = g.terminals;
            const std::size_t n = gp.terminals.size();
            if (n < 2) {
                throw std::logic_error(
                    "RadiusSpineMSTBuilder: group has less than 2 terminals");
            }

            // radius → terminal indices (descending radius = spine first)
            std::map<float, std::vector<std::size_t>, std::greater<float>> bins_map;
            for (std::size_t i = 0; i < n; ++i) {
                bins_map[gp.terminals[i].radius].push_back(i);
            }
            std::vector<std::vector<std::size_t>> bins;
            bins.reserve(bins_map.size());
            for (auto& kv : bins_map) {
                bins.push_back(std::move(kv.second));
            }
            merge_singleton_bins(bins);

            gp.edges.reserve(n - 1);
            gp.grow_order.reserve(n - 1);

            const int num_levels = static_cast<int>(bins.size());
            int level_id = num_levels - 1;  // larger = thicker / earlier
            std::vector<std::size_t> prev;  // previous bin (bridge target)

            for (std::size_t bi = 0; bi < bins.size(); ++bi) {
                const std::vector<std::size_t>& idxs = bins[bi];
                const bool is_spine = prev.empty();
                const EdgeKind within_kind =
                    is_spine ? EdgeKind::Spine : EdgeKind::Branch;

                // --- Bridge first (so GroupSession sees last bin before within-bin) ---
                if (!prev.empty()) {
                    float best = std::numeric_limits<float>::infinity();
                    std::size_t best_u = 0;
                    std::size_t best_v = 0;
                    for (std::size_t ui : idxs) {
                        for (std::size_t vi : prev) {
                            const float w =
                                bridge_cost(gp.terminals[ui], gp.terminals[vi]);
                            if (w < best) {
                                best = w;
                                best_u = ui;
                                best_v = vi;
                            }
                        }
                    }
                    const Terminal& tu = gp.terminals[best_u];
                    const Terminal& tv = gp.terminals[best_v];
                    TopologyEdge edge(
                        tu.id + "-" + tv.id,
                        tu.id,
                        tv.id,
                        merge_radius(tu.radius, tv.radius),
                        level_id,
                        EdgeKind::Bridge);
                    gp.grow_order.push_back(edge.id);
                    gp.edges.push_back(std::move(edge));
                }

                // --- MST within this bin ---
                if (idxs.size() >= 2) {
                    std::vector<Terminal> sub;
                    sub.reserve(idxs.size());
                    for (std::size_t ti : idxs) {
                        sub.push_back(gp.terminals[ti]);
                    }
                    const std::vector<float> cost = cost_map(sub);
                    const std::vector<MstEdge> tree =
                        minimum_spanning_tree(sub.size(), cost);
                    for (const MstEdge& e : tree) {
                        const Terminal& tu = gp.terminals[idxs[e.u]];
                        const Terminal& tv = gp.terminals[idxs[e.v]];
                        TopologyEdge edge(
                            tu.id + "-" + tv.id,
                            tu.id,
                            tv.id,
                            merge_radius(tu.radius, tv.radius),
                            level_id,
                            within_kind);
                        gp.grow_order.push_back(edge.id);
                        gp.edges.push_back(std::move(edge));
                    }
                }

                prev = idxs;
                --level_id;
            }

            plan_.groups.push_back(std::move(gp));
        }
    }

private:
    /// Forward-merge size-1 bins into the next; trailing singleton merges backward.
    static void merge_singleton_bins(std::vector<std::vector<std::size_t>>& bins) {
        bool changed = true;
        while (changed && !bins.empty()) {
            changed = false;
            for (std::size_t i = 0; i < bins.size();) {
                if (bins[i].size() != 1) {
                    ++i;
                    continue;
                }
                changed = true;
                if (i + 1 < bins.size()) {
                    // append singleton to the front of the next bin
                    auto& next = bins[i + 1];
                    next.insert(next.begin(), bins[i].begin(), bins[i].end());
                    bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(i));
                    // stay at i (now the former next)
                } else if (i > 0) {
                    // last bin singleton → previous
                    auto& prev = bins[i - 1];
                    prev.insert(prev.end(), bins[i].begin(), bins[i].end());
                    bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(i));
                } else {
                    // only one bin left with a single terminal — cannot merge
                    break;
                }
            }
        }
    }

    /// Pairwise bridge weight (Euclidean / Volume / Dijkstra).
    float bridge_cost(const Terminal& a, const Terminal& b) const {
        switch (cost_type_) {
        case CostType::Dijkstra: {
            if (problem_ == nullptr) {
                throw std::logic_error(
                    "RadiusSpineMSTBuilder: Dijkstra bridge requires build(problem)");
            }
            const grid::StaticClearanceGrid& grid = problem_->static_grid();
            heuristic::Dijkstra dijk(grid);
            dijk.prepare(grid.world_to_cell(a.pos));
            const double d = dijk.distance(grid.world_to_cell(b.pos));
            if (!(d < heuristic::Dijkstra::kUnreachable)) {
                throw std::logic_error(
                    "RadiusSpineMSTBuilder: unreachable bridge (" + a.id + " -> " +
                    b.id + ")");
            }
            return static_cast<float>(d);
        }
        case CostType::Volume: {
            const float dist = (a.pos - b.pos).norm();
            const float r = merge_radius(a.radius, b.radius);
            return dist * r * r;
        }
        case CostType::Euclidean:
        default:
            return (a.pos - b.pos).norm();
        }
    }
};

}  // namespace pipe_design::topology
