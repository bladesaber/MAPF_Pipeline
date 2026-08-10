#pragma once

#include "topology/PureMST.h"

#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pipe_design::topology {

/// Hybrid MST: bin terminals by radius (thick → thin), MST within each bin
/// (skipped if bin size < 2), bridge each new bin to **all prior bins**
/// (nearest terminal pair) **before** within-bin edges so GroupSession
/// ConnectedSets see the prior layers first. A size-1 bin is bridge-only.
class RadiusSpineMSTBuilder : public PureMST {
public:
    using PureMST::PureMST;

    void build(const problem::Problem& problem) override {
        problem_ = &problem;
        clear_dijkstra_cache();
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

            gp.edges.reserve(n - 1);
            gp.grow_order.reserve(n - 1);

            const int num_levels = static_cast<int>(bins.size());
            int level_id = num_levels - 1;  // larger = thicker / earlier
            std::vector<std::size_t> prev;  // union of all prior bins
            prev.reserve(n);

            for (std::size_t bi = 0; bi < bins.size(); ++bi) {
                const std::vector<std::size_t>& idxs = bins[bi];
                const bool is_spine = prev.empty();
                const EdgeKind within_kind =
                    is_spine ? EdgeKind::Spine : EdgeKind::Branch;

                // --- Bridge first (so GroupSession sees prior layers before within-bin) ---
                if (!prev.empty()) {
                    float best = std::numeric_limits<float>::infinity();
                    std::size_t best_u = 0;
                    std::size_t best_v = 0;
                    for (std::size_t ui : idxs) {
                        for (std::size_t vi : prev) {
                            const float w =
                                pair_cost(gp.terminals[ui], gp.terminals[vi]);
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

                // --- MST within this bin (size-1 → bridge only, no within edges) ---
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

                prev.insert(prev.end(), idxs.begin(), idxs.end());
                --level_id;
            }

            plan_.groups.push_back(std::move(gp));
        }
    }
};

}  // namespace pipe_design::topology
