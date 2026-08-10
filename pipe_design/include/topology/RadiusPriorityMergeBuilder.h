#pragma once

#include "topology/PureMST.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pipe_design::topology {

/// Greedy radius-priority merge (Python `HyperRadiusGrowthTree`).
///
/// Repeatedly connect the nearest terminal pair between the current
/// max-radius component(s) and the next priority set, then merge those
/// two components (`RadiusMerge`). Edge order = `grow_order`.
class RadiusPriorityMergeBuilder : public PureMST {
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
                    "RadiusPriorityMergeBuilder: group has less than 2 terminals");
            }

            // One component per terminal (Python leaf).
            std::vector<std::vector<std::size_t>> comps(n);
            std::vector<float> comp_r(n);
            for (std::size_t i = 0; i < n; ++i) {
                comps[i] = {i};
                comp_r[i] = gp.terminals[i].radius;
            }

            gp.edges.reserve(n - 1);
            gp.grow_order.reserve(n - 1);

            while (comps.size() > 1) {
                // first priority = max radius; if unique, second = next max
                float max_r = -std::numeric_limits<float>::infinity();
                for (float r : comp_r) {
                    max_r = std::max(max_r, r);
                }
                std::vector<std::size_t> first;
                for (std::size_t i = 0; i < comps.size(); ++i) {
                    if (comp_r[i] == max_r) {
                        first.push_back(i);
                    }
                }
                std::vector<std::size_t> second;
                if (first.size() > 1) {
                    second = first;
                } else {
                    float max2 = -std::numeric_limits<float>::infinity();
                    for (std::size_t i = 0; i < comps.size(); ++i) {
                        if (comp_r[i] != max_r) {
                            max2 = std::max(max2, comp_r[i]);
                        }
                    }
                    for (std::size_t i = 0; i < comps.size(); ++i) {
                        if (comp_r[i] == max2) {
                            second.push_back(i);
                        }
                    }
                }

                float best = std::numeric_limits<float>::infinity();
                std::size_t best_c0 = 0, best_c1 = 0;
                std::size_t best_t0 = 0, best_t1 = 0;
                for (std::size_t c0 : first) {
                    for (std::size_t c1 : second) {
                        // When first==second (multiple max-radius comps), skip self-pairs.
                        if (c0 == c1) {
                            continue;
                        }
                        for (std::size_t t0 : comps[c0]) {
                            for (std::size_t t1 : comps[c1]) {
                                if (gp.terminals[t0].id == gp.terminals[t1].id) {
                                    throw std::logic_error(
                                        "RadiusPriorityMergeBuilder: duplicate terminal id in "
                                        "different components (" +
                                        gp.terminals[t0].id + ")");
                                }
                                const float w =
                                    pair_cost(gp.terminals[t0], gp.terminals[t1]);
                                if (w < best) {
                                    best = w;
                                    best_c0 = c0;
                                    best_c1 = c1;
                                    best_t0 = t0;
                                    best_t1 = t1;
                                }
                            }
                        }
                    }
                }

                const float new_r = merge_radius(comp_r[best_c0], comp_r[best_c1]);
                const Terminal& tu = gp.terminals[best_t0];
                const Terminal& tv = gp.terminals[best_t1];
                TopologyEdge edge(tu.id + "-" + tv.id, tu.id, tv.id, new_r);
                gp.grow_order.push_back(edge.id);
                gp.edges.push_back(std::move(edge));

                // Merge c1 into c0, then drop c1.
                if (best_c0 > best_c1) {
                    std::swap(best_c0, best_c1);
                }
                comps[best_c0].insert(comps[best_c0].end(), comps[best_c1].begin(), comps[best_c1].end());
                comp_r[best_c0] = new_r;
                comps.erase(comps.begin() + static_cast<std::ptrdiff_t>(best_c1));
                comp_r.erase(comp_r.begin() + static_cast<std::ptrdiff_t>(best_c1));
            }

            plan_.groups.push_back(std::move(gp));
        }
    }
};

}  // namespace pipe_design::topology
