#include "conflict/ConflictAnalyzer.h"
#include "grid/StaticClearanceGrid.h"
#include "topology/Types.h"

#include <cstdio>
#include <cstdlib>

namespace {

struct DeficitAnalyzer : pipe_design::conflict::ConflictAnalyzer {
    void score(pipe_design::conflict::Conflict& c) const override { c.score = c.deficit; }
    void score(pipe_design::conflict::ConflictCluster& cl) const override {
        cl.score = cl.max_deficit();
    }
    const pipe_design::conflict::Conflict* select_conflict() const override {
        if (conflicts().empty()) {
            return nullptr;
        }
        const pipe_design::conflict::Conflict* best = &conflicts().front();
        for (const auto& c : conflicts()) {
            if (c.score > best->score) {
                best = &c;
            }
        }
        return best;
    }
};

}  // namespace

int main() {
    using pipe_design::grid::CellCoord;
    using pipe_design::grid::StaticClearanceGrid;
    using pipe_design::planner::Path;
    using pipe_design::planner::SearchStatus;
    using pipe_design::topology::GlobalTopologyPlan;
    using pipe_design::topology::GroupTopologyPlan;
    using pipe_design::topology::Task;
    using pipe_design::topology::Terminal;
    using pipe_design::topology::TopologyEdge;

    StaticClearanceGrid grid(32, 32, 8, 1.f, 5.f);
    grid.clear();

    GroupTopologyPlan g0{"g0"};
    g0.terminals.push_back(Terminal{"t0", {0.5f, 0.5f, 0.5f}, 1.f});
    g0.terminals.push_back(Terminal{"t1", {10.5f, 0.5f, 0.5f}, 1.f});
    g0.edges.push_back(TopologyEdge{"e0", "t0", "t1", 1.f});

    GroupTopologyPlan g1{"g1"};
    g1.terminals.push_back(Terminal{"u0", {5.5f, -2.5f, 0.5f}, 1.f});
    g1.terminals.push_back(Terminal{"u1", {5.5f, 4.5f, 0.5f}, 1.f});
    g1.edges.push_back(TopologyEdge{"e1", "u0", "u1", 1.f});

    GlobalTopologyPlan plans;
    plans.groups.push_back(g0);
    plans.groups.push_back(g1);

    Task a{"g0:e0", "g0", "e0"};
    Path pa;
    for (int x = 0; x <= 10; ++x) {
        pa.cells.push_back(CellCoord{x, 0, 0});
    }
    a.set_result(SearchStatus::Success, pa);

    Task b{"g1:e1", "g1", "e1"};
    Path pb;
    for (int y = -2; y <= 4; ++y) {
        pb.cells.push_back(CellCoord{5, y, 0});
    }
    b.set_result(SearchStatus::Success, pb);

    DeficitAnalyzer az;
    az.set_grid(&grid);
    az.set_plans(&plans);
    az.set_sample_step_factor(1.f);
    az.normalize({&a, &b});

    if (az.sampled_paths().size() != 2) {
        std::fprintf(stderr, "normalize groups=%zu\n", az.sampled_paths().size());
        return 1;
    }
    const auto& paths0 = az.sampled_paths().at("g0");
    const auto& paths1 = az.sampled_paths().at("g1");
    if (paths0.size() != 1 || paths1.size() != 1) {
        std::fprintf(stderr, "expected 1 path per group\n");
        return 2;
    }
    if (paths0[0].samples.size() < 2 || paths1[0].samples.size() < 2) {
        std::fprintf(stderr, "too few samples\n");
        return 3;
    }

    az.build_graphs();
    if (az.graphs().size() != 2) {
        std::fprintf(stderr, "graphs groups=%zu\n", az.graphs().size());
        return 5;
    }
    if (az.graphs().at("g0").node_count() < 2 || az.graphs().at("g1").node_count() < 2) {
        std::fprintf(stderr, "graph too small\n");
        return 6;
    }
    if (az.graphs().at("g0").edge_count() < 1 || az.graphs().at("g1").edge_count() < 1) {
        std::fprintf(stderr, "graph missing edges\n");
        return 7;
    }

    az.find_conflicts();
    if (az.conflicts().empty()) {
        std::fprintf(stderr, "expected conflicts at crossing\n");
        return 4;
    }

    az.find_clusters();
    if (az.clusters().empty()) {
        std::fprintf(stderr, "expected at least one cluster\n");
        return 9;
    }
    std::size_t clustered = 0;
    for (const auto& cl : az.clusters()) {
        clustered += cl.size();
        if (cl.groups.size() < 2) {
            std::fprintf(stderr, "cluster should touch >=2 groups\n");
            return 10;
        }
    }
    if (clustered != az.conflicts().size()) {
        std::fprintf(stderr, "clustered %zu vs conflicts %zu\n", clustered, az.conflicts().size());
        return 11;
    }

    // Same group only → no conflict
    az.clear();
    az.set_grid(&grid);
    az.set_plans(&plans);
    az.normalize({&a, &a});
    az.build_graphs();
    az.find_conflicts();
    az.find_clusters();
    if (!az.conflicts().empty() || !az.clusters().empty()) {
        std::fprintf(stderr, "same-group should not conflict\n");
        return 8;
    }

    std::puts("ok");
    return 0;
}
