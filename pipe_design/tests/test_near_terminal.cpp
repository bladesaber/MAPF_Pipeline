#include "conflict/NearTerminalAnalyzer.h"
#include "grid/StaticClearanceGrid.h"
#include "topology/Types.h"

#include <cstdio>
#include <unordered_map>

int main() {
    using pipe_design::conflict::NearTerminalAnalyzer;
    using pipe_design::conflict::SampledPoint;
    using pipe_design::geom::Vec3;
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
    g0.terminals = {Terminal{"t0", {0.5f, 0.5f, 0.5f}, 1.f},
                    Terminal{"t1", {10.5f, 0.5f, 0.5f}, 1.f}};
    g0.edges.push_back(TopologyEdge{"e0", "t0", "t1", 1.f});

    GroupTopologyPlan g1{"g1"};
    g1.terminals = {Terminal{"u0", {5.5f, -1.5f, 0.5f}, 1.f},
                    Terminal{"u1", {5.5f, 4.5f, 0.5f}, 1.f}};
    g1.edges.push_back(TopologyEdge{"e1", "u0", "u1", 1.f});

    GlobalTopologyPlan plans({g0, g1});

    Task a{"g0:e0", "g0", "e0"};
    Path pa;
    for (int x = 0; x <= 10; ++x) {
        pa.cells.push_back(CellCoord{x, 0, 0});
    }
    a.set_result(SearchStatus::Success, std::move(pa));

    Task b{"g1:e1", "g1", "e1"};
    Path pb;
    for (int y = -2; y <= 4; ++y) {
        pb.cells.push_back(CellCoord{5, y, 0});
    }
    b.set_result(SearchStatus::Success, std::move(pb));

    std::unordered_map<std::string, std::vector<Vec3>> terminals;
    terminals["g0"] = {Vec3{0.5f, 0.5f, 0.5f}, Vec3{10.5f, 0.5f, 0.5f}};
    terminals["g1"] = {Vec3{5.5f, -1.5f, 0.5f}, Vec3{5.5f, 4.5f, 0.5f}};

    NearTerminalAnalyzer az(std::move(terminals));
    az.set_grid(&grid);
    az.set_plans(&plans);
    az.analyze({&a, &b});

    if (az.conflicts().empty()) {
        std::fprintf(stderr, "no conflicts\n");
        return 1;
    }
    if (az.candidates().empty()) {
        std::fprintf(stderr, "no rank-0 candidates\n");
        return 2;
    }
    if (az.select_conflict() == nullptr) {
        std::fprintf(stderr, "select_conflict null\n");
        return 3;
    }

    std::puts("ok");
    return 0;
}
