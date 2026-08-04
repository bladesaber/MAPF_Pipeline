#include "geom/Types.h"
#include "grid/CompositeGrid.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"
#include "heuristic/Geometric.h"
#include "planner/low/AStar.h"
#include "planner/low/AdaptiveStepAStar.h"
#include "planner/low/SearchNode.h"
#include "planner/motion/AdaptiveShell.h"
#include "planner/motion/SixConnected.h"
#include "planner/open/LazyBinaryHeap.h"

#include <cmath>
#include <iostream>

using pipe_design::grid::CellCoord;
using pipe_design::grid::CompositeGrid;
using pipe_design::grid::StaticClearanceGrid;
using pipe_design::heuristic::Euclidean;
using pipe_design::heuristic::Manhattan;
using pipe_design::planner::AStar;
using pipe_design::planner::AdaptiveStepAStar;
using pipe_design::planner::LazyBinaryHeap;
using pipe_design::planner::Path;
using pipe_design::planner::SearchNode;
using pipe_design::planner::SearchStatus;
using pipe_design::planner::motion::AdaptiveShell;
using pipe_design::planner::motion::SixConnected;

static int g_failures = 0;

#define EXPECT_TRUE(cond)                                                          \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " #cond "\n"; \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

#define EXPECT_NEAR(a, b, eps)                                                     \
    do {                                                                           \
        if (std::fabs((a) - (b)) > (eps)) {                                        \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " #a "≈" #b   \
                      << " got " << (a) << " vs " << (b) << "\n";                  \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

static void test_step_size_for() {
    AdaptiveShell shell(1.0, /*max_step=*/4);
    LazyBinaryHeap open;
    AdaptiveStepAStar planner;
    StaticClearanceGrid static_grid(10, 10, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Euclidean h(1.0);
    planner.reset(grid, CellCoord{0, 0, 0}, CellCoord{1, 0, 0}, 0.1f, h, open, shell);

    EXPECT_TRUE(planner.step_size_for(0.5f, 0.1f) == 1);   // floor(0.4)=0 → clamp 1
    EXPECT_TRUE(planner.step_size_for(2.5f, 0.1f) == 2);   // floor(2.4)=2
    EXPECT_TRUE(planner.step_size_for(10.f, 0.1f) == 4);   // clamp to max_step
}

static void test_straight_line() {
    StaticClearanceGrid static_grid(30, 10, 1, 1.f, 8.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Euclidean h(1.0);
    AdaptiveShell shell(1.0, /*max_step=*/4);
    LazyBinaryHeap open;

    AdaptiveStepAStar planner;
    const CellCoord start{0, 0, 0};
    const CellCoord goal{12, 0, 0};
    planner.reset(grid, start, goal, 0.1f, h, open, shell);
    EXPECT_TRUE(planner.search() == SearchStatus::Success);
    const Path p = planner.path();
    EXPECT_TRUE(!p.empty());
    EXPECT_TRUE(p.cells.front() == start);
    EXPECT_TRUE(p.cells.back() == goal);
    EXPECT_TRUE(p.cost + 1e-9 >= 12.0);
}

static void test_start_equals_goal() {
    StaticClearanceGrid static_grid(5, 5, 1, 1.f, 3.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Euclidean h(1.0);
    AdaptiveShell shell(1.0);
    LazyBinaryHeap open;

    AdaptiveStepAStar planner;
    const CellCoord s{2, 2, 0};
    planner.reset(grid, s, s, 0.1f, h, open, shell);
    EXPECT_TRUE(planner.status() == SearchStatus::Success);
    EXPECT_NEAR(planner.path().cost, 0.0, 1e-9);
}

static void test_path_nodes_fewer_than_astar() {
    // Empty map, diagonal goal → AdaptiveStepAStar ≈ any-angle (shorter cost + fewer nodes).
    StaticClearanceGrid static_grid(50, 40, 1, 1.f, 12.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    const CellCoord start{0, 0, 0};
    const CellCoord goal{24, 18, 0};
    const float r = 0.1f;
    const double eu_opt = std::sqrt(24.0 * 24.0 + 18.0 * 18.0);

    LazyBinaryHeap open_astar;
    Manhattan manh(1.0);
    SixConnected six(1.0);
    AStar<SearchNode> astar;
    astar.reset(grid, start, goal, r, manh, open_astar, six);
    EXPECT_TRUE(astar.search() == SearchStatus::Success);
    const Path p_astar = astar.path();

    LazyBinaryHeap open_adapt;
    Euclidean eu(1.0);
    AdaptiveShell shell(1.0, /*max_step=*/4);
    AdaptiveStepAStar adapt;
    adapt.reset(grid, start, goal, r, eu, open_adapt, shell);
    EXPECT_TRUE(adapt.search() == SearchStatus::Success);
    const Path p_adapt = adapt.path();

    std::cout << "A* SixConnected:     path_nodes=" << p_astar.length()
              << " cost=" << p_astar.cost << " expansions=" << astar.expansions() << '\n';
    std::cout << "AdaptiveStepAStar:   path_nodes=" << p_adapt.length()
              << " cost=" << p_adapt.cost << " expansions=" << adapt.expansions()
              << " (Euclidean opt≈" << eu_opt << ")\n";

    EXPECT_TRUE(p_astar.cells.front() == start && p_astar.cells.back() == goal);
    EXPECT_TRUE(p_adapt.cells.front() == start && p_adapt.cells.back() == goal);
    EXPECT_TRUE(p_adapt.length() < p_astar.length());
    EXPECT_TRUE(p_adapt.cost + 1e-6 < p_astar.cost);
    EXPECT_NEAR(p_adapt.cost, eu_opt, 1e-6);
}

int main() {
    test_step_size_for();
    test_straight_line();
    test_start_equals_goal();
    test_path_nodes_fewer_than_astar();
    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
