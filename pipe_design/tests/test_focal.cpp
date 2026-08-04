#include "geom/Types.h"
#include "grid/CompositeGrid.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"
#include "heuristic/Geometric.h"
#include "planner/low/FocalAStar.h"
#include "planner/motion/SixConnected.h"
#include "planner/open/LazyBinaryHeap.h"

#include <cmath>
#include <iostream>

using pipe_design::grid::CellCoord;
using pipe_design::grid::CompositeGrid;
using pipe_design::grid::StaticClearanceGrid;
using pipe_design::heuristic::Manhattan;
using pipe_design::planner::FocalAStar;
using pipe_design::planner::LazyBinaryHeap;
using pipe_design::planner::Path;
using pipe_design::planner::SearchStatus;
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

static bool path_is_6_connected(const Path& path) {
    if (path.cells.empty()) {
        return false;
    }
    for (std::size_t i = 1; i < path.cells.size(); ++i) {
        const CellCoord d = path.cells[i] - path.cells[i - 1];
        const int manh = std::abs(d.x) + std::abs(d.y) + std::abs(d.z);
        if (manh != 1) {
            return false;
        }
    }
    return true;
}

static void test_focal_weight_one_matches_path() {
    StaticClearanceGrid static_grid(20, 10, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;

    FocalAStar focal;
    focal.set_focal_weight(1.0);
    const CellCoord start{0, 0, 0};
    const CellCoord goal{6, 0, 0};
    focal.reset(grid, start, goal, 0.1f, h, open, actions);
    EXPECT_TRUE(focal.search() == SearchStatus::Success);
    const Path p = focal.path();
    EXPECT_TRUE(p.cells.front() == start);
    EXPECT_TRUE(p.cells.back() == goal);
    EXPECT_TRUE(path_is_6_connected(p));
    EXPECT_NEAR(p.cost, 6.0, 1e-9);
}

static void test_focal_suboptimal_reaches() {
    StaticClearanceGrid static_grid(20, 10, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;

    FocalAStar focal;
    focal.set_focal_weight(2.5);
    const CellCoord start{0, 0, 0};
    const CellCoord goal{8, 3, 0};
    focal.reset(grid, start, goal, 0.1f, h, open, actions);
    EXPECT_TRUE(focal.search() == SearchStatus::Success);
    const Path p = focal.path();
    EXPECT_TRUE(p.cells.front() == start);
    EXPECT_TRUE(p.cells.back() == goal);
    EXPECT_TRUE(path_is_6_connected(p));
    EXPECT_TRUE(p.cost + 1e-9 >= 11.0);
}

static void test_start_equals_goal() {
    StaticClearanceGrid static_grid(5, 5, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;

    FocalAStar focal;
    focal.set_focal_weight(3.0);
    const CellCoord s{1, 1, 0};
    focal.reset(grid, s, s, 0.1f, h, open, actions);
    EXPECT_TRUE(focal.status() == SearchStatus::Success);
    EXPECT_NEAR(focal.path().cost, 0.0, 1e-9);
}

int main() {
    test_focal_weight_one_matches_path();
    test_focal_suboptimal_reaches();
    test_start_equals_goal();
    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
