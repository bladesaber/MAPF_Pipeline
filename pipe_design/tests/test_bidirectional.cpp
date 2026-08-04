#include "geom/Types.h"
#include "grid/CompositeGrid.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"
#include "heuristic/Geometric.h"
#include "planner/middle/BidirectionalAStar.h"
#include "planner/motion/SixConnected.h"
#include "planner/open/LazyBinaryHeap.h"

#include <cmath>
#include <iostream>

using pipe_design::grid::CellCoord;
using pipe_design::grid::CompositeGrid;
using pipe_design::grid::StaticClearanceGrid;
using pipe_design::heuristic::Manhattan;
using pipe_design::planner::BidirectionalAStar;
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

static void test_start_equals_goal() {
    StaticClearanceGrid static_grid(5, 5, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open_fwd, open_bwd;

    BidirectionalAStar bi;
    const CellCoord s{2, 2, 0};
    bi.reset(grid, s, s, 0.1f, h, open_fwd, open_bwd, actions);
    EXPECT_TRUE(bi.status() == SearchStatus::Success);
    EXPECT_TRUE(bi.expansions() == 0);
    const Path p = bi.path();
    EXPECT_TRUE(p.length() == 1);
    EXPECT_TRUE(p.cells[0] == s);
    EXPECT_NEAR(p.cost, 0.0, 1e-9);
}

static void test_straight_line() {
    StaticClearanceGrid static_grid(20, 10, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open_fwd, open_bwd;

    BidirectionalAStar bi;
    const CellCoord start{0, 0, 0};
    const CellCoord goal{5, 0, 0};
    bi.reset(grid, start, goal, 0.1f, h, open_fwd, open_bwd, actions);
    EXPECT_TRUE(bi.search() == SearchStatus::Success);
    const Path p = bi.path();
    EXPECT_TRUE(!p.empty());
    EXPECT_TRUE(p.cells.front() == start);
    EXPECT_TRUE(p.cells.back() == goal);
    EXPECT_TRUE(path_is_6_connected(p));
    EXPECT_NEAR(p.cost, 5.0, 1e-9);
}

static void test_weighted_still_reaches() {
    StaticClearanceGrid static_grid(20, 10, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open_fwd, open_bwd;

    BidirectionalAStar bi;
    bi.set_heuristic_weight(2.0);
    const CellCoord start{0, 0, 0};
    const CellCoord goal{8, 0, 0};
    bi.reset(grid, start, goal, 0.1f, h, open_fwd, open_bwd, actions);
    EXPECT_TRUE(bi.search() == SearchStatus::Success);
    const Path p = bi.path();
    EXPECT_TRUE(p.cells.front() == start);
    EXPECT_TRUE(p.cells.back() == goal);
    EXPECT_TRUE(path_is_6_connected(p));
    EXPECT_TRUE(p.cost >= 8.0 - 1e-9);
}

static void test_blocked_start() {
    StaticClearanceGrid static_grid(10, 5, 1, 1.f, 3.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open_fwd, open_bwd;

    BidirectionalAStar bi;
    // radius larger than clearance_cap → nowhere free
    bi.reset(grid, CellCoord{0, 0, 0}, CellCoord{5, 0, 0}, 10.f, h, open_fwd, open_bwd, actions);
    EXPECT_TRUE(bi.status() == SearchStatus::Failure);
    EXPECT_TRUE(bi.path().empty());
}

int main() {
    test_start_equals_goal();
    test_straight_line();
    test_weighted_still_reaches();
    test_blocked_start();
    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
