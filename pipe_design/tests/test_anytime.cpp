#include "geom/Types.h"
#include "grid/CompositeGrid.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"
#include "heuristic/Geometric.h"
#include "planner/low/ANAStar.h"
#include "planner/low/ARAStar.h"
#include "planner/low/AStar.h"
#include "planner/motion/SixConnected.h"
#include "planner/open/LazyBinaryHeap.h"

#include <cmath>
#include <iostream>
#include <vector>

using pipe_design::geom::SphereObstacle;
using pipe_design::geom::Vec3;
using pipe_design::grid::CellCoord;
using pipe_design::grid::CompositeGrid;
using pipe_design::grid::StaticClearanceGrid;
using pipe_design::heuristic::Manhattan;
using pipe_design::planner::ANAStar;
using pipe_design::planner::ARAStar;
using pipe_design::planner::AStar;
using pipe_design::planner::LazyBinaryHeap;
using pipe_design::planner::Path;
using pipe_design::planner::SearchNode;
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

static void test_ara_matches_astar_when_eps_one() {
    StaticClearanceGrid static_grid(15, 10, 1, 1.f, 3.f);
    std::vector<SphereObstacle> walls;
    for (int y = 0; y <= 4; ++y) {
        walls.push_back(SphereObstacle{Vec3{5.5f, static_cast<float>(y) + 0.5f, 0.5f}, 0.6f});
    }
    static_grid.bake(walls);
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open_a;
    LazyBinaryHeap open_r;

    const CellCoord start{0, 2, 0};
    const CellCoord goal{10, 2, 0};

    AStar<SearchNode> astar;
    astar.reset(grid, start, goal, 0.2f, h, open_a, actions);
    EXPECT_TRUE(astar.search() == SearchStatus::Success);

    ARAStar ara;
    ara.set_eps_schedule(1.0, 1.0, 0.5);
    ara.reset(grid, start, goal, 0.2f, h, open_r, actions);
    EXPECT_TRUE(ara.search() == SearchStatus::Success);
    EXPECT_NEAR(ara.path().cost, astar.path().cost, 1e-9);
    EXPECT_TRUE(path_is_6_connected(ara.path()));
}

static void test_ara_anytime_improves() {
    StaticClearanceGrid static_grid(20, 12, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;

    ARAStar ara;
    ara.set_eps_schedule(5.0, 1.0, 1.0);
    const CellCoord start{0, 0, 0};
    const CellCoord goal{15, 8, 0};
    ara.reset(grid, start, goal, 0.1f, h, open, actions);

    // Budget: should get an incumbent before finishing all ε rounds.
    ara.search(/*max_expansions=*/80);
    EXPECT_TRUE(ara.goal_node() != nullptr);
    EXPECT_TRUE(!ara.path().empty());
    const double early_cost = ara.path().cost;

    ara.search();  // finish
    EXPECT_TRUE(ara.status() == SearchStatus::Success);
    EXPECT_TRUE(ara.path().cost <= early_cost + 1e-9);
    EXPECT_NEAR(ara.path().cost, 23.0, 1e-9);  // Manhattan optimal on empty grid
    EXPECT_TRUE(path_is_6_connected(ara.path()));
}

static void test_ana_finds_optimal() {
    StaticClearanceGrid static_grid(15, 10, 1, 1.f, 3.f);
    std::vector<SphereObstacle> walls;
    for (int y = 0; y <= 4; ++y) {
        walls.push_back(SphereObstacle{Vec3{5.5f, static_cast<float>(y) + 0.5f, 0.5f}, 0.6f});
    }
    static_grid.bake(walls);
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open_a;
    LazyBinaryHeap open_n;

    const CellCoord start{0, 2, 0};
    const CellCoord goal{10, 2, 0};

    AStar<SearchNode> astar;
    astar.reset(grid, start, goal, 0.2f, h, open_a, actions);
    EXPECT_TRUE(astar.search() == SearchStatus::Success);

    ANAStar ana;
    ana.reset(grid, start, goal, 0.2f, h, open_n, actions);
    EXPECT_TRUE(ana.search() == SearchStatus::Success);
    EXPECT_NEAR(ana.path().cost, astar.path().cost, 1e-9);
    EXPECT_NEAR(ana.incumbent_cost(), astar.path().cost, 1e-9);
    EXPECT_TRUE(path_is_6_connected(ana.path()));
}

static void test_ana_incumbent_before_done() {
    StaticClearanceGrid static_grid(25, 15, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;

    ANAStar ana;
    ana.reset(grid, CellCoord{0, 0, 0}, CellCoord{20, 10, 0}, 0.1f, h, open, actions);
    ana.search(/*max_expansions=*/50);
    EXPECT_TRUE(ana.goal_node() != nullptr);
    EXPECT_TRUE((ana.path().cells.front() == CellCoord{0, 0, 0}));
    EXPECT_TRUE((ana.path().cells.back() == CellCoord{20, 10, 0}));

    ana.search();
    EXPECT_TRUE(ana.status() == SearchStatus::Success);
    EXPECT_NEAR(ana.path().cost, 30.0, 1e-9);
}

int main() {
    test_ara_matches_astar_when_eps_one();
    test_ara_anytime_improves();
    test_ana_finds_optimal();
    test_ana_incumbent_before_done();

    if (g_failures != 0) {
        std::cerr << g_failures << " assertion(s) failed.\n";
        return 1;
    }
    std::cout << "All anytime search tests passed.\n";
    return 0;
}
