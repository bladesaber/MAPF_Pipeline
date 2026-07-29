#include "geom/Types.h"
#include "grid/CompositeGrid.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"
#include "heuristic/Geometric.h"
#include "planner/low/AStar.h"
#include "planner/low/DirectedAStar.h"
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
using pipe_design::planner::AStar;
using pipe_design::planner::DirectedAStar;
using pipe_design::planner::SearchNode;
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
    LazyBinaryHeap open;

    AStar<SearchNode> astar;
    const CellCoord s{2, 2, 0};
    astar.reset(grid, s, s, 0.1f, h, open, actions);
    EXPECT_TRUE(astar.status() == SearchStatus::Success);
    EXPECT_TRUE(astar.expansions() == 0);
    EXPECT_TRUE(astar.is_goal(s));
    const Path p = astar.path();
    EXPECT_TRUE(p.length() == 1);
    EXPECT_TRUE(p.cells[0] == s);
    EXPECT_NEAR(p.cost, 0.0, 1e-9);
}

static void test_empty_straight_line() {
    StaticClearanceGrid static_grid(20, 10, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;

    AStar<SearchNode> astar;
    const CellCoord start{0, 0, 0};
    const CellCoord goal{5, 0, 0};
    astar.reset(grid, start, goal, 0.1f, h, open, actions);
    EXPECT_TRUE(astar.search() == SearchStatus::Success);

    const Path p = astar.path();
    EXPECT_TRUE(p.cells.front() == start);
    EXPECT_TRUE(p.cells.back() == goal);
    EXPECT_TRUE(p.length() == 6);
    EXPECT_NEAR(p.cost, 5.0, 1e-9);
    EXPECT_TRUE(path_is_6_connected(p));
    EXPECT_TRUE(astar.goal_node() != nullptr);
    EXPECT_NEAR(astar.goal_node()->h, 0.0, 1e-9);
}

static void test_blocked_detour() {
    StaticClearanceGrid static_grid(15, 10, 1, 1.f, 3.f);
    std::vector<SphereObstacle> walls;
    for (int y = 0; y <= 4; ++y) {
        walls.push_back(SphereObstacle{Vec3{5.5f, static_cast<float>(y) + 0.5f, 0.5f}, 0.6f});
    }
    static_grid.bake(walls);
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;

    AStar<SearchNode> astar;
    const CellCoord start{0, 2, 0};
    const CellCoord goal{10, 2, 0};
    astar.reset(grid, start, goal, 0.2f, h, open, actions);
    EXPECT_TRUE(astar.search() == SearchStatus::Success);

    const Path p = astar.path();
    EXPECT_TRUE(p.cells.front() == start);
    EXPECT_TRUE(p.cells.back() == goal);
    EXPECT_TRUE(path_is_6_connected(p));
    EXPECT_TRUE(p.cost > 10.0);
}

static void test_fully_blocked_fails() {
    StaticClearanceGrid static_grid(10, 5, 1, 1.f, 3.f);
    std::vector<SphereObstacle> walls;
    for (int y = 0; y < 5; ++y) {
        walls.push_back(SphereObstacle{Vec3{4.5f, static_cast<float>(y) + 0.5f, 0.5f}, 0.7f});
    }
    static_grid.bake(walls);
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;

    AStar<SearchNode> astar;
    astar.reset(grid, CellCoord{0, 2, 0}, CellCoord{8, 2, 0}, 0.2f, h, open, actions);
    EXPECT_TRUE(astar.search() == SearchStatus::Failure);
    EXPECT_TRUE(astar.path().empty());
}

static void test_step_matches_search() {
    StaticClearanceGrid static_grid(12, 8, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open_a;
    LazyBinaryHeap open_b;

    const CellCoord start{1, 1, 0};
    const CellCoord goal{8, 5, 0};

    AStar<SearchNode> via_search;
    via_search.reset(grid, start, goal, 0.1f, h, open_a, actions);
    EXPECT_TRUE(via_search.search() == SearchStatus::Success);
    const Path p_search = via_search.path();

    AStar<SearchNode> via_step;
    via_step.reset(grid, start, goal, 0.1f, h, open_b, actions);
    EXPECT_TRUE(via_step.status() == SearchStatus::Running);
    while (via_step.status() == SearchStatus::Running) {
        via_step.step();
    }
    EXPECT_TRUE(via_step.status() == SearchStatus::Success);
    const Path p_step = via_step.path();

    EXPECT_TRUE(p_step.cells == p_search.cells);
    EXPECT_NEAR(p_step.cost, p_search.cost, 1e-9);
    EXPECT_TRUE(via_step.expansions() == via_search.expansions());
    EXPECT_TRUE(via_step.last_expanded() != nullptr);
    EXPECT_TRUE(via_step.last_expanded()->coord == goal);
}

static void test_invalid_start_fails() {
    StaticClearanceGrid static_grid(8, 8, 1, 1.f, 3.f);
    static_grid.bake({SphereObstacle{Vec3{1.5f, 1.5f, 0.5f}, 1.f}});
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;

    AStar<SearchNode> astar;
    astar.reset(grid, CellCoord{1, 1, 0}, CellCoord{6, 6, 0}, 0.5f, h, open, actions);
    EXPECT_TRUE(astar.status() == SearchStatus::Failure);
}

static void test_multi_start_multi_goal() {
    StaticClearanceGrid static_grid(20, 10, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;

    AStar<SearchNode> astar;
    const std::vector<CellCoord> starts{{0, 0, 0}, {0, 9, 0}};
    const std::vector<CellCoord> goals{{19, 0, 0}, {19, 9, 0}};
    astar.reset(grid, starts, goals, 0.1f, h, open, actions);
    EXPECT_TRUE(astar.search() == SearchStatus::Success);

    const Path p = astar.path();
    EXPECT_TRUE(astar.is_goal(p.cells.back()));
    EXPECT_TRUE(
        (p.cells.front() == CellCoord{0, 0, 0}) || (p.cells.front() == CellCoord{0, 9, 0}));
    EXPECT_NEAR(p.cost, 19.0, 1e-9);
    EXPECT_TRUE(path_is_6_connected(p));
}

static void test_weighted_combine_key() {
    StaticClearanceGrid static_grid(10, 5, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;

    AStar<SearchNode> astar;
    astar.set_heuristic_weight(2.0);
    astar.reset(grid, CellCoord{0, 0, 0}, CellCoord{4, 0, 0}, 0.1f, h, open, actions);
    EXPECT_TRUE(astar.search() == SearchStatus::Success);

    const Path p = astar.path();
    EXPECT_NEAR(p.cost, 4.0, 1e-9);
    EXPECT_TRUE(astar.goal_node()->h == 0.0);
    EXPECT_NEAR(astar.goal_node()->f, 4.0, 1e-9);
}

static void test_directed_with_turn_weight() {
    StaticClearanceGrid static_grid(12, 8, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    actions.set_turn_weight(2.0);
    LazyBinaryHeap open;

    DirectedAStar astar;
    astar.set_heading_bins(8, 1);
    const CellCoord start{0, 0, 0};
    const CellCoord goal{5, 3, 0};
    astar.reset(grid, start, goal, 0.1f, h, open, actions);
    EXPECT_TRUE(astar.search() == SearchStatus::Success);

    const Path p = astar.path();
    EXPECT_TRUE(p.cells.front() == start);
    EXPECT_TRUE(p.cells.back() == goal);
    EXPECT_TRUE(path_is_6_connected(p));
    // With turn cost, g may exceed pure Manhattan length.
    EXPECT_TRUE(p.cost >= 8.0);
    EXPECT_TRUE(astar.goal_node() != nullptr);
    EXPECT_TRUE(!astar.goal_node()->in_dir.is_none());
}

int main() {
    test_start_equals_goal();
    test_empty_straight_line();
    test_blocked_detour();
    test_fully_blocked_fails();
    test_step_matches_search();
    test_invalid_start_fails();
    test_multi_start_multi_goal();
    test_weighted_combine_key();
    test_directed_with_turn_weight();

    if (g_failures != 0) {
        std::cerr << g_failures << " assertion(s) failed.\n";
        return 1;
    }
    std::cout << "All A* tests passed.\n";
    return 0;
}
