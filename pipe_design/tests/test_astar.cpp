#include "geom/Types.h"
#include "grid/CompositeGrid.h"
#include "grid/ObstacleChain.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"
#include "heuristic/Geometric.h"
#include "planner/low/AStar.h"
#include "planner/low/Dijkstra.h"
#include "planner/low/DirectedAStar.h"
#include "planner/low/GreedyBestFirst.h"
#include "planner/low/LPAStar.h"
#include "planner/low/LazyThetaStar.h"
#include "planner/low/ThetaStar.h"
#include "planner/motion/SixConnected.h"
#include "planner/open/LPAOpenSet.h"
#include "planner/open/LazyBinaryHeap.h"

#include <cmath>
#include <iostream>
#include <vector>

using pipe_design::geom::SphereObstacle;
using pipe_design::geom::Vec3;
using pipe_design::grid::CellCoord;
using pipe_design::grid::CompositeGrid;
using pipe_design::grid::StaticClearanceGrid;
using pipe_design::heuristic::Euclidean;
using pipe_design::heuristic::Manhattan;
using pipe_design::planner::AStar;
using pipe_design::planner::Dijkstra;
using pipe_design::planner::DirectedAStar;
using pipe_design::planner::GreedyBestFirst;
using pipe_design::planner::LPAOpenSet;
using pipe_design::planner::LPAStar;
using pipe_design::planner::LazyThetaStar;
using pipe_design::planner::SearchNode;
using pipe_design::planner::LazyBinaryHeap;
using pipe_design::planner::Path;
using pipe_design::planner::SearchStatus;
using pipe_design::planner::ThetaStar;
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

static bool path_has_line_of_sight(const CompositeGrid& grid, const Path& path, float r) {
    if (path.cells.size() < 2) {
        return !path.cells.empty();
    }
    for (std::size_t i = 1; i < path.cells.size(); ++i) {
        if (!ThetaStar::line_of_sight(grid, path.cells[i - 1], path.cells[i], r)) {
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

static void test_dijkstra_matches_astar_cost() {
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
    LazyBinaryHeap open_d;

    const CellCoord start{0, 2, 0};
    const CellCoord goal{10, 2, 0};

    AStar<SearchNode> astar;
    astar.reset(grid, start, goal, 0.2f, h, open_a, actions);
    EXPECT_TRUE(astar.search() == SearchStatus::Success);

    Dijkstra dijkstra;
    dijkstra.reset(grid, start, goal, 0.2f, h, open_d, actions);
    EXPECT_TRUE(dijkstra.search() == SearchStatus::Success);

    EXPECT_NEAR(dijkstra.path().cost, astar.path().cost, 1e-9);
    EXPECT_TRUE(path_is_6_connected(dijkstra.path()));
    // Priority key is g only; at goal f == g.
    EXPECT_NEAR(dijkstra.goal_node()->f, dijkstra.goal_node()->g, 1e-9);
}

static void test_greedy_finds_path() {
    StaticClearanceGrid static_grid(20, 10, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;

    GreedyBestFirst greedy;
    const CellCoord start{0, 0, 0};
    const CellCoord goal{8, 4, 0};
    greedy.reset(grid, start, goal, 0.1f, h, open, actions);
    EXPECT_TRUE(greedy.search() == SearchStatus::Success);

    const Path p = greedy.path();
    EXPECT_TRUE(p.cells.front() == start);
    EXPECT_TRUE(p.cells.back() == goal);
    EXPECT_TRUE(path_is_6_connected(p));
    // Priority key is h only; at goal f == 0.
    EXPECT_NEAR(greedy.goal_node()->f, 0.0, 1e-9);
    EXPECT_NEAR(greedy.goal_node()->h, 0.0, 1e-9);
}

static void test_greedy_detour() {
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

    GreedyBestFirst greedy;
    greedy.reset(grid, CellCoord{0, 2, 0}, CellCoord{10, 2, 0}, 0.2f, h, open, actions);
    EXPECT_TRUE(greedy.search() == SearchStatus::Success);
    EXPECT_TRUE(path_is_6_connected(greedy.path()));
    EXPECT_TRUE(greedy.path().cost > 10.0);
}

static void test_theta_diagonal_shorter_than_astar() {
    StaticClearanceGrid static_grid(20, 20, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Euclidean h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open_a;
    LazyBinaryHeap open_t;

    const CellCoord start{0, 0, 0};
    const CellCoord goal{8, 8, 0};
    const float r = 0.1f;

    AStar<SearchNode> astar;
    astar.reset(grid, start, goal, r, h, open_a, actions);
    EXPECT_TRUE(astar.search() == SearchStatus::Success);

    ThetaStar theta;
    theta.set_step_cost(1.0);
    theta.reset(grid, start, goal, r, h, open_t, actions);
    EXPECT_TRUE(theta.search() == SearchStatus::Success);

    const Path p = theta.path();
    EXPECT_TRUE(p.cells.front() == start);
    EXPECT_TRUE(p.cells.back() == goal);
    EXPECT_TRUE(path_has_line_of_sight(grid, p, r));
    // Any-angle should beat 6-connect Manhattan (16).
    EXPECT_TRUE(p.cost < astar.path().cost);
    EXPECT_NEAR(p.cost, std::sqrt(2.0) * 8.0, 1e-6);
}

static void test_theta_blocked_detour() {
    StaticClearanceGrid static_grid(15, 10, 1, 1.f, 3.f);
    std::vector<SphereObstacle> walls;
    for (int y = 0; y <= 4; ++y) {
        walls.push_back(SphereObstacle{Vec3{5.5f, static_cast<float>(y) + 0.5f, 0.5f}, 0.6f});
    }
    static_grid.bake(walls);
    CompositeGrid grid(static_grid);
    Euclidean h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;

    ThetaStar theta;
    theta.set_step_cost(1.0);
    const CellCoord start{0, 2, 0};
    const CellCoord goal{10, 2, 0};
    const float r = 0.2f;
    theta.reset(grid, start, goal, r, h, open, actions);
    EXPECT_TRUE(theta.search() == SearchStatus::Success);
    const Path p = theta.path();
    EXPECT_TRUE(p.cells.front() == start);
    EXPECT_TRUE(p.cells.back() == goal);
    EXPECT_TRUE(path_has_line_of_sight(grid, p, r));
    EXPECT_TRUE(p.cost > 10.0);
}

static void test_lazy_theta_matches_theta_cost_open() {
    StaticClearanceGrid static_grid(20, 20, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Euclidean h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open_t;
    LazyBinaryHeap open_l;

    const CellCoord start{1, 1, 0};
    const CellCoord goal{12, 7, 0};
    const float r = 0.1f;

    ThetaStar theta;
    theta.set_step_cost(1.0);
    theta.reset(grid, start, goal, r, h, open_t, actions);
    EXPECT_TRUE(theta.search() == SearchStatus::Success);

    LazyThetaStar lazy;
    lazy.set_step_cost(1.0);
    lazy.reset(grid, start, goal, r, h, open_l, actions);
    EXPECT_TRUE(lazy.search() == SearchStatus::Success);

    EXPECT_NEAR(lazy.path().cost, theta.path().cost, 1e-6);
    EXPECT_TRUE(path_has_line_of_sight(grid, lazy.path(), r));
}

static void test_line_of_sight_blocked() {
    StaticClearanceGrid static_grid(10, 5, 1, 1.f, 3.f);
    std::vector<SphereObstacle> walls;
    for (int y = 0; y < 5; ++y) {
        walls.push_back(SphereObstacle{Vec3{4.5f, static_cast<float>(y) + 0.5f, 0.5f}, 0.7f});
    }
    static_grid.bake(walls);
    CompositeGrid grid(static_grid);
    EXPECT_TRUE(ThetaStar::line_of_sight(grid, CellCoord{0, 2, 0}, CellCoord{1, 2, 0}, 0.2f));
    EXPECT_TRUE(!ThetaStar::line_of_sight(grid, CellCoord{0, 2, 0}, CellCoord{8, 2, 0}, 0.2f));
}

static void test_lpa_straight_and_obstacle_update() {
    StaticClearanceGrid static_grid(20, 10, 1, 1.f, 5.f);
    static_grid.clear();
    CompositeGrid grid(static_grid);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LPAOpenSet open;

    LPAStar lpa;
    const CellCoord start{0, 0, 0};
    const CellCoord goal{5, 0, 0};
    lpa.reset(grid, start, goal, 0.1f, h, open, actions);
    EXPECT_TRUE(lpa.search() == SearchStatus::Success);
    Path p = lpa.path();
    EXPECT_TRUE(p.cells.front() == start);
    EXPECT_TRUE(p.cells.back() == goal);
    EXPECT_NEAR(p.cost, 5.0, 1e-9);
    EXPECT_TRUE(path_is_6_connected(p));

    // Block the corridor; reuse LPA state.
    std::vector<SphereObstacle> wall;
    for (int y = 0; y <= 4; ++y) {
        wall.push_back(SphereObstacle{Vec3{2.5f, static_cast<float>(y) + 0.5f, 0.5f}, 0.6f});
    }
    static_grid.bake(wall);
    std::vector<CellCoord> changed;
    for (int y = 0; y <= 4; ++y) {
        changed.push_back(CellCoord{2, y, 0});
    }
    lpa.update_obstacles(changed);
    EXPECT_TRUE(lpa.search() == SearchStatus::Success);
    p = lpa.path();
    EXPECT_TRUE(p.cells.front() == start);
    EXPECT_TRUE(p.cells.back() == goal);
    EXPECT_TRUE(path_is_6_connected(p));
    EXPECT_TRUE(p.cost > 5.0);
}

/// Fresh A* cost on the same grid (for LPA incremental checks).
static double astar_cost_on(
    CompositeGrid& grid,
    const CellCoord& start,
    const CellCoord& goal,
    float r,
    Manhattan& h,
    SixConnected& actions) {
    LazyBinaryHeap open;
    AStar<SearchNode> astar;
    astar.reset(grid, start, goal, r, h, open, actions);
    EXPECT_TRUE(astar.search() == SearchStatus::Success);
    return astar.path().cost;
}

static void test_lpa_add_remove_and_readd_same_obstacle() {
    StaticClearanceGrid static_grid(15, 10, 1, 1.f, 5.f);
    static_grid.clear();
    auto root = pipe_design::grid::ObstacleChain::make_root();
    CompositeGrid grid(static_grid, root);
    Manhattan h(1.0);
    SixConnected actions(1.0);
    LPAOpenSet open_lpa;

    const CellCoord start{0, 2, 0};
    const CellCoord goal{10, 2, 0};
    const float r = 0.2f;

    std::vector<SphereObstacle> wall;
    for (int y = 0; y <= 4; ++y) {
        wall.push_back(SphereObstacle{Vec3{5.5f, static_cast<float>(y) + 0.5f, 0.5f}, 0.6f});
    }
    auto with_wall = root->extend(wall, {});

    std::vector<CellCoord> changed;
    for (int y = 0; y <= 5; ++y) {
        changed.push_back(CellCoord{5, y, 0});
        changed.push_back(CellCoord{4, y, 0});
        changed.push_back(CellCoord{6, y, 0});
    }

    auto check_path = [&](const Path& p, double expect_cost, bool blocked_map) {
        EXPECT_TRUE(p.cells.front() == start);
        EXPECT_TRUE(p.cells.back() == goal);
        EXPECT_TRUE(path_is_6_connected(p));
        EXPECT_NEAR(p.cost, expect_cost, 1e-9);
        for (const CellCoord& c : p.cells) {
            EXPECT_TRUE(grid.is_free(c, r));
        }
        if (blocked_map) {
            EXPECT_TRUE(p.cost > 10.0 - 1e-9);
        }
    };

    LPAStar lpa;
    lpa.reset(grid, start, goal, r, h, open_lpa, actions);
    EXPECT_TRUE(lpa.search() == SearchStatus::Success);
    const double cost_empty = lpa.path().cost;
    EXPECT_NEAR(cost_empty, 10.0, 1e-9);
    EXPECT_NEAR(cost_empty, astar_cost_on(grid, start, goal, r, h, actions), 1e-9);

    // --- 1) add obstacle + replan ---
    grid.set_chain(with_wall);
    lpa.update_obstacles(changed);
    EXPECT_TRUE(lpa.search() == SearchStatus::Success);
    const double cost_blocked = lpa.path().cost;
    check_path(lpa.path(), cost_blocked, /*blocked_map=*/true);
    EXPECT_NEAR(cost_blocked, astar_cost_on(grid, start, goal, r, h, actions), 1e-9);
    EXPECT_TRUE(cost_blocked > cost_empty);

    // --- 2) remove same obstacle + replan ---
    grid.set_chain(root);
    lpa.update_obstacles(changed);
    EXPECT_TRUE(lpa.search() == SearchStatus::Success);
    check_path(lpa.path(), cost_empty, /*blocked_map=*/false);
    EXPECT_NEAR(lpa.path().cost, astar_cost_on(grid, start, goal, r, h, actions), 1e-9);

    // --- 3) add → replan → remove (same) → replan → add (same) → replan ---
    grid.set_chain(with_wall);
    lpa.update_obstacles(changed);
    EXPECT_TRUE(lpa.search() == SearchStatus::Success);
    EXPECT_NEAR(lpa.path().cost, cost_blocked, 1e-9);
    EXPECT_NEAR(lpa.path().cost, astar_cost_on(grid, start, goal, r, h, actions), 1e-9);

    grid.set_chain(root);
    lpa.update_obstacles(changed);
    EXPECT_TRUE(lpa.search() == SearchStatus::Success);
    EXPECT_NEAR(lpa.path().cost, cost_empty, 1e-9);
    EXPECT_NEAR(lpa.path().cost, astar_cost_on(grid, start, goal, r, h, actions), 1e-9);

    grid.set_chain(with_wall);
    lpa.update_obstacles(changed);
    EXPECT_TRUE(lpa.search() == SearchStatus::Success);
    const Path p_final = lpa.path();
    check_path(p_final, cost_blocked, /*blocked_map=*/true);
    EXPECT_NEAR(p_final.cost, astar_cost_on(grid, start, goal, r, h, actions), 1e-9);
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
    test_dijkstra_matches_astar_cost();
    test_greedy_finds_path();
    test_greedy_detour();
    test_theta_diagonal_shorter_than_astar();
    test_theta_blocked_detour();
    test_lazy_theta_matches_theta_cost_open();
    test_line_of_sight_blocked();
    test_lpa_straight_and_obstacle_update();
    test_lpa_add_remove_and_readd_same_obstacle();

    if (g_failures != 0) {
        std::cerr << g_failures << " assertion(s) failed.\n";
        return 1;
    }
    std::cout << "All A* tests passed.\n";
    return 0;
}
