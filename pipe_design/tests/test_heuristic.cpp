#include "geom/Types.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"
#include "heuristic/Geometric.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using pipe_design::geom::SphereObstacle;
using pipe_design::geom::Vec3;
using pipe_design::grid::CellCoord;
using pipe_design::grid::StaticClearanceGrid;
using pipe_design::heuristic::Dijkstra;
using pipe_design::heuristic::Euclidean;
using pipe_design::heuristic::Heuristic;
using pipe_design::heuristic::Manhattan;
using pipe_design::heuristic::Octile;

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

static void test_manhattan() {
    Manhattan h(1.0);
    EXPECT_NEAR(h.estimate(CellCoord{0, 0, 0}, CellCoord{3, 4, 0}), 7.0, 1e-9);
    Manhattan h2(2.0);
    EXPECT_NEAR(h2.estimate(CellCoord{0, 0, 0}, CellCoord{1, 0, 0}), 2.0, 1e-9);
}

static void test_euclidean() {
    Euclidean h(1.0);
    EXPECT_NEAR(h.estimate(CellCoord{0, 0, 0}, CellCoord{3, 4, 0}), 5.0, 1e-9);
}

static void test_octile() {
    Octile h(1.0);
    EXPECT_NEAR(h.estimate(CellCoord{0, 0, 0}, CellCoord{5, 0, 0}), 5.0, 1e-9);
    EXPECT_NEAR(h.estimate(CellCoord{0, 0, 0}, CellCoord{1, 1, 0}), std::sqrt(2.0), 1e-9);
    EXPECT_NEAR(
        h.estimate(CellCoord{0, 0, 0}, CellCoord{3, 4, 0}),
        (std::sqrt(2.0) - 1.0) * 3.0 + 4.0,
        1e-9);
    EXPECT_NEAR(h.estimate(CellCoord{0, 0, 0}, CellCoord{1, 1, 1}), std::sqrt(3.0), 1e-9);
}

static void test_polymorphic() {
    Manhattan manh(1.0);
    const Heuristic& h = manh;
    EXPECT_NEAR(h.estimate(CellCoord{1, 1, 1}, CellCoord{2, 3, 4}), 6.0, 1e-9);
}

static void test_dijkstra_empty() {
    StaticClearanceGrid grid(10, 10, 1, 1.f, 5.f);
    grid.clear();
    Dijkstra h(grid, 1.0);
    const CellCoord goal{7, 2, 0};
    h.prepare(goal);

    EXPECT_TRUE(h.prepared());
    EXPECT_NEAR(h.estimate(CellCoord{7, 2, 0}, goal), 0.0, 1e-9);
    EXPECT_NEAR(h.estimate(CellCoord{5, 2, 0}, goal), 2.0, 1e-9);
    EXPECT_NEAR(h.estimate(CellCoord{0, 0, 0}, goal), 9.0, 1e-9);
    // `to` ignored when prepared (multi-source API)
    EXPECT_NEAR(h.estimate(CellCoord{5, 2, 0}, CellCoord{0, 0, 0}), 2.0, 1e-9);
}

static void test_dijkstra_multi_source() {
    StaticClearanceGrid grid(20, 10, 1, 1.f, 5.f);
    grid.clear();
    Dijkstra h(grid, 1.0);
    h.prepare({CellCoord{0, 0, 0}, CellCoord{19, 0, 0}});

    EXPECT_TRUE(h.prepared());
    EXPECT_NEAR(h.distance(CellCoord{0, 0, 0}), 0.0, 1e-9);
    EXPECT_NEAR(h.distance(CellCoord{19, 0, 0}), 0.0, 1e-9);
    // Midpoint equidistant → 9.5? with integer steps: cell 9 and 10
    EXPECT_NEAR(h.distance(CellCoord{9, 0, 0}), 9.0, 1e-9);
    EXPECT_NEAR(h.distance(CellCoord{10, 0, 0}), 9.0, 1e-9);
}

static void test_dijkstra_around_obstacle() {
    StaticClearanceGrid grid(12, 8, 1, 1.f, 3.f);
    std::vector<SphereObstacle> walls;
    for (int y = 0; y < 6; ++y) {
        walls.push_back(SphereObstacle{Vec3{5.5f, static_cast<float>(y) + 0.5f, 0.5f}, 0.6f});
    }
    grid.bake(walls);

    Dijkstra h(grid, 1.0);
    const CellCoord goal{10, 2, 0};
    h.prepare(goal);

    const double manh = 10.0;
    const double td = h.estimate(CellCoord{0, 2, 0}, goal);
    EXPECT_TRUE(std::isfinite(td));
    EXPECT_TRUE(td > manh);
}

static void test_dijkstra_blocked_goal() {
    StaticClearanceGrid grid(5, 5, 1, 1.f, 3.f);
    grid.bake({SphereObstacle{Vec3{2.5f, 2.5f, 0.5f}, 1.f}});
    Dijkstra h(grid, 1.0);
    h.prepare(CellCoord{2, 2, 0});
    // Center cell has clearance 0 → not traversable under relaxed bake
    EXPECT_TRUE(!h.prepared());
    EXPECT_TRUE(h.distance(CellCoord{0, 0, 0}) == Dijkstra::kUnreachable);
}

static void test_dijkstra_clear() {
    StaticClearanceGrid grid(8, 8, 1, 1.f, 5.f);
    grid.clear();
    Dijkstra h(grid, 1.0);
    h.prepare(CellCoord{4, 4, 0});
    EXPECT_TRUE(h.prepared());
    h.clear();
    EXPECT_TRUE(!h.prepared());
    EXPECT_TRUE(h.goals().empty());
    EXPECT_NEAR(h.estimate(CellCoord{0, 0, 0}, CellCoord{4, 4, 0}), 0.0, 1e-9);
}

int main() {
    test_manhattan();
    test_euclidean();
    test_octile();
    test_polymorphic();
    test_dijkstra_empty();
    test_dijkstra_multi_source();
    test_dijkstra_around_obstacle();
    test_dijkstra_blocked_goal();
    test_dijkstra_clear();

    if (g_failures != 0) {
        std::cerr << g_failures << " assertion(s) failed.\n";
        return 1;
    }
    std::cout << "All heuristic tests passed.\n";
    return 0;
}
