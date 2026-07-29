#include "grid/CompositeGrid.h"
#include "grid/ObstacleChain.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"

#include "geom/Types.h"

#include <cassert>
#include <cmath>
#include <iostream>

using pipe_design::geom::CapsuleObstacle;
using pipe_design::geom::SphereObstacle;
using pipe_design::geom::Vec3;
using pipe_design::grid::CellCoord;
using pipe_design::grid::CompositeGrid;
using pipe_design::grid::ObstacleChain;
using pipe_design::grid::StaticClearanceGrid;

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

void test_obstacle_chain_sharing() {
    auto root = ObstacleChain::make_root();
    EXPECT_TRUE(root->total_count() == 0);

    auto child_a = root->extend(SphereObstacle{Vec3{5.f, 5.f, 5.f}, 1.f});
    auto child_b = root->extend(SphereObstacle{Vec3{8.f, 5.f, 5.f}, 1.f});
    auto child_a2 = child_a->extend(CapsuleObstacle{Vec3{0.f, 0.f, 0.f}, Vec3{2.f, 0.f, 0.f}, 0.5f});

    EXPECT_TRUE(child_a->total_count() == 1);
    EXPECT_TRUE(child_b->total_count() == 1);
    EXPECT_TRUE(child_a2->total_count() == 2);
    EXPECT_TRUE(child_a->parent() == root.get());
    EXPECT_TRUE(child_a2->parent() == child_a.get());

    EXPECT_TRUE(child_a->collides(Vec3{5.f, 5.f, 5.f}, 0.1f));
    EXPECT_TRUE(!child_a->collides(Vec3{8.f, 5.f, 5.f}, 0.1f));
    EXPECT_TRUE(child_b->collides(Vec3{8.f, 5.f, 5.f}, 0.1f));
    EXPECT_TRUE(!child_b->collides(Vec3{5.f, 5.f, 5.f}, 0.1f));

    EXPECT_TRUE(child_a2->collides(Vec3{5.f, 5.f, 5.f}, 0.1f));
    EXPECT_TRUE(child_a2->collides(Vec3{1.f, 0.f, 0.f}, 0.1f));
}

void test_static_clearance_bake() {
    StaticClearanceGrid grid(20, 20, 1, 1.f, 3.f);
    grid.bake({SphereObstacle{Vec3{10.5f, 10.5f, 0.5f}, 1.f}});

    EXPECT_TRUE(!grid.is_free(CellCoord{10, 10, 0}, 0.5f));
    EXPECT_NEAR(grid.clearance(CellCoord{0, 0, 0}), 3.f, 1e-4f);
    EXPECT_TRUE(grid.is_free(CellCoord{0, 0, 0}, 1.f));
    EXPECT_NEAR(grid.clearance(CellCoord{13, 10, 0}), 2.f, 1e-3f);
}

void test_composite_grid() {
    StaticClearanceGrid static_grid(20, 20, 1, 1.f, 3.f);
    static_grid.bake({SphereObstacle{Vec3{5.5f, 5.5f, 0.5f}, 1.f}});

    auto root = ObstacleChain::make_root();
    auto chain = root->extend(SphereObstacle{Vec3{15.5f, 5.5f, 0.5f}, 1.f});

    CompositeGrid composite(static_grid, chain);

    EXPECT_TRUE(!composite.is_free(CellCoord{5, 5, 0}, 0.5f));
    EXPECT_TRUE(!composite.is_free(CellCoord{15, 5, 0}, 0.5f));
    EXPECT_TRUE(composite.is_free(CellCoord{0, 0, 0}, 1.f));

    CompositeGrid static_only(static_grid, root);
    EXPECT_TRUE(static_only.is_free(CellCoord{15, 5, 0}, 0.5f));
}

int main() {
    test_obstacle_chain_sharing();
    test_static_clearance_bake();
    test_composite_grid();

    if (g_failures == 0) {
        std::cout << "All grid tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " assertion(s) failed.\n";
    return 1;
}
