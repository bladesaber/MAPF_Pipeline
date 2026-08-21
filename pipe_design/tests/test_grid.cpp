#include "grid/CompositeGrid.h"
#include "grid/ObstacleChain.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"

#include "geom/Geometry.h"
#include "geom/Types.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

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

void test_segment_capsule_signed_clearance() {
    using pipe_design::geom::signed_clearance;
    using pipe_design::geom::segment_to_segment_distance;
    using pipe_design::geom::AabbObstacle;

    // Parallel skew: query segment along x at y=2, capsule along x at y=0, r=0.5
    // → distance 2, signed clearance 1.5
    const CapsuleObstacle cap{Vec3{0.f, 0.f, 0.f}, Vec3{10.f, 0.f, 0.f}, 0.5f};
    EXPECT_NEAR(
        signed_clearance(Vec3{1.f, 2.f, 0.f}, Vec3{3.f, 2.f, 0.f}, cap),
        1.5f,
        1e-4f);

    // Crossing through capsule axis → penetrating
    EXPECT_TRUE(
        signed_clearance(Vec3{5.f, -1.f, 0.f}, Vec3{5.f, 1.f, 0.f}, cap) < 0.f);

    // Endpoint–endpoint degenerate
    EXPECT_NEAR(
        segment_to_segment_distance(
            Vec3{0.f, 0.f, 0.f},
            Vec3{0.f, 0.f, 0.f},
            Vec3{3.f, 4.f, 0.f},
            Vec3{3.f, 4.f, 0.f}),
        5.f,
        1e-4f);

    // Segment vs sphere: closest approach at mid
    const SphereObstacle sph{Vec3{0.f, 0.f, 0.f}, 1.f};
    EXPECT_NEAR(
        signed_clearance(Vec3{0.f, 3.f, 0.f}, Vec3{0.f, 5.f, 0.f}, sph),
        2.f,
        1e-4f);

    // Point outside AABB
    const AabbObstacle box{Vec3{0.f, 0.f, 0.f}, Vec3{2.f, 2.f, 2.f}};
    EXPECT_NEAR(signed_clearance(Vec3{4.f, 1.f, 1.f}, box), 2.f, 1e-4f);
    EXPECT_TRUE(signed_clearance(Vec3{1.f, 1.f, 1.f}, box) < 0.f);
    // Segment grazing past face at y=3 (box ends at y=2) → clearance 1
    EXPECT_NEAR(
        signed_clearance(Vec3{-1.f, 3.f, 1.f}, Vec3{3.f, 3.f, 1.f}, box),
        1.f,
        1e-3f);
}

void test_obstacle_chain_sharing() {
    auto root = ObstacleChain::make_root();
    EXPECT_TRUE(root->total_count() == 0);

    auto child_a = root->extend(SphereObstacle{Vec3{5.f, 5.f, 5.f}, 1.f});
    auto child_b = root->extend(SphereObstacle{Vec3{8.f, 5.f, 5.f}, 1.f});
    auto child_a2 = child_a->extend(CapsuleObstacle{Vec3{0.f, 0.f, 0.f}, Vec3{2.f, 0.f, 0.f}, 0.5f});
    auto child_box = root->extend(
        pipe_design::geom::AabbObstacle{Vec3{10.f, 10.f, 10.f}, Vec3{12.f, 12.f, 12.f}});

    EXPECT_TRUE(child_a->total_count() == 1);
    EXPECT_TRUE(child_b->total_count() == 1);
    EXPECT_TRUE(child_a2->total_count() == 2);
    EXPECT_TRUE(child_box->total_count() == 1);
    EXPECT_TRUE(child_a->parent() == root.get());
    EXPECT_TRUE(child_a2->parent() == child_a.get());

    EXPECT_TRUE(child_a->collides(Vec3{5.f, 5.f, 5.f}, 0.1f));
    EXPECT_TRUE(!child_a->collides(Vec3{8.f, 5.f, 5.f}, 0.1f));
    EXPECT_TRUE(child_b->collides(Vec3{8.f, 5.f, 5.f}, 0.1f));
    EXPECT_TRUE(!child_b->collides(Vec3{5.f, 5.f, 5.f}, 0.1f));

    EXPECT_TRUE(child_a2->collides(Vec3{5.f, 5.f, 5.f}, 0.1f));
    EXPECT_TRUE(child_a2->collides(Vec3{1.f, 0.f, 0.f}, 0.1f));
    EXPECT_TRUE(child_box->collides(Vec3{11.f, 11.f, 11.f}, 0.1f));
    EXPECT_TRUE(!child_box->collides(Vec3{5.f, 5.f, 5.f}, 0.1f));
}

void test_static_clearance_bake() {
    StaticClearanceGrid grid(20, 20, 1, 1.f, 3.f);
    grid.bake({SphereObstacle{Vec3{10.5f, 10.5f, 0.5f}, 1.f}});

    EXPECT_TRUE(grid.spheres().size() == 1);
    EXPECT_TRUE(grid.capsules().empty());
    EXPECT_TRUE(grid.aabbs().empty());
    EXPECT_TRUE(!grid.is_free(CellCoord{10, 10, 0}, 0.5f));
    EXPECT_NEAR(grid.clearance(CellCoord{0, 0, 0}), 3.f, 1e-4f);
    EXPECT_TRUE(grid.is_free(CellCoord{0, 0, 0}, 1.f));
    EXPECT_NEAR(grid.clearance(CellCoord{13, 10, 0}), 2.f, 1e-3f);

    // Analytic world-point query matches signed clearance to retained sphere.
    const Vec3 near_obs{11.5f, 10.5f, 0.5f};
    EXPECT_NEAR(grid.clearance(near_obs), 0.f, 1e-4f);
    EXPECT_TRUE(!grid.is_free(near_obs, 0.1f));

    grid.clear();
    EXPECT_TRUE(grid.spheres().empty());
    EXPECT_NEAR(grid.clearance(near_obs), 3.f, 1e-4f);

    StaticClearanceGrid grid_box(20, 20, 1, 1.f, 3.f);
    grid_box.bake(
        {},
        {},
        {pipe_design::geom::AabbObstacle{Vec3{8.f, 8.f, 0.f}, Vec3{12.f, 12.f, 1.f}}});
    EXPECT_TRUE(grid_box.aabbs().size() == 1);
    EXPECT_TRUE(!grid_box.is_free(CellCoord{10, 10, 0}, 0.1f));
    EXPECT_TRUE(grid_box.is_free(CellCoord{0, 0, 0}, 0.1f));
}

void test_analytic_point_beats_cell_sample() {
    // Small sphere near the corner of cell (1,1,0). Cell-center clearance stays
    // large enough for r=0.2, but a world sample next to the sphere is blocked.
    StaticClearanceGrid grid(10, 10, 1, 1.f, 3.f);
    grid.bake({SphereObstacle{Vec3{1.9f, 1.9f, 0.5f}, 0.1f}});

    const CellCoord cell{1, 1, 0};
    const Vec3 sample{1.85f, 1.85f, 0.5f};
    const float r = 0.2f;

    EXPECT_TRUE(grid.is_free(cell, r));
    EXPECT_TRUE(!grid.is_free(sample, r));

    CompositeGrid composite(grid);
    EXPECT_TRUE(composite.is_free(cell, r));
    EXPECT_TRUE(!composite.is_free(sample, r));
}

void test_is_move_free_corner_cut() {
    StaticClearanceGrid grid(10, 10, 1, 1.f, 3.f);
    // Block face-neighbor (2,1,0); diagonal A→B would cut that corner.
    grid.bake({SphereObstacle{Vec3{2.5f, 1.5f, 0.5f}, 0.4f}});

    const CellCoord from{1, 1, 0};
    const CellCoord to{2, 2, 0};
    const float r = 0.5f;

    EXPECT_TRUE(grid.is_free(from, r));
    EXPECT_TRUE(grid.is_free(to, r));
    EXPECT_TRUE(!grid.is_free(CellCoord{2, 1, 0}, r));
    EXPECT_TRUE(!grid.is_move_free(from, to, r));
    EXPECT_TRUE(grid.is_move_free(from, CellCoord{1, 2, 0}, r));

    // Longer step falls back to analytic segment check.
    EXPECT_TRUE(!grid.is_move_free(CellCoord{0, 1, 0}, CellCoord{4, 1, 0}, r));
    EXPECT_TRUE(grid.is_move_free(CellCoord{0, 3, 0}, CellCoord{4, 3, 0}, r));
}

void test_is_move_free_continuous() {
    StaticClearanceGrid grid(20, 20, 1, 1.f, 3.f);
    grid.bake({SphereObstacle{Vec3{10.5f, 10.5f, 0.5f}, 1.f}});

    const Vec3 a{5.5f, 10.5f, 0.5f};
    const Vec3 b{15.5f, 10.5f, 0.5f};
    const Vec3 c{5.5f, 5.5f, 0.5f};
    const float r = 0.5f;

    EXPECT_TRUE(grid.is_free(a, r));
    EXPECT_TRUE(grid.is_free(b, r));
    EXPECT_TRUE(!grid.is_move_free(a, b, r));
    EXPECT_TRUE(grid.is_move_free(a, c, r));
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

void test_composite_is_move_free() {
    StaticClearanceGrid static_grid(10, 10, 1, 1.f, 3.f);
    // No static obstacles; chain blocks the corner cell of a diagonal step.
    auto root = ObstacleChain::make_root();
    auto chain = root->extend(SphereObstacle{Vec3{2.5f, 1.5f, 0.5f}, 0.4f});
    CompositeGrid grid(static_grid, chain);

    const CellCoord from{1, 1, 0};
    const CellCoord to{2, 2, 0};
    const float r = 0.5f;

    EXPECT_TRUE(grid.is_free(from, r));
    EXPECT_TRUE(grid.is_free(to, r));
    EXPECT_TRUE(!grid.is_free(CellCoord{2, 1, 0}, r));
    EXPECT_TRUE(!grid.is_move_free(from, to, r));
    EXPECT_TRUE(grid.is_move_free(from, CellCoord{1, 2, 0}, r));

    EXPECT_TRUE(!grid.is_move_free(grid.cell_center(from), grid.cell_center(to), r));

    // Longer step falls back to analytic segment check.
    EXPECT_TRUE(!grid.is_move_free(CellCoord{0, 1, 0}, CellCoord{4, 1, 0}, r));
    EXPECT_TRUE(grid.is_move_free(CellCoord{0, 3, 0}, CellCoord{4, 3, 0}, r));
}

int main() {
    test_segment_capsule_signed_clearance();
    test_obstacle_chain_sharing();
    test_static_clearance_bake();
    test_analytic_point_beats_cell_sample();
    test_is_move_free_corner_cut();
    test_is_move_free_continuous();
    test_composite_grid();
    test_composite_is_move_free();

    if (g_failures == 0) {
        std::cout << "All grid tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " assertion(s) failed.\n";
    return 1;
}
