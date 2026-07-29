#include "geom/Heading.h"
#include "planner/low/SearchNode.h"

#include <cmath>
#include <functional>
#include <iostream>

using pipe_design::geom::Heading;
using pipe_design::geom::Vec3;
using pipe_design::grid::CellCoord;
using pipe_design::planner::CellStateKey;
using pipe_design::planner::DirectedSearchNode;
using pipe_design::planner::DirectedStateKey;
using pipe_design::planner::SearchNode;
using pipe_design::planner::heading_between;

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

static void test_from_vector_normalizes() {
    EXPECT_TRUE(Heading{}.is_none());
    const Heading x = Heading::from_vector(Vec3{2.f, 0.f, 0.f});
    EXPECT_NEAR(x.forward().x, 1.f, 1e-5f);
    EXPECT_NEAR(x.forward().y, 0.f, 1e-5f);
    EXPECT_NEAR(x.forward().norm(), 1.f, 1e-5f);
    EXPECT_TRUE(x.nearly_equal(Heading::from_grid_step(1, 0, 0)));
    EXPECT_TRUE(x.is_opposite(Heading::from_grid_step(-1, 0, 0)));
}

static void test_yaw_pitch_roundtrip_bins() {
    const Heading h = Heading::from_yaw_pitch(0.f, 0.f);  // +X
    EXPECT_TRUE(h.yaw_bin(8) == 0);
    EXPECT_NEAR(h.forward().norm(), 1.f, 1e-5f);

    const Heading back = Heading::from_bin(0, 8);
    EXPECT_NEAR(back.forward().x, 1.f, 0.15f);
    EXPECT_TRUE(std::fabs(back.forward().z) < 0.2f);

    const int linear = h.bin(16, 1);
    const Heading again = Heading::from_linear_bin(linear, 16, 1);
    EXPECT_TRUE(h.alignment(again) > 0.9f);
}

static void test_heading_between_cells() {
    const Heading step = heading_between(CellCoord{0, 0, 0}, CellCoord{1, 1, 0});
    EXPECT_NEAR(step.forward().x, step.forward().y, 1e-5f);
    EXPECT_NEAR(step.forward().norm(), 1.f, 1e-5f);
}

static void test_directed_node_helpers() {
    DirectedSearchNode n(CellCoord{1, 0, 0});
    n.set_arrival(Heading::from_grid_step(1, 0, 0), /*yaw_bins=*/8);
    EXPECT_TRUE(n.heading_bin != DirectedStateKey::kNoneHeadingBin);
    EXPECT_TRUE(n.is_straight(Heading::from_grid_step(1, 0, 0)));
    EXPECT_TRUE(n.next_turns_streak(Heading::from_grid_step(0, 1, 0)) == 1);
    EXPECT_NEAR(n.heading_alignment(Heading::from_grid_step(1, 0, 0)), 1.f, 1e-5f);
}

static void test_state_keys() {
    SearchNode a(CellCoord{1, 2, 3});
    SearchNode b(CellCoord{1, 2, 3});
    EXPECT_TRUE(a.key() == b.key());
    EXPECT_TRUE(a.key() == SearchNode::make_key(CellCoord{1, 2, 3}));

    const Heading hx = Heading::from_grid_step(1, 0, 0);
    const Heading hy = Heading::from_grid_step(0, 1, 0);
    const auto kx = DirectedSearchNode::make_key(CellCoord{0, 0, 0}, hx, 8);
    const auto ky = DirectedSearchNode::make_key(CellCoord{0, 0, 0}, hy, 8);
    EXPECT_TRUE(kx.coord == ky.coord);
    EXPECT_TRUE(kx.heading_bin != ky.heading_bin);
    EXPECT_TRUE(kx != ky);

    DirectedSearchNode start(CellCoord{0, 0, 0});
    EXPECT_TRUE(start.key().heading_bin == DirectedStateKey::kNoneHeadingBin);

    EXPECT_TRUE(std::hash<CellStateKey>{}(a.key()) == std::hash<CellStateKey>{}(b.key()));
    EXPECT_TRUE(std::hash<DirectedStateKey>{}(kx) != std::hash<DirectedStateKey>{}(ky));
}

int main() {
    test_from_vector_normalizes();
    test_yaw_pitch_roundtrip_bins();
    test_heading_between_cells();
    test_directed_node_helpers();
    test_state_keys();

    if (g_failures != 0) {
        std::cerr << g_failures << " assertion(s) failed.\n";
        return 1;
    }
    std::cout << "All heading tests passed.\n";
    return 0;
}
