#include "geom/Heading.h"
#include "grid/Types.h"
#include "planner/motion/Motion.h"

#include <cmath>
#include <iostream>
#include <vector>

using pipe_design::geom::Heading;
using pipe_design::grid::CellCoord;
using pipe_design::planner::motion::EightConnected;
using pipe_design::planner::motion::SixConnected;
using pipe_design::planner::motion::Successor;
using pipe_design::planner::motion::TwentySixConnected;

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

static void test_six_count_and_heading() {
    SixConnected m(1.0);
    std::vector<Successor> out;
    m.successors(CellCoord{0, 0, 0}, out);
    EXPECT_TRUE(out.size() == 6);
    for (const Successor& s : out) {
        EXPECT_NEAR(s.cost, 1.0, 1e-12);
        EXPECT_TRUE(!s.heading.is_none());
        EXPECT_NEAR(s.heading.forward().norm(), 1.f, 1e-5f);
    }
}

static void test_six_turn_penalty() {
    SixConnected m(1.0);
    m.set_turn_weight(2.0);

    const Heading east = Heading::from_grid_step(1, 0, 0);
    std::vector<Successor> out;
    m.successors(CellCoord{0, 0, 0}, east, out);

    bool found_straight = false;
    bool found_left = false;
    for (const Successor& s : out) {
        if (s.heading.nearly_equal(east)) {
            found_straight = true;
            EXPECT_NEAR(s.cost, 1.0, 1e-12);
        } else if (s.heading.nearly_equal(Heading::from_grid_step(0, 1, 0))) {
            found_left = true;
            EXPECT_NEAR(s.cost, 2.0, 1e-12);  // 1 + 2 * 0.5 (alignment 0)
        }
    }
    EXPECT_TRUE(found_straight);
    EXPECT_TRUE(found_left);
}

static void test_eight_costs() {
    EightConnected m(1.0);
    std::vector<Successor> out;
    m.successors(CellCoord{3, 4, 7}, out);
    EXPECT_TRUE(out.size() == 8);

    int faces = 0;
    int diags = 0;
    for (const Successor& s : out) {
        EXPECT_TRUE(s.cell.z == 7);
        const CellCoord d = s.cell - CellCoord{3, 4, 7};
        const int axes = (d.x != 0) + (d.y != 0) + (d.z != 0);
        EXPECT_TRUE(axes == 1 || axes == 2);
        if (axes == 1) {
            ++faces;
            EXPECT_NEAR(s.cost, 1.0, 1e-12);
        } else {
            ++diags;
            EXPECT_NEAR(s.cost, std::sqrt(2.0), 1e-12);
        }
    }
    EXPECT_TRUE(faces == 4);
    EXPECT_TRUE(diags == 4);
}

static void test_twenty_six_costs() {
    TwentySixConnected m(2.0);
    std::vector<Successor> out;
    m.successors(CellCoord{5, 5, 5}, out);
    EXPECT_TRUE(out.size() == 26);

    int faces = 0;
    int edges = 0;
    int corners = 0;
    for (const Successor& s : out) {
        const CellCoord d = s.cell - CellCoord{5, 5, 5};
        const int axes = (d.x != 0) + (d.y != 0) + (d.z != 0);
        EXPECT_TRUE(axes >= 1 && axes <= 3);
        if (axes == 1) {
            ++faces;
            EXPECT_NEAR(s.cost, 2.0, 1e-12);
        } else if (axes == 2) {
            ++edges;
            EXPECT_NEAR(s.cost, 2.0 * std::sqrt(2.0), 1e-12);
        } else {
            ++corners;
            EXPECT_NEAR(s.cost, 2.0 * std::sqrt(3.0), 1e-12);
        }
    }
    EXPECT_TRUE(faces == 6);
    EXPECT_TRUE(edges == 12);
    EXPECT_TRUE(corners == 8);
}

int main() {
    test_six_count_and_heading();
    test_six_turn_penalty();
    test_eight_costs();
    test_twenty_six_costs();

    if (g_failures != 0) {
        std::cerr << g_failures << " assertion(s) failed.\n";
        return 1;
    }
    std::cout << "All motion tests passed.\n";
    return 0;
}
