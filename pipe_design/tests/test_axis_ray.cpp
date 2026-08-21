#include "benchmark/axis_ray/AxisRayGenerator.h"

#include <iostream>

using pipe_design::benchmark::Instance;
using pipe_design::benchmark::axis_ray::AxisRayGenerator;
using pipe_design::benchmark::axis_ray::GenParams;

static int g_fail = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " #cond    \
                      << "\n";                                                 \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

static void test_seed_two_groups() {
    GenParams p;
    p.size_x = p.size_y = p.size_z = 24;
    p.radius = 0.4f;
    p.terminal_gap = 2;
    p.num_groups = 2;
    p.terminals_per_group = {2, 2};
    p.seed = 7;
    p.init_waypoints_num = 2;
    p.attach_waypoints_num = 2;
    p.min_path_size = 2;
    p.max_path_size = 12;
    p.path_gap = 1;
    p.branch_gap = 2;

    AxisRayGenerator gen(p);
    Instance out;
    const bool ok = gen.generate(out);
    CHECK(ok);
    CHECK(out.success);
    CHECK(out.groups.size() == 2);
    for (const auto& g : out.groups) {
        CHECK(g.terminals.size() == 2);
        CHECK(!g.paths.empty());
    }
}

static void test_attach() {
    GenParams p;
    p.size_x = p.size_y = p.size_z = 28;
    p.radius = 0.4f;
    p.terminal_gap = 2;
    p.num_groups = 1;
    p.terminals_per_group = {3};
    p.seed = 12;
    p.init_waypoints_num = 3;
    p.attach_waypoints_num = 2;
    p.min_path_size = 2;
    p.max_path_size = 10;
    p.path_gap = 1;
    p.branch_gap = 2;

    AxisRayGenerator gen(p);
    Instance out;
    const bool ok = gen.generate(out);
    CHECK(ok);
    CHECK(out.success);
    CHECK(out.groups.size() == 1);
    CHECK(out.groups[0].terminals.size() == 3);
}

static void test_attach_large_path_gap() {
    GenParams p;
    p.size_x = p.size_y = p.size_z = 36;
    p.radius = 0.4f;
    p.terminal_gap = 2;
    p.num_groups = 1;
    p.terminals_per_group = {3};
    p.seed = 19;
    p.init_waypoints_num = 2;
    p.attach_waypoints_num = 2;
    p.min_path_size = 2;
    p.max_path_size = 14;
    p.path_gap = 4;
    p.branch_gap = 5;
    p.max_attempts = 128;

    AxisRayGenerator gen(p);
    Instance out;
    const bool ok = gen.generate(out);
    CHECK(ok);
    CHECK(out.success);
    CHECK(out.groups[0].terminals.size() == 3);
}

int main() {
    test_seed_two_groups();
    test_attach();
    test_attach_large_path_gap();
    if (g_fail) {
        std::cerr << g_fail << " check(s) failed\n";
        return 1;
    }
    std::cout << "test_axis_ray OK\n";
    return 0;
}
