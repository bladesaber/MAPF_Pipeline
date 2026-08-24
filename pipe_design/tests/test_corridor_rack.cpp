#include "benchmark/corridor_rack/CorridorRackGenerator.h"
#include "planner/low/Path.h"

#include <iostream>

using pipe_design::benchmark::GroupCorridors;
using pipe_design::benchmark::Instance;
using pipe_design::benchmark::corridor_rack::CorridorRackGenerator;
using pipe_design::geom::Vec3;
using pipe_design::grid::CellCoord;
using pipe_design::planner::Path;

static int g_fail = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " #cond    \
                      << "\n";                                                 \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

static Instance make_corridor_instance() {
    Instance inst;
    inst.params.size_x = inst.params.size_y = inst.params.size_z = 16;
    inst.params.radius = 0.4f;
    inst.params.seed = 1;
    inst.success = true;

    GroupCorridors g;
    g.id = "g0";
    g.terminals.push_back({"t0", Vec3{0.5f, 8.5f, 8.5f}, 0.4f});
    g.terminals.push_back({"t1", Vec3{15.5f, 8.5f, 8.5f}, 0.4f});
    Path path;
    path.cost = 15.0;
    for (int x = 0; x < 16; ++x) {
        path.cells.push_back(CellCoord{x, 8, 8});
    }
    g.paths.push_back(path);
    inst.groups.push_back(g);
    return inst;
}

static void test_fills_complement() {
    Instance inst = make_corridor_instance();
    pipe_design::benchmark::corridor_rack::GenParams op;
    op.dilate_steps = 1;
    op.seed = 1;

    CorridorRackGenerator gen(op);
    CHECK(gen.generate(inst));
    CHECK(gen.dilate_steps() == 1);
    CHECK(gen.eligible0() > 0);
    CHECK(!inst.aabbs.empty());
    CHECK(gen.density() > 0.f);
    CHECK(inst.obstacle_params.density_target == gen.density());
    for (const auto& box : inst.aabbs) {
        CHECK(box.mx.x > box.mn.x);
        CHECK(box.mx.y > box.mn.y);
        CHECK(box.mx.z > box.mn.z);
    }
}

static void test_larger_dilate_lowers_density() {
    Instance a = make_corridor_instance();
    Instance b = make_corridor_instance();
    pipe_design::benchmark::corridor_rack::GenParams pa;
    pipe_design::benchmark::corridor_rack::GenParams pb;
    pa.dilate_steps = 1;
    pb.dilate_steps = 3;
    CorridorRackGenerator ga(pa);
    CorridorRackGenerator gb(pb);
    CHECK(ga.generate(a));
    CHECK(gb.generate(b));
    CHECK(gb.density() < ga.density());
    CHECK(gb.eligible0() < ga.eligible0());
}

static void test_full_dilate_empty_obstacles() {
    Instance inst = make_corridor_instance();
    pipe_design::benchmark::corridor_rack::GenParams op;
    // Large enough to cover the whole 16^3 domain from the centerline.
    op.dilate_steps = 32;
    CorridorRackGenerator gen(op);
    CHECK(gen.generate(inst));
    CHECK(gen.eligible0() == 0);
    CHECK(inst.aabbs.empty());
    CHECK(gen.density() == 0.f);
}

static void test_preserves_shell_aabbs() {
    Instance inst = make_corridor_instance();
    inst.aabbs.push_back(
        {Vec3{0.f, 0.f, 0.f}, Vec3{16.f, 1.f, 16.f}});
    pipe_design::benchmark::corridor_rack::GenParams op;
    op.dilate_steps = 1;
    CorridorRackGenerator gen(op);
    Instance bare = make_corridor_instance();
    CorridorRackGenerator gen_bare(op);
    CHECK(gen_bare.generate(bare));
    CHECK(gen.generate(inst));
    CHECK(inst.aabbs.size() >= 2);
    CHECK(inst.aabbs[0].mx.y == 1.f);
    CHECK(gen.eligible0() < gen_bare.eligible0());
    CHECK(gen.density() == gen_bare.density());
}

int main() {
    test_fills_complement();
    test_larger_dilate_lowers_density();
    test_full_dilate_empty_obstacles();
    test_preserves_shell_aabbs();
    if (g_fail) {
        std::cerr << g_fail << " check(s) failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
