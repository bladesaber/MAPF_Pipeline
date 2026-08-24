#include "benchmark/pillar_field/PillarFieldGenerator.h"
#include "planner/low/Path.h"

#include <iostream>

using pipe_design::benchmark::Instance;
using pipe_design::benchmark::GroupCorridors;
using pipe_design::benchmark::pillar_field::PillarFieldGenerator;
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

static void test_fills_toward_density() {
    Instance inst = make_corridor_instance();

    pipe_design::benchmark::pillar_field::GenParams op;
    op.density_target = 0.35f;
    op.s_min = 2;
    op.candidates_per_round = 24;
    op.max_rounds = 64;
    op.corridor_margin = 1;
    op.seed = 42;

    PillarFieldGenerator gen(op);
    CHECK(gen.generate(inst));
    CHECK(inst.obstacle_params.density_target == op.density_target);
    CHECK(inst.obstacle_params.seed == op.seed);
    CHECK(inst.obstacle_params.corridor_margin == op.corridor_margin);
    CHECK(gen.eligible0() > 0);
    CHECK(!inst.aabbs.empty());
    CHECK(gen.density() + 1e-3f >= op.density_target ||
          gen.density() > 0.2f);
    for (const auto& box : inst.aabbs) {
        CHECK(box.mx.x > box.mn.x);
        CHECK(box.mx.y > box.mn.y);
        CHECK(box.mx.z > box.mn.z);
    }
}

static void test_zero_density() {
    Instance inst = make_corridor_instance();
    pipe_design::benchmark::pillar_field::GenParams op;
    op.density_target = 0.f;
    op.seed = 1;
    PillarFieldGenerator gen(op);
    CHECK(gen.generate(inst));
    CHECK(inst.aabbs.empty());
    CHECK(gen.density() == 0.f);
}

static void test_isometric_fills() {
    Instance inst = make_corridor_instance();
    pipe_design::benchmark::pillar_field::GenParams op;
    op.density_target = 0.25f;
    op.s_min = 1;
    op.candidates_per_round = 24;
    op.max_rounds = 64;
    op.corridor_margin = 1;
    op.seed = 7;
    op.isometric = true;
    PillarFieldGenerator gen(op);
    CHECK(gen.generate(inst));
    CHECK(!inst.aabbs.empty());
}

int main() {
    test_fills_toward_density();
    test_zero_density();
    test_isometric_fills();
    if (g_fail) {
        std::cerr << g_fail << " check(s) failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
