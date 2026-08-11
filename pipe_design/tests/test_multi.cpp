#include "geom/Types.h"
#include "grid/CompositeGrid.h"
#include "grid/ObstacleChain.h"
#include "grid/StaticClearanceGrid.h"
#include "heuristic/Geometric.h"
#include "planner/low/AStar.h"
#include "planner/motion/SixConnected.h"
#include "planner/multi/GroupSession.h"
#include "planner/multi/PartialOrderSequencer.h"
#include "planner/multi/PrioritySequencer.h"
#include "planner/open/LazyBinaryHeap.h"
#include "topology/Types.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using pipe_design::geom::SphereObstacle;
using pipe_design::geom::Vec3;
using pipe_design::grid::CompositeGrid;
using pipe_design::grid::ObstacleChain;
using pipe_design::grid::StaticClearanceGrid;
using pipe_design::heuristic::Manhattan;
using pipe_design::planner::AStar;
using pipe_design::planner::GroupSession;
using pipe_design::planner::LazyBinaryHeap;
using pipe_design::planner::PartialOrderPast;
using pipe_design::planner::PartialOrderSequencer;
using pipe_design::planner::PriorityConstraint;
using pipe_design::planner::PrioritySequencer;
using pipe_design::planner::SearchNode;
using pipe_design::planner::SearchStatus;
using pipe_design::planner::motion::SixConnected;
using pipe_design::topology::EdgeKind;
using pipe_design::topology::GlobalTopologyPlan;
using pipe_design::topology::GroupTopologyPlan;
using pipe_design::topology::Terminal;
using pipe_design::topology::TopologyEdge;

static int g_failures = 0;

#define EXPECT_TRUE(cond)                                                          \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " #cond "\n"; \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

static GroupTopologyPlan make_line_group(const std::string& gid, float z) {
    GroupTopologyPlan plan(gid);
    plan.terminals = {
        Terminal{"a" + gid, Vec3{0.5f, 0.5f, z}, 0.2f},
        Terminal{"b" + gid, Vec3{4.5f, 0.5f, z}, 0.2f},
        Terminal{"c" + gid, Vec3{8.5f, 0.5f, z}, 0.2f},
    };
    plan.edges = {
        TopologyEdge{"e0", "a" + gid, "b" + gid, 0.2f, 0, EdgeKind::Spine},
        TopologyEdge{"e1", "b" + gid, "c" + gid, 0.2f, 0, EdgeKind::Branch},
    };
    plan.grow_order = {"e0", "e1"};
    return plan;
}

/// Snapshot of entry CompositeGrid identity that must not change across multi search.
struct GridSnapshot {
    const StaticClearanceGrid* static_ptr = nullptr;
    std::shared_ptr<const ObstacleChain> chain;
    std::size_t chain_total = 0;
    float sample_clearance = 0.f;
};

static GridSnapshot snapshot_grid(const CompositeGrid& grid) {
    GridSnapshot s;
    s.static_ptr = &grid.static_grid();
    s.chain = grid.chain();
    s.chain_total = s.chain ? s.chain->total_count() : 0;
    s.sample_clearance = grid.clearance(pipe_design::grid::CellCoord{1, 1, 0});
    return s;
}

static void expect_grid_unchanged(const CompositeGrid& grid, const GridSnapshot& before) {
    EXPECT_TRUE(&grid.static_grid() == before.static_ptr);
    EXPECT_TRUE(grid.chain() == before.chain);  // same shared_ptr instance
    const std::size_t after_total = grid.chain() ? grid.chain()->total_count() : 0;
    EXPECT_TRUE(after_total == before.chain_total);
    EXPECT_TRUE(grid.clearance(pipe_design::grid::CellCoord{1, 1, 0}) == before.sample_clearance);
}

static void test_group_session_preserves_grid() {
    StaticClearanceGrid sg(10, 3, 3, 1.f, 2.f);
    sg.clear();
    auto base = ObstacleChain::make_root()->extend(SphereObstacle{Vec3{9.5f, 2.5f, 0.5f}, 0.3f});
    CompositeGrid grid(sg, base);
    const GridSnapshot before = snapshot_grid(grid);

    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;
    AStar<SearchNode> low;

    GroupTopologyPlan plan = make_line_group("G", 0.5f);
    GroupSession session;
    session.reset(plan, grid, low, h, open, actions);
    EXPECT_TRUE(session.search() == SearchStatus::Success);
    EXPECT_TRUE(session.tasks().size() == 2);

    expect_grid_unchanged(grid, before);
    EXPECT_TRUE(before.chain_total == 1);
}

static void test_priority_sequencer_preserves_grid() {
    StaticClearanceGrid sg(10, 5, 3, 1.f, 2.f);
    sg.clear();
    // Non-empty entry chain: must survive multi-group PP-style routing.
    auto base = ObstacleChain::make_root()->extend(SphereObstacle{Vec3{9.5f, 4.5f, 2.5f}, 0.25f});
    CompositeGrid grid(sg, base);
    const GridSnapshot before = snapshot_grid(grid);
    EXPECT_TRUE(before.chain != nullptr);
    EXPECT_TRUE(before.chain_total == 1);

    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;
    AStar<SearchNode> low;

    GlobalTopologyPlan global;
    global.groups.push_back(make_line_group("A", 0.5f));
    global.groups.push_back(make_line_group("B", 2.5f));

    PrioritySequencer seq;
    seq.reset(global, grid, low, h, open, actions);
    EXPECT_TRUE(seq.search() == SearchStatus::Success);
    EXPECT_TRUE(seq.tasks().size() == 4);
    for (const auto& t : seq.tasks()) {
        EXPECT_TRUE(t.succeeded());
    }

    expect_grid_unchanged(grid, before);
}

static void test_partial_order_topo_and_route() {
    // Constraints: B before A (B higher). Seed enumerate order is A then B.
    const std::vector<std::string> seed = {"A:e0", "A:e1", "B:e0", "B:e1"};
    const std::vector<PriorityConstraint> cons = {
        {"B:e0", "A:e0"},
        {"B:e1", "A:e0"},
        {"A:e0", "A:e1"},
        {"B:e0", "B:e1"},
    };
    const auto order = PartialOrderSequencer::topological_sort(seed, cons);
    EXPECT_TRUE(order.size() == 4);
    // B block before A block; intra order preserved.
    EXPECT_TRUE(order[0] == "B:e0");
    EXPECT_TRUE(order[1] == "B:e1");
    EXPECT_TRUE(order[2] == "A:e0");
    EXPECT_TRUE(order[3] == "A:e1");

    StaticClearanceGrid sg(10, 5, 3, 1.f, 2.f);
    sg.clear();
    CompositeGrid grid(sg);

    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;
    AStar<SearchNode> low;

    GlobalTopologyPlan global;
    global.groups.push_back(make_line_group("A", 0.5f));
    global.groups.push_back(make_line_group("B", 2.5f));

    PartialOrderSequencer seq;
    seq.reset(global, cons, grid, low, h, open, actions);
    EXPECT_TRUE(seq.status() == SearchStatus::Running);
    EXPECT_TRUE(seq.order() == order);

    const auto higher = seq.higher_of("A:e1");
    EXPECT_TRUE(higher.count("A:e0") != 0);
    EXPECT_TRUE(higher.count("B:e0") != 0);

    EXPECT_TRUE(seq.search() == SearchStatus::Success);
    EXPECT_TRUE(seq.tasks().size() == 4);
    for (const auto& t : seq.tasks()) {
        EXPECT_TRUE(t.succeeded());
    }

    const PartialOrderPast past = seq.snapshot();
    EXPECT_TRUE(past.by_id.size() == 4);
    const auto& a1 = past.by_id.at("A:e1");
    EXPECT_TRUE(a1.foreign_higher.count("B:e0") != 0);
    EXPECT_TRUE(a1.foreign_higher.count("B:e1") != 0);
    EXPECT_TRUE(a1.foreign_higher.count("A:e0") == 0);
    EXPECT_TRUE(a1.same_group_prev.count("A:e0") != 0);
    EXPECT_TRUE(a1.task.succeeded());
    EXPECT_TRUE(seq.replanned_ids().size() == 4);

    PartialOrderSequencer seq2;
    seq2.reset(global, cons, grid, low, h, open, actions, past);
    EXPECT_TRUE(seq2.can_reuse("B:e0", past));
    EXPECT_TRUE(seq2.can_reuse("A:e1", past));
    EXPECT_TRUE(seq2.search() == SearchStatus::Success);
    EXPECT_TRUE(seq2.replanned_ids().empty());
    for (const auto& t : seq2.tasks()) {
        EXPECT_TRUE(t.succeeded());
    }

    // Cycle → Failure
    PartialOrderSequencer bad;
    bad.reset(global, {{"A:e0", "B:e0"}, {"B:e0", "A:e0"}}, grid, low, h, open, actions);
    EXPECT_TRUE(bad.status() == SearchStatus::Failure);
}

int main() {
    test_group_session_preserves_grid();
    test_priority_sequencer_preserves_grid();
    test_partial_order_topo_and_route();
    if (g_failures == 0) {
        std::cout << "ok\n";
        return 0;
    }
    std::cerr << g_failures << " failure(s)\n";
    return 1;
}
