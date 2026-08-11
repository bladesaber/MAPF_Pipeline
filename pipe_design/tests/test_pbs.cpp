#include "conflict/ConflictAnalyzer.h"
#include "geom/Types.h"
#include "grid/CompositeGrid.h"
#include "grid/StaticClearanceGrid.h"
#include "heuristic/Geometric.h"
#include "planner/high/PBS.h"
#include "planner/low/AStar.h"
#include "planner/motion/SixConnected.h"
#include "planner/multi/PartialOrderSequencer.h"
#include "planner/open/LazyBinaryHeap.h"
#include "topology/Types.h"

#include <iostream>
#include <limits>
#include <string>
#include <vector>

using pipe_design::conflict::Conflict;
using pipe_design::conflict::ConflictAnalyzer;
using pipe_design::conflict::ConflictCluster;
using pipe_design::geom::Vec3;
using pipe_design::grid::CompositeGrid;
using pipe_design::grid::StaticClearanceGrid;
using pipe_design::heuristic::Manhattan;
using pipe_design::planner::AStar;
using pipe_design::planner::LazyBinaryHeap;
using pipe_design::planner::PBS;
using pipe_design::planner::PartialOrderSequencer;
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

class MaxDeficitAnalyzer : public ConflictAnalyzer {
public:
    void score(Conflict& conflict) const override { conflict.score = conflict.deficit; }

    void score(ConflictCluster& cluster) const override {
        cluster.score = cluster.max_deficit();
    }

    const Conflict* select_conflict() const override {
        const Conflict* best = nullptr;
        float best_score = -std::numeric_limits<float>::infinity();
        for (const Conflict& c : conflicts_) {
            if (best == nullptr || c.score > best_score) {
                best = &c;
                best_score = c.score;
            }
        }
        return best;
    }
};

static GroupTopologyPlan make_pair_group(
    const std::string& gid,
    Vec3 a,
    Vec3 b,
    float radius = 0.25f) {
    GroupTopologyPlan plan(gid);
    plan.terminals = {
        Terminal{"s" + gid, a, radius},
        Terminal{"g" + gid, b, radius},
    };
    plan.edges = {
        TopologyEdge{"e0", "s" + gid, "g" + gid, radius, 0, EdgeKind::Spine},
    };
    plan.grow_order = {"e0"};
    return plan;
}

static void test_parallel_no_conflict() {
    StaticClearanceGrid sg(12, 6, 4, 1.f, 2.f);
    sg.clear();
    CompositeGrid grid(sg);

    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;
    AStar<SearchNode> low;
    PartialOrderSequencer sequencer;
    MaxDeficitAnalyzer analyzer;

    GlobalTopologyPlan plans;
    plans.groups.push_back(make_pair_group("A", Vec3{0.5f, 1.5f, 0.5f}, Vec3{10.5f, 1.5f, 0.5f}));
    plans.groups.push_back(make_pair_group("B", Vec3{0.5f, 4.5f, 0.5f}, Vec3{10.5f, 4.5f, 0.5f}));

    PBS pbs;
    pbs.set_search_mode(pipe_design::planner::PBSSearchMode::DepthFirst);
    pbs.reset(plans, grid, low, h, open, actions, sequencer, analyzer);
    EXPECT_TRUE(pbs.status() == SearchStatus::Running);
    EXPECT_TRUE(pbs.search() == SearchStatus::Success);
    EXPECT_TRUE(pbs.solution().size() == 2);
    for (const auto& t : pbs.solution()) {
        EXPECT_TRUE(t.succeeded());
    }
}

static void test_crossing_resolves() {
    StaticClearanceGrid sg(11, 11, 5, 1.f, 2.f);
    sg.clear();
    CompositeGrid grid(sg);

    Manhattan h(1.0);
    SixConnected actions(1.0);
    LazyBinaryHeap open;
    AStar<SearchNode> low;
    PartialOrderSequencer sequencer;
    MaxDeficitAnalyzer analyzer;

    GlobalTopologyPlan plans;
    plans.groups.push_back(make_pair_group("A", Vec3{0.5f, 5.5f, 0.5f}, Vec3{10.5f, 5.5f, 0.5f}));
    plans.groups.push_back(make_pair_group("B", Vec3{5.5f, 0.5f, 0.5f}, Vec3{5.5f, 10.5f, 0.5f}));

    PBS pbs;
    pbs.set_search_mode(pipe_design::planner::PBSSearchMode::DepthFirst);
    pbs.reset(plans, grid, low, h, open, actions, sequencer, analyzer);
    EXPECT_TRUE(pbs.status() == SearchStatus::Running);
    const SearchStatus st = pbs.search(/*max_expansions=*/64);
    EXPECT_TRUE(st == SearchStatus::Success);
    if (st == SearchStatus::Success) {
        EXPECT_TRUE(pbs.solution().size() == 2);
        for (const auto& t : pbs.solution()) {
            EXPECT_TRUE(t.succeeded());
        }
        EXPECT_TRUE(pbs.expansions() >= 1);
    }
}

int main() {
    test_parallel_no_conflict();
    test_crossing_resolves();
    if (g_failures == 0) {
        std::cout << "ok\n";
        return 0;
    }
    std::cerr << g_failures << " failure(s)\n";
    return 1;
}
