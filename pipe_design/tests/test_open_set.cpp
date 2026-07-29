#include "planner/low/SearchNode.h"
#include "planner/open/BinaryHeap.h"
#include "planner/open/BucketHeap.h"
#include "planner/open/LazyBinaryHeap.h"
#include "planner/open/PairingHeap.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using pipe_design::grid::CellCoord;
using pipe_design::planner::BinaryHeap;
using pipe_design::planner::BucketHeap;
using pipe_design::planner::LazyBinaryHeap;
using pipe_design::planner::NodeStatus;
using pipe_design::planner::OpenSet;
using pipe_design::planner::PairingHeap;
using pipe_design::planner::SearchNode;

static int g_failures = 0;

#define EXPECT_TRUE(cond)                                                          \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " #cond "\n"; \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

static void expect_pop_order(OpenSet& heap, const std::vector<SearchNode*>& expected) {
    for (SearchNode* node : expected) {
        EXPECT_TRUE(!heap.empty());
        SearchNode* got = heap.pop();
        EXPECT_TRUE(got == node);
    }
    EXPECT_TRUE(heap.empty());
}

static void test_binary_heap_order_and_decrease() {
    BinaryHeap heap;
    SearchNode a(CellCoord{0, 0, 0});
    SearchNode b(CellCoord{1, 0, 0});
    SearchNode c(CellCoord{2, 0, 0});
    a.f = 5.0;
    a.g = 1.0;
    b.f = 3.0;
    b.g = 1.0;
    c.f = 4.0;
    c.g = 1.0;

    heap.push(&a);
    heap.push(&b);
    heap.push(&c);
    EXPECT_TRUE(heap.size() == 3);
    EXPECT_TRUE(heap.contains(&b));

    c.f = 1.0;
    EXPECT_TRUE(heap.decrease_key(&c));
    expect_pop_order(heap, {&c, &b, &a});
}

static void test_binary_heap_tie_break_larger_g() {
    BinaryHeap heap;
    SearchNode a(CellCoord{0, 0, 0});
    SearchNode b(CellCoord{1, 0, 0});
    a.f = 5.0;
    a.g = 1.0;
    b.f = 5.0;
    b.g = 4.0;
    heap.push(&a);
    heap.push(&b);
    EXPECT_TRUE(heap.pop() == &b);
    EXPECT_TRUE(heap.pop() == &a);
}

static void test_lazy_heap_stale_entries() {
    LazyBinaryHeap heap;
    SearchNode a(CellCoord{0, 0, 0});
    a.status = NodeStatus::Open;
    a.f = 10.0;
    a.g = 0.0;
    heap.push(&a);

    a.f = 2.0;
    a.g = 1.0;
    heap.decrease_key(&a);
    heap.push(&a);

    SearchNode* best = heap.pop();
    EXPECT_TRUE(best == &a);
    EXPECT_TRUE(std::fabs(best->f - 2.0) < 1e-9);
    EXPECT_TRUE(heap.empty());
}

static void test_pairing_heap_decrease() {
    PairingHeap heap;
    SearchNode a(CellCoord{0, 0, 0});
    SearchNode b(CellCoord{1, 0, 0});
    SearchNode c(CellCoord{2, 0, 0});
    a.f = 9.0;
    b.f = 8.0;
    c.f = 7.0;
    a.g = b.g = c.g = 0.0;
    heap.push(&a);
    heap.push(&b);
    heap.push(&c);

    a.f = 1.0;
    EXPECT_TRUE(heap.decrease_key(&a));
    EXPECT_TRUE(heap.pop() == &a);
    EXPECT_TRUE(heap.pop() == &c);
    EXPECT_TRUE(heap.pop() == &b);
    EXPECT_TRUE(heap.empty());
}

static void test_pairing_many_pushes() {
    PairingHeap heap;
    std::vector<std::unique_ptr<SearchNode>> nodes;
    for (int i = 0; i < 50; ++i) {
        nodes.push_back(std::make_unique<SearchNode>(CellCoord{i, 0, 0}));
        nodes.back()->f = static_cast<double>(100 - i);
        nodes.back()->g = 0.0;
        heap.push(nodes.back().get());
    }
    double last = -1.0;
    while (!heap.empty()) {
        SearchNode* n = heap.pop();
        EXPECT_TRUE(n->f >= last);
        last = n->f;
    }
}

static void test_bucket_heap_order_and_decrease() {
    BucketHeap heap(/*delta=*/0.5, /*max_cost=*/100.0);
    SearchNode a(CellCoord{0, 0, 0});
    SearchNode b(CellCoord{1, 0, 0});
    SearchNode c(CellCoord{2, 0, 0});
    a.f = 5.0;
    a.g = 1.0;
    b.f = 3.0;
    b.g = 1.0;
    c.f = 4.0;
    c.g = 1.0;

    heap.push(&a);
    heap.push(&b);
    heap.push(&c);
    EXPECT_TRUE(heap.size() == 3);

    c.f = 1.0;
    EXPECT_TRUE(heap.decrease_key(&c));
    expect_pop_order(heap, {&c, &b, &a});
}

static void test_bucket_heap_same_bin_exact_order() {
    BucketHeap heap(/*delta=*/1.0, /*max_cost=*/10.0);
    SearchNode a(CellCoord{0, 0, 0});
    SearchNode b(CellCoord{1, 0, 0});
    a.f = 1.9;
    a.g = 0.0;
    b.f = 1.1;
    b.g = 0.0;
    heap.push(&a);
    heap.push(&b);
    EXPECT_TRUE(heap.pop() == &b);
    EXPECT_TRUE(heap.pop() == &a);
}

static void test_bucket_heap_throws_on_over_limit() {
    BucketHeap heap(/*delta=*/0.1, /*max_cost=*/5.0);
    SearchNode a(CellCoord{0, 0, 0});
    a.f = 5.1;
    a.g = 0.0;
    bool threw = false;
    try {
        heap.push(&a);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

// ---------------------------------------------------------------------------
// Speed benchmark
// ---------------------------------------------------------------------------

enum class DecreaseMode {
    TrueDecrease,  // Binary / Pairing / Bucket: update f then decrease_key
    LazyRepush,    // LazyBinary: update f then push again
};

struct BenchResult {
    std::string name;
    double ms = 0.0;
};

static uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void reset_nodes_for_bench(std::vector<SearchNode>& nodes, uint32_t seed) {
    uint32_t rng = seed;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        SearchNode& n = nodes[i];
        n.coord = CellCoord{static_cast<int>(i % 1000), static_cast<int>(i / 1000), 0};
        n.g = static_cast<double>(xorshift32(rng) % 1000) * 0.01;
        n.f = n.g + static_cast<double>(xorshift32(rng) % 5000) * 0.01;
        n.parent = nullptr;
        n.status = NodeStatus::Open;
    }
}

/// A*-like workload: push all → decrease ~25% keys → pop until empty.
static void run_open_set_workload(OpenSet& heap, std::vector<SearchNode>& nodes, DecreaseMode mode) {
    heap.clear();
    for (SearchNode& n : nodes) {
        n.status = NodeStatus::Open;
        heap.push(&n);
    }

    for (std::size_t i = 0; i < nodes.size(); i += 4) {
        SearchNode& n = nodes[i];
        if (n.status != NodeStatus::Open) {
            continue;
        }
        n.f = std::max(0.0, n.f * 0.5);
        n.g += 0.1;
        if (mode == DecreaseMode::TrueDecrease) {
            heap.decrease_key(&n);
        } else {
            heap.decrease_key(&n);
            heap.push(&n);
        }
    }

    while (!heap.empty()) {
        SearchNode* n = heap.pop();
        n->status = NodeStatus::Closed;
    }
}

static BenchResult benchmark_one(
    const std::string& name,
    OpenSet& heap,
    std::vector<SearchNode>& nodes,
    DecreaseMode mode,
    int rounds) {
    reset_nodes_for_bench(nodes, 1);
    run_open_set_workload(heap, nodes, mode);

    double sum_ms = 0.0;
    for (int r = 0; r < rounds; ++r) {
        reset_nodes_for_bench(nodes, static_cast<uint32_t>(1000 + r));
        const auto t0 = std::chrono::steady_clock::now();
        run_open_set_workload(heap, nodes, mode);
        const auto t1 = std::chrono::steady_clock::now();
        sum_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    BenchResult out;
    out.name = name;
    out.ms = sum_ms / static_cast<double>(rounds);
    return out;
}

static void benchmark_open_set_speed() {
    constexpr int kN = 50000;
    constexpr int kRounds = 8;

    std::vector<SearchNode> nodes(static_cast<std::size_t>(kN));

    BinaryHeap binary;
    LazyBinaryHeap lazy;
    PairingHeap pairing;
    BucketHeap bucket_fine(/*delta=*/0.1, /*max_cost=*/100.0);
    BucketHeap bucket_coarse(/*delta=*/1.0, /*max_cost=*/100.0);

    std::vector<BenchResult> results;
    results.push_back(benchmark_one("BinaryHeap", binary, nodes, DecreaseMode::TrueDecrease, kRounds));
    results.push_back(benchmark_one("LazyBinaryHeap", lazy, nodes, DecreaseMode::LazyRepush, kRounds));
    results.push_back(benchmark_one("PairingHeap", pairing, nodes, DecreaseMode::TrueDecrease, kRounds));
    results.push_back(
        benchmark_one("BucketHeap(delta=0.1)", bucket_fine, nodes, DecreaseMode::TrueDecrease, kRounds));
    results.push_back(
        benchmark_one("BucketHeap(delta=1.0)", bucket_coarse, nodes, DecreaseMode::TrueDecrease, kRounds));

    const double baseline = results.front().ms;
    std::cout << "\n=== Open-set speed benchmark ===\n";
    std::cout << "workload: push N + decrease ~N/4 + pop all\n";
    std::cout << "N=" << kN << "  rounds=" << kRounds << " (avg ms)\n";
    std::cout << std::left << std::setw(28) << "name" << std::right << std::setw(12) << "ms"
              << std::setw(12) << "vs Binary" << '\n';
    std::cout << std::string(52, '-') << '\n';
    std::cout << std::fixed << std::setprecision(3);
    for (const BenchResult& r : results) {
        const double rel = (baseline > 0.0) ? (r.ms / baseline) : 0.0;
        std::cout << std::left << std::setw(28) << r.name << std::right << std::setw(12) << r.ms
                  << std::setw(11) << rel << 'x' << '\n';
    }
    std::cout << "(smaller ms is faster; vs Binary < 1 means faster than BinaryHeap)\n";
}

int main() {
    test_binary_heap_order_and_decrease();
    test_binary_heap_tie_break_larger_g();
    test_lazy_heap_stale_entries();
    test_pairing_heap_decrease();
    test_pairing_many_pushes();
    test_bucket_heap_order_and_decrease();
    test_bucket_heap_same_bin_exact_order();
    test_bucket_heap_throws_on_over_limit();

    if (g_failures != 0) {
        std::cerr << g_failures << " assertion(s) failed.\n";
        return 1;
    }

    std::cout << "All open-set tests passed.\n";
    benchmark_open_set_speed();
    return 0;
}
