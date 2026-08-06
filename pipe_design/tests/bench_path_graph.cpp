// Bench: PathGraph vs Boost.Graph on a long chain (pipe-like), N >= 50000.
// Build: cmake --build . --target bench_path_graph && ./bench_path_graph [N]

#include "conflict/PathGraph.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

using BoostGraph = boost::adjacency_list<
    boost::vecS,
    boost::vecS,
    boost::undirectedS,
    boost::no_property,
    boost::property<boost::edge_weight_t, float>>;

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10))
                                   : 50000;
    if (n < 2) {
        std::fprintf(stderr, "N must be >= 2\n");
        return 1;
    }

    std::printf("=== PathGraph vs Boost (chain, N=%zu nodes, %zu edges) ===\n", n, n - 1);

    // --- PathGraph: build chain ---
    pipe_design::conflict::PathGraph pg(1e-4f);
    pg.set_dijkstra_cache_limit(0);  // no cache; fair single-query compare
    auto t0 = Clock::now();
    std::vector<pipe_design::conflict::NodeId> ids;
    ids.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        ids.push_back(pg.add_node(
            pipe_design::conflict::SampledPoint{{static_cast<float>(i), 0.f, 0.f}, 1.f}));
    }
    for (std::size_t i = 0; i + 1 < n; ++i) {
        pg.add_edge(ids[i], ids[i + 1], 1.f);
    }
    const double pg_build_ms = ms_since(t0);

    t0 = Clock::now();
    const float pg_len = pg.shortest_path_length(ids.front(), ids.back());
    const double pg_dijkstra_ms = ms_since(t0);

    t0 = Clock::now();
    const auto pg_path = pg.shortest_path(ids.front(), ids.back());
    const double pg_path_ms = ms_since(t0);

    // --- Boost: build same chain ---
    BoostGraph bg;
    using V = boost::graph_traits<BoostGraph>::vertex_descriptor;
    std::vector<V> bv;
    bv.reserve(n);
    t0 = Clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        bv.push_back(boost::add_vertex(bg));
    }
    auto weight_map = boost::get(boost::edge_weight, bg);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        auto e = boost::add_edge(bv[i], bv[i + 1], bg);
        weight_map[e.first] = 1.f;
    }
    const double boost_build_ms = ms_since(t0);

    std::vector<V> pred(n);
    std::vector<float> dist(n);
    t0 = Clock::now();
    boost::dijkstra_shortest_paths(
        bg,
        bv.front(),
        boost::predecessor_map(pred.data()).distance_map(dist.data()).weight_map(weight_map));
    const double boost_dijkstra_ms = ms_since(t0);
    const float boost_len = dist[bv.back()];

    t0 = Clock::now();
    std::vector<V> boost_path;
    for (V cur = bv.back();; cur = pred[cur]) {
        boost_path.push_back(cur);
        if (cur == bv.front()) {
            break;
        }
    }
    std::reverse(boost_path.begin(), boost_path.end());
    const double boost_path_ms = ms_since(t0);

    std::printf("\nBuild graph\n");
    std::printf("  PathGraph:  %8.2f ms\n", pg_build_ms);
    std::printf("  Boost:      %8.2f ms\n", boost_build_ms);
    std::printf("  ratio PathGraph/Boost: %.2fx\n", pg_build_ms / std::max(1e-9, boost_build_ms));

    std::printf("\nDijkstra (source→sink length only)\n");
    std::printf("  PathGraph:  %8.2f ms  len=%.1f\n", pg_dijkstra_ms, pg_len);
    std::printf("  Boost:      %8.2f ms  len=%.1f\n", boost_dijkstra_ms, boost_len);
    std::printf("  ratio PathGraph/Boost: %.2fx\n",
                pg_dijkstra_ms / std::max(1e-9, boost_dijkstra_ms));

    std::printf("\nPath reconstruct (after / including Dijkstra as implemented)\n");
    std::printf("  PathGraph shortest_path(): %8.2f ms  hops=%zu\n",
                pg_path_ms,
                pg_path.size());
    std::printf("  Boost reconstruct only:    %8.2f ms  hops=%zu\n",
                boost_path_ms,
                boost_path.size());

    if (std::fabs(pg_len - boost_len) > 1e-2f || pg_path.size() != boost_path.size()) {
        std::fprintf(stderr, "ERROR: length/path mismatch\n");
        return 2;
    }
    std::printf("\nOK: lengths match (%.1f), path size %zu\n", pg_len, pg_path.size());
    return 0;
}
