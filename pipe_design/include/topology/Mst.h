#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pipe_design::topology {

/// One undirected MST edge (indices into the cost-map vertex set).
struct MstEdge {
    std::size_t u = 0;
    std::size_t v = 0;
    float cost = 0.f;
};

/// Dense Prim MST on a complete (or dense) cost map.
///
/// `cost` is row-major `n × n`; edge `{i,j}` weight is `cost[i * n + j]`.
/// Use `+infinity` for a missing edge. Symmetric costs expected.
///
/// Returns `|V|-1` edges if the finite subgraph is connected; fewer if a forest.
/// Time / space: O(n²) / O(n) — optimal for dense maps (typical terminal MST).
inline std::vector<MstEdge> minimum_spanning_tree(std::size_t n, const float* cost) {
    std::vector<MstEdge> out;
    if (n < 2 || cost == nullptr) {
        throw std::invalid_argument("minimum_spanning_tree: invalid input");
    }

    constexpr float kInf = std::numeric_limits<float>::infinity();
    std::vector<float> key(n, kInf);
    std::vector<std::size_t> parent(n, n);  // n = none
    std::vector<unsigned char> in_tree(n, 0);

    key[0] = 0.f;
    out.reserve(n - 1);

    for (std::size_t iter = 0; iter < n; ++iter) {
        std::size_t u = n;
        float best = kInf;
        for (std::size_t i = 0; i < n; ++i) {
            if (!in_tree[i] && key[i] < best) {
                best = key[i];
                u = i;
            }
        }
        if (u == n) {
            break;  // remaining vertices unreachable
        }

        in_tree[u] = 1;
        if (parent[u] != n) {
            out.push_back(MstEdge{parent[u], u, key[u]});
        }

        const float* row = cost + u * n;
        for (std::size_t v = 0; v < n; ++v) {
            if (in_tree[v]) {
                continue;
            }
            const float w = row[v];
            if (w < key[v]) {
                key[v] = w;
                parent[v] = u;
            }
        }
    }
    return out;
}

inline std::vector<MstEdge> minimum_spanning_tree(std::size_t n,
                                                   const std::vector<float>& cost) {
    return minimum_spanning_tree(n, cost.data());
}

}  // namespace pipe_design::topology
