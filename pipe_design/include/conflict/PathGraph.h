#pragma once

#include "conflict/Types.h"
#include "geom/Types.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace pipe_design::conflict {

using NodeId = std::size_t;

/// Invalid / missing node id (e.g. Dijkstra predecessor sentinel).
inline constexpr NodeId kInvalidNode = static_cast<NodeId>(-1);

/// One undirected adjacency entry (stored once per direction).
struct PathEdge {
    NodeId to = 0;
    float weight = 0.f;

    PathEdge() = default;
    PathEdge(NodeId to_, float weight_) : to(to_), weight(weight_) {}
};

/// Undirected weighted graph of one group's pipe network (`SampledPoint` / `SampledPath` only).
///
/// Not related to the planning occupancy grid.
///
/// `add_node` identity = quantized key `(⌊x/eps⌋, ⌊y/eps⌋, ⌊z/eps⌋)`: same key → same
/// `NodeId` (shared terminals connect paths into a tree).
///
/// **Assumption:** any two distinct samples are farther than `eps_`, so each key holds
/// at most one logical point and along-path chains are never collapsed.
class PathGraph {
public:
    static constexpr float kUnreachable = std::numeric_limits<float>::infinity();
    static constexpr float kDefaultEps = 1e-4f;

    PathGraph() = default;
    explicit PathGraph(float eps) : eps_(eps) {}

    void clear();

    void set_eps(float eps);
    float eps() const { return eps_; }

    /// Reuse `NodeId` if quantized key already exists; else copy-insert.
    /// On reuse, stored `radius` becomes `max(existing, incoming)` (geometry unchanged).
    NodeId add_node(const SampledPoint& node);

    /// Lookup by quantized key of `point.center`. Throws if absent.
    NodeId find_node(const SampledPoint& point) const;

    bool add_edge(NodeId a, NodeId b, float weight);
    bool add_edge(NodeId a, NodeId b);

    bool add_direct_edge(NodeId a, NodeId b, float weight);
    bool add_direct_edge(NodeId a, NodeId b);

    std::vector<NodeId> add_sampled_path(const SampledPath& path);

    bool empty() const { return nodes_.empty(); }
    std::size_t node_count() const { return nodes_.size(); }
    std::size_t edge_count() const { return edge_count_; }

    bool valid(NodeId id) const { return id < nodes_.size(); }

    const SampledPoint& node(NodeId id) const { return nodes_[id]; }
    const std::vector<SampledPoint>& nodes() const { return nodes_; }

    std::size_t degree(NodeId id) const;
    const std::vector<PathEdge>& neighbors(NodeId id) const { return adj_[id]; }

    float shortest_path_length(NodeId from, NodeId to) const;
    std::vector<NodeId> shortest_path(NodeId from, NodeId to) const;
    std::vector<NodeId> terminal_nodes() const;

    /// Number of connected components (isolated subgraphs). Isolated nodes count as 1 each.
    std::size_t component_count() const;

    /// Each entry is one connected component (node ids). Order of components / nodes is unspecified.
    std::vector<std::vector<NodeId>> components() const;

    void set_dijkstra_cache_limit(std::size_t limit);
    std::size_t dijkstra_cache_limit() const { return dijkstra_cache_limit_; }
    void clear_dijkstra_cache() const { dijkstra_cache_.clear(); }

private:
    struct CellKey {
        std::int32_t x = 0;
        std::int32_t y = 0;
        std::int32_t z = 0;

        bool operator==(CellKey o) const { return x == o.x && y == o.y && z == o.z; }
    };

    struct CellKeyHash {
        std::size_t operator()(CellKey k) const noexcept {
            const std::size_t hx = static_cast<std::size_t>(static_cast<std::uint32_t>(k.x));
            const std::size_t hy = static_cast<std::size_t>(static_cast<std::uint32_t>(k.y));
            const std::size_t hz = static_cast<std::size_t>(static_cast<std::uint32_t>(k.z));
            return hx * 73856093u ^ hy * 19349663u ^ hz * 83492791u;
        }
    };

    struct DijkstraCacheEntry {
        NodeId source = kInvalidNode;
        std::vector<float> dist;
        std::vector<NodeId> pred;
    };

    CellKey cell_key(const geom::Vec3& p) const;
    void dijkstra(NodeId source,
                  std::vector<float>& dist,
                  std::vector<NodeId>& pred) const;
    /// FIFO cache by `source`; evict oldest when over `dijkstra_cache_limit_`.
    const DijkstraCacheEntry& cached_dijkstra(NodeId source) const;

    float eps_ = kDefaultEps;
    std::vector<SampledPoint> nodes_;
    std::vector<std::vector<PathEdge>> adj_;
    std::unordered_map<CellKey, NodeId, CellKeyHash> spatial_;
    std::size_t edge_count_ = 0;

    static constexpr std::size_t kDefaultDijkstraCacheLimit = 32;
    std::size_t dijkstra_cache_limit_ = kDefaultDijkstraCacheLimit;
    mutable std::vector<DijkstraCacheEntry> dijkstra_cache_;
};

}  // namespace pipe_design::conflict
