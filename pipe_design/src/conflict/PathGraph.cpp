#include "conflict/PathGraph.h"

#include "geom/Geometry.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <utility>

namespace pipe_design::conflict {

namespace {

struct DistNode {
    float dist = 0.f;
    NodeId id = 0;

    bool operator>(const DistNode& o) const { return dist > o.dist; }
};

}  // namespace

void PathGraph::clear() {
    nodes_.clear();
    adj_.clear();
    spatial_.clear();
    edge_count_ = 0;
    dijkstra_cache_.clear();
}

void PathGraph::set_dijkstra_cache_limit(std::size_t limit) {
    dijkstra_cache_limit_ = limit;
    while (dijkstra_cache_.size() > dijkstra_cache_limit_) {
        dijkstra_cache_.erase(dijkstra_cache_.begin());
    }
}

void PathGraph::set_eps(float eps) {
    eps_ = eps;
    spatial_.clear();
    for (NodeId i = 0; i < nodes_.size(); ++i) {
        spatial_[cell_key(nodes_[i].center)] = i;
    }
}

PathGraph::CellKey PathGraph::cell_key(const geom::Vec3& p) const {
    const float e = eps_ > 0.f ? eps_ : kDefaultEps;
    return CellKey{
        static_cast<std::int32_t>(std::floor(p.x / e)),
        static_cast<std::int32_t>(std::floor(p.y / e)),
        static_cast<std::int32_t>(std::floor(p.z / e)),
    };
}

NodeId PathGraph::add_node(const SampledPoint& node) {
    const CellKey key = cell_key(node.center);
    const auto it = spatial_.find(key);
    if (it != spatial_.end()) {
        const NodeId id = it->second;
        if (node.radius > nodes_[id].radius) {
            nodes_[id].radius = node.radius;
        }
        return id;
    }

    const NodeId id = nodes_.size();
    nodes_.push_back(node);
    adj_.emplace_back();
    spatial_.emplace(key, id);
    dijkstra_cache_.clear();
    return id;
}

NodeId PathGraph::find_node(const SampledPoint& point) const {
    const auto it = spatial_.find(cell_key(point.center));
    if (it == spatial_.end()) {
        throw std::runtime_error("Node not found");
    }
    return it->second;
}

bool PathGraph::add_edge(NodeId a, NodeId b, float weight) {
    if (!valid(a) || !valid(b) || a == b) {
        return false;
    }
    adj_[a].push_back(PathEdge{b, weight});
    adj_[b].push_back(PathEdge{a, weight});
    ++edge_count_;
    dijkstra_cache_.clear();
    return true;
}

bool PathGraph::add_edge(NodeId a, NodeId b) {
    if (!valid(a) || !valid(b) || a == b) {
        return false;
    }
    return add_edge(a, b, geom::distance(nodes_[a].center, nodes_[b].center));
}

bool PathGraph::add_direct_edge(NodeId a, NodeId b, float weight) {
    if (!valid(a) || !valid(b) || a == b) {
        return false;
    }
    adj_[a].push_back(PathEdge{b, weight});
    ++edge_count_;
    dijkstra_cache_.clear();
    return true;
}

bool PathGraph::add_direct_edge(NodeId a, NodeId b) {
    if (!valid(a) || !valid(b) || a == b) {
        return false;
    }
    return add_direct_edge(a, b, geom::distance(nodes_[a].center, nodes_[b].center));
}

std::vector<NodeId> PathGraph::add_sampled_path(const SampledPath& path) {
    std::vector<NodeId> ids;
    for (const SampledPoint& s : path.samples) {
        ids.push_back(add_node(s));
    }
    for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
        add_edge(ids[i], ids[i + 1]);
    }
    return ids;
}

std::size_t PathGraph::degree(NodeId id) const {
    if (!valid(id)) {
        return 0;
    }
    return adj_[id].size();
}

void PathGraph::dijkstra(NodeId source,
                         std::vector<float>& dist,
                         std::vector<NodeId>& pred) const {
    const std::size_t n = nodes_.size();
    dist.assign(n, kUnreachable);
    pred.assign(n, kInvalidNode);
    if (!valid(source)) {
        return;
    }

    dist[source] = 0.f;
    std::priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>> open;
    open.push(DistNode{0.f, source});

    while (!open.empty()) {
        const DistNode cur = open.top();
        open.pop();
        if (cur.dist > dist[cur.id]) {
            continue;
        }
        for (const PathEdge& e : adj_[cur.id]) {
            const float nd = cur.dist + e.weight;
            if (nd < dist[e.to]) {
                dist[e.to] = nd;
                pred[e.to] = cur.id;
                open.push(DistNode{nd, e.to});
            }
        }
    }
}

const PathGraph::DijkstraCacheEntry& PathGraph::cached_dijkstra(NodeId source) const {
    for (DijkstraCacheEntry& e : dijkstra_cache_) {
        if (e.source == source) {
            return e;
        }
    }

    DijkstraCacheEntry entry;
    entry.source = source;
    dijkstra(source, entry.dist, entry.pred);

    if (dijkstra_cache_limit_ == 0) {
        dijkstra_cache_.clear();
        dijkstra_cache_.push_back(std::move(entry));
        return dijkstra_cache_.back();
    }

    while (dijkstra_cache_.size() >= dijkstra_cache_limit_) {
        dijkstra_cache_.erase(dijkstra_cache_.begin());
    }
    dijkstra_cache_.push_back(std::move(entry));
    return dijkstra_cache_.back();
}

float PathGraph::shortest_path_length(NodeId from, NodeId to) const {
    if (!valid(from) || !valid(to)) {
        return kUnreachable;
    }
    if (from == to) {
        return 0.f;
    }
    return cached_dijkstra(from).dist[to];
}

std::vector<NodeId> PathGraph::shortest_path(NodeId from, NodeId to) const {
    std::vector<NodeId> path;
    if (!valid(from) || !valid(to)) {
        return path;
    }
    if (from == to) {
        path.push_back(from);
        return path;
    }

    const DijkstraCacheEntry& entry = cached_dijkstra(from);
    if (entry.dist[to] == kUnreachable) {
        throw std::runtime_error("Path not found");
    }

    for (NodeId cur = to; cur != kInvalidNode; cur = entry.pred[cur]) {
        path.push_back(cur);
        if (cur == from) {
            break;
        }
    }
    std::reverse(path.begin(), path.end());
    if (path.empty() || path.front() != from) {
        path.clear();
    }
    return path;
}

std::vector<NodeId> PathGraph::terminal_nodes() const {
    std::vector<NodeId> out;
    for (NodeId id = 0; id < nodes_.size(); ++id) {
        if (adj_[id].size() == 1) {
            out.push_back(id);
        }
    }
    return out;
}

std::size_t PathGraph::component_count() const {
    return components().size();
}

std::vector<std::vector<NodeId>> PathGraph::components() const {
    const std::size_t n = nodes_.size();
    std::vector<std::vector<NodeId>> out;
    if (n == 0) {
        return out;
    }
    std::vector<char> seen(n, 0);
    std::vector<NodeId> stack;
    for (NodeId s = 0; s < n; ++s) {
        if (seen[s]) {
            continue;
        }
        std::vector<NodeId> comp;
        stack.clear();
        stack.push_back(s);
        seen[s] = 1;
        while (!stack.empty()) {
            const NodeId u = stack.back();
            stack.pop_back();
            comp.push_back(u);
            for (const PathEdge& e : adj_[u]) {
                if (!seen[e.to]) {
                    seen[e.to] = 1;
                    stack.push_back(e.to);
                }
            }
        }
        out.push_back(std::move(comp));
    }
    return out;
}

}  // namespace pipe_design::conflict
