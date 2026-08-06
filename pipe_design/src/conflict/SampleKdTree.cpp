#include "conflict/SampleKdTree.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace pipe_design::conflict {

namespace {

float dist_sq(const geom::Vec3& a, const geom::Vec3& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

float coord(const geom::Vec3& p, int axis) {
    return axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
}

}  // namespace

void SampleKdTree::clear() {
    points_ = nullptr;
    size_ = 0;
    nodes_.clear();
    root_ = -1;
}

void SampleKdTree::build(const SampledPoint* points, std::size_t count) {
    clear();
    if (points == nullptr || count == 0) {
        return;
    }
    points_ = points;
    size_ = count;
    nodes_.reserve(count);
    std::vector<std::size_t> idx(count);
    for (std::size_t i = 0; i < count; ++i) {
        idx[i] = i;
    }
    root_ = build_range(idx.data(), count, 0);
}

int SampleKdTree::build_range(std::size_t* idx, std::size_t count, int depth) {
    if (count == 0) {
        return -1;
    }
    const int axis = depth % 3;
    const std::size_t mid = count / 2;
    std::nth_element(idx, idx + mid, idx + count, [&](std::size_t a, std::size_t b) {
        return coord(points_[a].center, axis) < coord(points_[b].center, axis);
    });

    const int node = static_cast<int>(nodes_.size());
    nodes_.push_back(Node{});
    nodes_[node].axis = axis;
    nodes_[node].index = idx[mid];
    nodes_[node].left = build_range(idx, mid, depth + 1);
    nodes_[node].right = build_range(idx + mid + 1, count - mid - 1, depth + 1);
    return node;
}

void SampleKdTree::radius_rec(int node,
                              const geom::Vec3& query,
                              float radius_sq,
                              std::vector<std::size_t>& indices,
                              std::vector<float>& dists) const {
    if (node < 0) {
        return;
    }
    const Node& n = nodes_[static_cast<std::size_t>(node)];
    const SampledPoint& p = points_[n.index];
    const float d2 = dist_sq(query, p.center);
    if (d2 <= radius_sq) {
        indices.push_back(n.index);
        dists.push_back(std::sqrt(d2));
    }

    const float delta = coord(query, n.axis) - coord(p.center, n.axis);
    const int first = delta < 0.f ? n.left : n.right;
    const int second = delta < 0.f ? n.right : n.left;
    radius_rec(first, query, radius_sq, indices, dists);
    if (delta * delta <= radius_sq) {
        radius_rec(second, query, radius_sq, indices, dists);
    }
}

void SampleKdTree::radius_search(const geom::Vec3& query,
                                 float radius,
                                 std::vector<std::size_t>& indices,
                                 std::vector<float>& dists) const {
    indices.clear();
    dists.clear();
    if (root_ < 0 || radius < 0.f) {
        return;
    }
    radius_rec(root_, query, radius * radius, indices, dists);
}

void SampleKdTree::knn_rec(int node,
                           const geom::Vec3& query,
                           std::size_t k,
                           std::vector<std::pair<float, std::size_t>>& heap) const {
    if (node < 0) {
        return;
    }
    const Node& n = nodes_[static_cast<std::size_t>(node)];
    const SampledPoint& p = points_[n.index];
    const float d2 = dist_sq(query, p.center);

    if (heap.size() < k) {
        heap.push_back({d2, n.index});
        std::push_heap(heap.begin(), heap.end());
    } else if (d2 < heap.front().first) {
        std::pop_heap(heap.begin(), heap.end());
        heap.back() = {d2, n.index};
        std::push_heap(heap.begin(), heap.end());
    }

    const float delta = coord(query, n.axis) - coord(p.center, n.axis);
    const int first = delta < 0.f ? n.left : n.right;
    const int second = delta < 0.f ? n.right : n.left;
    knn_rec(first, query, k, heap);
    const float worst = heap.size() < k ? std::numeric_limits<float>::infinity() : heap.front().first;
    if (delta * delta <= worst) {
        knn_rec(second, query, k, heap);
    }
}

void SampleKdTree::knn_search(const geom::Vec3& query,
                              std::size_t k,
                              std::vector<std::size_t>& indices,
                              std::vector<float>& dists) const {
    indices.clear();
    dists.clear();
    if (root_ < 0 || k == 0) {
        return;
    }
    std::vector<std::pair<float, std::size_t>> heap;
    heap.reserve(k);
    knn_rec(root_, query, k, heap);
    std::sort_heap(heap.begin(), heap.end());
    indices.reserve(heap.size());
    dists.reserve(heap.size());
    for (const auto& e : heap) {
        indices.push_back(e.second);
        dists.push_back(std::sqrt(e.first));
    }
}

}  // namespace pipe_design::conflict
