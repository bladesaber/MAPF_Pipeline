#pragma once

#include "conflict/Types.h"
#include "geom/Types.h"

#include <cstddef>
#include <vector>

namespace pipe_design::conflict {

/// Exact 3D KD-tree over `SampledPoint` (radius acts as weight).
///
/// Non-owning / read-only: `build` does **not** copy or mutate points. Caller must
/// keep the point array alive for the lifetime of this tree (e.g. `PathGraph::nodes()`).
class SampleKdTree {
public:
    SampleKdTree() = default;

    void clear();

    /// Bind to an external point array (no copy). Indices in search results refer to it.
    void build(const SampledPoint* points, std::size_t count);

    /// Same as `build(points.data(), points.size())`.
    void build(const std::vector<SampledPoint>& points) {
        build(points.data(), points.size());
    }

    bool empty() const { return size_ == 0; }
    std::size_t size() const { return size_; }

    const SampledPoint& point(std::size_t i) const { return points_[i]; }
    float weight(std::size_t i) const { return points_[i].radius; }

    /// All points with center distance ≤ `radius`. `dists` are Euclidean lengths.
    void radius_search(const geom::Vec3& query,
                       float radius,
                       std::vector<std::size_t>& indices,
                       std::vector<float>& dists) const;

    /// `k` nearest centers. `dists` are Euclidean lengths (same order as `indices`).
    void knn_search(const geom::Vec3& query,
                    std::size_t k,
                    std::vector<std::size_t>& indices,
                    std::vector<float>& dists) const;

private:
    struct Node {
        int left = -1;
        int right = -1;
        int axis = 0;
        std::size_t index = 0;
    };

    int build_range(std::size_t* idx, std::size_t count, int depth);
    void radius_rec(int node,
                    const geom::Vec3& query,
                    float radius_sq,
                    std::vector<std::size_t>& indices,
                    std::vector<float>& dists) const;
    void knn_rec(int node,
                 const geom::Vec3& query,
                 std::size_t k,
                 std::vector<std::pair<float, std::size_t>>& heap) const;

    const SampledPoint* points_ = nullptr;
    std::size_t size_ = 0;
    std::vector<Node> nodes_;
    int root_ = -1;
};

}  // namespace pipe_design::conflict
