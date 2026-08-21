#pragma once

#include "geom/Types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace pipe_design::geom {

struct Aabb {
    Vec3 mn;
    Vec3 mx;

    bool overlaps(const Aabb& o) const {
        return mn.x <= o.mx.x && mx.x >= o.mn.x && mn.y <= o.mx.y && mx.y >= o.mn.y &&
               mn.z <= o.mx.z && mx.z >= o.mn.z;
    }

    void merge(const Aabb& o) {
        mn.x = std::min(mn.x, o.mn.x);
        mn.y = std::min(mn.y, o.mn.y);
        mn.z = std::min(mn.z, o.mn.z);
        mx.x = std::max(mx.x, o.mx.x);
        mx.y = std::max(mx.y, o.mx.y);
        mx.z = std::max(mx.z, o.mx.z);
    }

    void expand(float r) {
        mn.x -= r;
        mn.y -= r;
        mn.z -= r;
        mx.x += r;
        mx.y += r;
        mx.z += r;
    }
};

inline Aabb aabb_of(const SphereObstacle& s) {
    return {
        {s.center.x - s.radius, s.center.y - s.radius, s.center.z - s.radius},
        {s.center.x + s.radius, s.center.y + s.radius, s.center.z + s.radius},
    };
}

inline Aabb aabb_of(const CapsuleObstacle& c) {
    return {
        {
            std::min(c.a.x, c.b.x) - c.radius,
            std::min(c.a.y, c.b.y) - c.radius,
            std::min(c.a.z, c.b.z) - c.radius,
        },
        {
            std::max(c.a.x, c.b.x) + c.radius,
            std::max(c.a.y, c.b.y) + c.radius,
            std::max(c.a.z, c.b.z) + c.radius,
        },
    };
}

inline Aabb aabb_of(const AabbObstacle& b) {
    return {b.mn, b.mx};
}

inline Aabb aabb_of_segment(const Vec3& a, const Vec3& b, float inflate = 0.f) {
    Aabb box{
        {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)},
        {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)},
    };
    if (inflate > 0.f) {
        box.expand(inflate);
    }
    return box;
}

/// Binary BVH over sphere/capsule/AABB solids. Leaf stores refs; narrowphase is caller's job.
class Bvh {
public:
    struct Ref {
        ObstacleKind kind = ObstacleKind::Sphere;
        int index = 0;
    };

    void clear() {
        nodes_.clear();
        items_.clear();
        root_ = -1;
    }

    void build(
        const std::vector<SphereObstacle>& spheres,
        const std::vector<CapsuleObstacle>& capsules,
        const std::vector<AabbObstacle>& aabbs = {}) {
        clear();
        items_.reserve(spheres.size() + capsules.size() + aabbs.size());
        for (int i = 0; i < static_cast<int>(spheres.size()); ++i) {
            items_.push_back(
                Item{ObstacleKind::Sphere, i, aabb_of(spheres[static_cast<std::size_t>(i)])});
        }
        for (int i = 0; i < static_cast<int>(capsules.size()); ++i) {
            items_.push_back(
                Item{ObstacleKind::Capsule, i, aabb_of(capsules[static_cast<std::size_t>(i)])});
        }
        for (int i = 0; i < static_cast<int>(aabbs.size()); ++i) {
            items_.push_back(
                Item{ObstacleKind::Aabb, i, aabb_of(aabbs[static_cast<std::size_t>(i)])});
        }
        if (items_.empty()) {
            return;
        }
        root_ = build_range(0, static_cast<int>(items_.size()));
    }

    bool empty() const { return root_ < 0; }

    /// Append refs whose stored AABB overlaps `box` (no dedup needed across nodes).
    void query(const Aabb& box, std::vector<Ref>& out) const {
        if (root_ < 0) {
            return;
        }
        query_node(root_, box, out);
    }

private:
    static constexpr int kLeafSize = 4;

    struct Node {
        Aabb box{};
        int left = -1;
        int right = -1;
        int begin = 0;
        int count = 0;  // >0 => leaf over items_[begin, begin+count)
    };

    struct Item {
        ObstacleKind kind = ObstacleKind::Sphere;
        int index = 0;
        Aabb box{};
    };

    int build_range(int begin, int end) {
        Node node;
        node.box = items_[static_cast<std::size_t>(begin)].box;
        for (int i = begin + 1; i < end; ++i) {
            node.box.merge(items_[static_cast<std::size_t>(i)].box);
        }

        const int count = end - begin;
        if (count <= kLeafSize) {
            node.begin = begin;
            node.count = count;
            nodes_.push_back(node);
            return static_cast<int>(nodes_.size()) - 1;
        }

        const float ex = node.box.mx.x - node.box.mn.x;
        const float ey = node.box.mx.y - node.box.mn.y;
        const float ez = node.box.mx.z - node.box.mn.z;
        int axis = 0;
        if (ey > ex && ey >= ez) {
            axis = 1;
        } else if (ez > ex && ez >= ey) {
            axis = 2;
        }

        const int mid = begin + count / 2;
        auto center = [axis](const Item& it) {
            if (axis == 0) {
                return 0.5f * (it.box.mn.x + it.box.mx.x);
            }
            if (axis == 1) {
                return 0.5f * (it.box.mn.y + it.box.mx.y);
            }
            return 0.5f * (it.box.mn.z + it.box.mx.z);
        };
        std::nth_element(
            items_.begin() + begin,
            items_.begin() + mid,
            items_.begin() + end,
            [&](const Item& a, const Item& b) { return center(a) < center(b); });

        const int node_index = static_cast<int>(nodes_.size());
        nodes_.push_back(Node{});  // placeholder

        const int left = build_range(begin, mid);
        const int right = build_range(mid, end);
        nodes_[static_cast<std::size_t>(node_index)].box = node.box;
        nodes_[static_cast<std::size_t>(node_index)].left = left;
        nodes_[static_cast<std::size_t>(node_index)].right = right;
        nodes_[static_cast<std::size_t>(node_index)].count = 0;
        return node_index;
    }

    void query_node(int node_index, const Aabb& box, std::vector<Ref>& out) const {
        const Node& node = nodes_[static_cast<std::size_t>(node_index)];
        if (!node.box.overlaps(box)) {
            return;
        }
        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const Item& item = items_[static_cast<std::size_t>(node.begin + i)];
                if (item.box.overlaps(box)) {
                    out.push_back(Ref{item.kind, item.index});
                }
            }
            return;
        }
        query_node(node.left, box, out);
        query_node(node.right, box, out);
    }

    std::vector<Node> nodes_;
    std::vector<Item> items_;
    int root_ = -1;
};

}  // namespace pipe_design::geom
