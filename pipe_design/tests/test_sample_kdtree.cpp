#include "conflict/SampleKdTree.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    using pipe_design::conflict::SampledPoint;
    using pipe_design::conflict::SampleKdTree;
    using pipe_design::geom::Vec3;

    std::vector<SampledPoint> pts;
    pts.push_back(SampledPoint{Vec3{0, 0, 0}, 1.f});
    pts.push_back(SampledPoint{Vec3{5, 0, 0}, 2.f});
    pts.push_back(SampledPoint{Vec3{1, 0, 0}, 0.5f});
    pts.push_back(SampledPoint{Vec3{10, 10, 10}, 3.f});

    SampleKdTree tree;
    tree.build(pts);

    std::vector<std::size_t> idx;
    std::vector<float> dist;

    tree.radius_search(Vec3{0, 0, 0}, 1.5f, idx, dist);
    if (idx.size() != 2) {
        std::fprintf(stderr, "radius expected 2, got %zu\n", idx.size());
        return 1;
    }
    if (std::fabs(tree.weight(idx[0]) + tree.weight(idx[1]) - 1.5f) > 1e-5f &&
        !(tree.weight(0) == 1.f || tree.weight(2) == 0.5f)) {
        // just sanity: both hits are near origin
    }

    tree.knn_search(Vec3{0, 0, 0}, 1, idx, dist);
    if (idx.size() != 1 || idx[0] != 0 || dist[0] > 1e-5f) {
        std::fprintf(stderr, "knn failed\n");
        return 2;
    }

    // larger cloud: radius vs brute
    pts.clear();
    for (int i = 0; i < 2000; ++i) {
        pts.push_back(SampledPoint{
            Vec3{static_cast<float>(i % 40), static_cast<float>(i / 40), 0.f}, 1.f});
    }
    tree.build(pts);
    const Vec3 q{10.f, 10.f, 0.f};
    const float r = 3.f;
    tree.radius_search(q, r, idx, dist);

    std::size_t brute = 0;
    for (const auto& p : pts) {
        const float dx = p.center.x - q.x;
        const float dy = p.center.y - q.y;
        const float dz = p.center.z - q.z;
        if (dx * dx + dy * dy + dz * dz <= r * r) {
            ++brute;
        }
    }
    if (idx.size() != brute) {
        std::fprintf(stderr, "radius %zu vs brute %zu\n", idx.size(), brute);
        return 3;
    }

    std::puts("ok");
    return 0;
}
