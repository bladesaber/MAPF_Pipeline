// Bench: SampleKdTree vs FLANN KDTreeSingleIndex (exact), N >= 50000.
// Usage: ./bench_sample_kdtree [N] [n_queries]

#include "conflict/SampleKdTree.h"
#include "conflict/Types.h"

#include <flann/flann.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using pipe_design::conflict::SampledPoint;
using pipe_design::conflict::SampleKdTree;
using pipe_design::geom::Vec3;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::vector<SampledPoint> make_points(std::size_t n, std::mt19937& rng) {
    // Pseudo-random cloud in a box (pipe-sample like density).
    std::uniform_real_distribution<float> u(0.f, 100.f);
    std::uniform_real_distribution<float> r(0.5f, 2.f);
    std::vector<SampledPoint> pts;
    pts.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts.push_back(SampledPoint{Vec3{u(rng), u(rng), u(rng)}, r(rng)});
    }
    return pts;
}

std::vector<Vec3> make_queries(std::size_t m, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(0.f, 100.f);
    std::vector<Vec3> q;
    q.reserve(m);
    for (std::size_t i = 0; i < m; ++i) {
        q.push_back(Vec3{u(rng), u(rng), u(rng)});
    }
    return q;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10))
                                   : 50000;
    const std::size_t nq = argc > 2 ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10))
                                    : 1000;
    if (n < 2 || nq < 1) {
        std::fprintf(stderr, "usage: %s [N>=2] [n_queries>=1]\n", argv[0]);
        return 1;
    }

    std::mt19937 rng(42);
    const auto points = make_points(n, rng);
    const auto queries = make_queries(nq, rng);
    const float radius = 3.f;
    const int k = 1;

    std::printf("=== SampleKdTree vs FLANN (N=%zu points, %zu queries, radius=%.1f, k=%d) ===\n",
                n,
                nq,
                radius,
                k);

    // ----- SampleKdTree build -----
    SampleKdTree ours;
    auto t0 = Clock::now();
    ours.build(points);
    const double ours_build_ms = ms_since(t0);

    // ----- FLANN build (exact single KD-tree, same as legacy) -----
    std::vector<float> flann_data(n * 3);
    for (std::size_t i = 0; i < n; ++i) {
        flann_data[i * 3 + 0] = points[i].center.x;
        flann_data[i * 3 + 1] = points[i].center.y;
        flann_data[i * 3 + 2] = points[i].center.z;
    }
    flann::Matrix<float> dataset(flann_data.data(), n, 3);

    t0 = Clock::now();
    flann::Index<flann::L2<float>> flann_index(dataset, flann::KDTreeSingleIndexParams());
    flann_index.buildIndex();
    const double flann_build_ms = ms_since(t0);

    std::printf("\nBuild\n");
    std::printf("  SampleKdTree: %8.2f ms\n", ours_build_ms);
    std::printf("  FLANN:        %8.2f ms\n", flann_build_ms);
    std::printf("  ratio ours/FLANN: %.2fx\n", ours_build_ms / std::max(1e-9, flann_build_ms));

    // ----- knn(k=1) correctness + speed -----
    std::vector<std::size_t> o_idx;
    std::vector<float> o_dist;
    std::vector<std::vector<int>> f_idx(1);
    std::vector<std::vector<float>> f_dist(1);
    // FLANN SearchParams: checks=-1 means exact for KDTreeSingle
    flann::SearchParams exact_params(-1);

    std::size_t knn_mismatch = 0;
    double ours_knn_ms = 0.0;
    double flann_knn_ms = 0.0;

    for (const Vec3& q : queries) {
        t0 = Clock::now();
        ours.knn_search(q, static_cast<std::size_t>(k), o_idx, o_dist);
        ours_knn_ms += ms_since(t0);

        float qdata[3] = {q.x, q.y, q.z};
        flann::Matrix<float> query(qdata, 1, 3);
        t0 = Clock::now();
        flann_index.knnSearch(query, f_idx, f_dist, k, exact_params);
        flann_knn_ms += ms_since(t0);

        if (o_idx.size() != f_idx[0].size()) {
            ++knn_mismatch;
            continue;
        }
        // FLANN L2 returns squared distance
        for (std::size_t i = 0; i < o_idx.size(); ++i) {
            if (static_cast<int>(o_idx[i]) != f_idx[0][i]) {
                // tie-breaking may differ if equal distance; check distance
                const float fd = std::sqrt(f_dist[0][i]);
                if (std::fabs(o_dist[i] - fd) > 1e-4f) {
                    ++knn_mismatch;
                    break;
                }
            } else if (std::fabs(o_dist[i] - std::sqrt(f_dist[0][i])) > 1e-3f) {
                ++knn_mismatch;
                break;
            }
        }
    }

    std::printf("\nknnSearch (k=%d, %zu queries)\n", k, nq);
    std::printf("  SampleKdTree: %8.2f ms  (%.3f us/query)\n",
                ours_knn_ms,
                1000.0 * ours_knn_ms / static_cast<double>(nq));
    std::printf("  FLANN:        %8.2f ms  (%.3f us/query)\n",
                flann_knn_ms,
                1000.0 * flann_knn_ms / static_cast<double>(nq));
    std::printf("  ratio ours/FLANN: %.2fx\n", ours_knn_ms / std::max(1e-9, flann_knn_ms));
    std::printf("  result mismatches: %zu / %zu\n", knn_mismatch, nq);

    // ----- radius_search correctness + speed -----
    std::size_t radius_count_mismatch = 0;
    std::size_t radius_set_mismatch = 0;
    std::size_t ours_hit_total = 0;
    std::size_t flann_hit_total = 0;
    double ours_radius_ms = 0.0;
    double flann_radius_ms = 0.0;
    const float radius_sq = radius * radius;

    for (const Vec3& q : queries) {
        t0 = Clock::now();
        ours.radius_search(q, radius, o_idx, o_dist);
        ours_radius_ms += ms_since(t0);
        ours_hit_total += o_idx.size();

        float qdata[3] = {q.x, q.y, q.z};
        flann::Matrix<float> query(qdata, 1, 3);
        t0 = Clock::now();
        // FLANN radiusSearch: radius is squared L2 distance
        flann_index.radiusSearch(query, f_idx, f_dist, radius_sq, exact_params);
        flann_radius_ms += ms_since(t0);
        flann_hit_total += f_idx[0].size();

        if (o_idx.size() != f_idx[0].size()) {
            ++radius_count_mismatch;
        }

        std::vector<int> a(o_idx.begin(), o_idx.end());
        std::vector<int> b = f_idx[0];
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) {
            ++radius_set_mismatch;
        }
    }

    std::printf("\nradiusSearch (r=%.1f, %zu queries)\n", radius, nq);
    std::printf("  SampleKdTree: %8.2f ms  (%.3f us/query)  hits=%zu\n",
                ours_radius_ms,
                1000.0 * ours_radius_ms / static_cast<double>(nq),
                ours_hit_total);
    std::printf("  FLANN:        %8.2f ms  (%.3f us/query)  hits=%zu\n",
                flann_radius_ms,
                1000.0 * flann_radius_ms / static_cast<double>(nq),
                flann_hit_total);
    std::printf("  ratio ours/FLANN: %.2fx\n", ours_radius_ms / std::max(1e-9, flann_radius_ms));
    std::printf("  count mismatches: %zu / %zu\n", radius_count_mismatch, nq);
    std::printf("  index-set mismatches: %zu / %zu\n", radius_set_mismatch, nq);

    // Spot-check vs brute on first query
    {
        const Vec3& q = queries[0];
        ours.radius_search(q, radius, o_idx, o_dist);
        std::size_t brute = 0;
        for (const auto& p : points) {
            const float dx = p.center.x - q.x;
            const float dy = p.center.y - q.y;
            const float dz = p.center.z - q.z;
            if (dx * dx + dy * dy + dz * dz <= radius_sq) {
                ++brute;
            }
        }
        std::printf("\nBrute check (query[0]): ours=%zu brute=%zu %s\n",
                    o_idx.size(),
                    brute,
                    o_idx.size() == brute ? "OK" : "FAIL");
        if (o_idx.size() != brute || radius_set_mismatch > 0 || knn_mismatch > nq / 100) {
            std::fprintf(stderr, "ERROR: correctness failure\n");
            return 2;
        }
    }

    std::printf("\nOK\n");
    return 0;
}
