#include "topology/Mst.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using pipe_design::topology::minimum_spanning_tree;
using pipe_design::topology::MstEdge;

int g_fails = 0;

void expect(bool ok, const char* msg) {
    if (!ok) {
        std::cerr << "FAIL: " << msg << '\n';
        ++g_fails;
    }
}

float total_cost(const std::vector<MstEdge>& edges) {
    float s = 0.f;
    for (const MstEdge& e : edges) {
        s += e.cost;
    }
    return s;
}

void test_triangle() {
    // 0--1 cost 1, 1--2 cost 1, 0--2 cost 3 → MST = {0-1, 1-2}
    const std::size_t n = 3;
    std::vector<float> cost(n * n, 0.f);
    auto set = [&](std::size_t i, std::size_t j, float w) {
        cost[i * n + j] = w;
        cost[j * n + i] = w;
    };
    set(0, 1, 1.f);
    set(1, 2, 1.f);
    set(0, 2, 3.f);

    const auto edges = minimum_spanning_tree(n, cost);
    expect(edges.size() == 2, "triangle: 2 edges");
    expect(std::fabs(total_cost(edges) - 2.f) < 1e-5f, "triangle: total cost 2");
}

void test_square() {
    // unit square: diagonal longer → MST uses three sides of length 1
    const std::size_t n = 4;
    std::vector<float> cost(n * n, 0.f);
    auto set = [&](std::size_t i, std::size_t j, float w) {
        cost[i * n + j] = w;
        cost[j * n + i] = w;
    };
    set(0, 1, 1.f);
    set(1, 2, 1.f);
    set(2, 3, 1.f);
    set(3, 0, 1.f);
    set(0, 2, std::sqrt(2.f));
    set(1, 3, std::sqrt(2.f));

    const auto edges = minimum_spanning_tree(n, cost);
    expect(edges.size() == 3, "square: 3 edges");
    expect(std::fabs(total_cost(edges) - 3.f) < 1e-5f, "square: total cost 3");
}

void test_disconnected() {
    constexpr float kInf = std::numeric_limits<float>::infinity();
    const std::size_t n = 3;
    std::vector<float> cost(n * n, kInf);
    cost[0 * n + 1] = 1.f;
    cost[1 * n + 0] = 1.f;
    // vertex 2 isolated

    const auto edges = minimum_spanning_tree(n, cost);
    expect(edges.size() == 1, "forest: one edge");
    expect(std::fabs(edges[0].cost - 1.f) < 1e-5f, "forest: edge cost 1");
}

void test_empty() {
    expect(minimum_spanning_tree(0, static_cast<const float*>(nullptr)).empty(),
           "n=0 empty");
    expect(minimum_spanning_tree(1, std::vector<float>{0.f}).empty(), "n=1 empty");
}

}  // namespace

int main() {
    test_empty();
    test_triangle();
    test_square();
    test_disconnected();
    if (g_fails != 0) {
        std::cerr << g_fails << " failure(s)\n";
        return EXIT_FAILURE;
    }
    std::cout << "mst_unit ok\n";
    return EXIT_SUCCESS;
}
