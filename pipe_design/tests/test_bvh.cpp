#include "geom/Bvh.h"
#include "geom/Types.h"

#include <iostream>
#include <vector>

using pipe_design::geom::Aabb;
using pipe_design::geom::Bvh;
using pipe_design::geom::CapsuleObstacle;
using pipe_design::geom::SphereObstacle;
using pipe_design::geom::Vec3;
using pipe_design::geom::aabb_of_segment;

static int g_failures = 0;

#define EXPECT_TRUE(cond)                                                          \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " #cond "\n"; \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

static void test_bvh_query_filters() {
    std::vector<SphereObstacle> spheres = {
        SphereObstacle{Vec3{0.f, 0.f, 0.f}, 0.5f},
        SphereObstacle{Vec3{10.f, 10.f, 0.f}, 0.5f},
        SphereObstacle{Vec3{20.f, 0.f, 0.f}, 0.5f},
    };
    std::vector<CapsuleObstacle> capsules = {
        CapsuleObstacle{Vec3{0.f, 5.f, 0.f}, Vec3{2.f, 5.f, 0.f}, 0.25f},
    };

    Bvh bvh;
    bvh.build(spheres, capsules);
    EXPECT_TRUE(!bvh.empty());

    std::vector<Bvh::Ref> hits;
    // Near origin: should see sphere 0 and maybe capsule, not far spheres.
    bvh.query(aabb_of_segment(Vec3{-1.f, 0.f, 0.f}, Vec3{1.f, 0.f, 0.f}, 0.1f), hits);

    bool saw0 = false;
    bool saw_far = false;
    bool saw_capsule = false;
    for (const Bvh::Ref& r : hits) {
        if (r.kind == Bvh::Kind::Sphere && r.index == 0) {
            saw0 = true;
        }
        if (r.kind == Bvh::Kind::Sphere && (r.index == 1 || r.index == 2)) {
            saw_far = true;
        }
        if (r.kind == Bvh::Kind::Capsule && r.index == 0) {
            saw_capsule = true;
        }
    }
    EXPECT_TRUE(saw0);
    EXPECT_TRUE(!saw_far);
    EXPECT_TRUE(!saw_capsule);

    hits.clear();
    bvh.query(aabb_of_segment(Vec3{0.f, 5.f, 0.f}, Vec3{1.f, 5.f, 0.f}, 0.1f), hits);
    saw_capsule = false;
    for (const Bvh::Ref& r : hits) {
        if (r.kind == Bvh::Kind::Capsule && r.index == 0) {
            saw_capsule = true;
        }
    }
    EXPECT_TRUE(saw_capsule);
}

static void test_bvh_empty() {
    Bvh bvh;
    bvh.build({}, {});
    EXPECT_TRUE(bvh.empty());
    std::vector<Bvh::Ref> hits;
    bvh.query(Aabb{{-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}}, hits);
    EXPECT_TRUE(hits.empty());
}

int main() {
    test_bvh_empty();
    test_bvh_query_filters();

    if (g_failures == 0) {
        std::cout << "All BVH tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " assertion(s) failed.\n";
    return 1;
}
