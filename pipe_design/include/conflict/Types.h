#pragma once

#include "geom/Types.h"
#include "topology/Types.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace pipe_design::conflict {

/// One sphere sample on a **normalized** tube (line-discretized capsule).
/// Independent of the original `planner::Path` cells — normalize copies geometry out.
struct SampledPoint {
    geom::Vec3 center;
    float radius = 0.f;

    /// TODO: add one more parameter to indicate whether its sampled or cell center, so we need to change name of this class

    SampledPoint() = default;
    SampledPoint(geom::Vec3 center_, float radius_) : center(center_), radius(radius_) {}
};

/// Normalized path as an ordered sphere list (does not mutate Task / Path results).
///
/// Consecutive `samples` should be farther apart than `PathGraph`'s merge `eps`
/// so junction merge cannot collapse the chain.
struct SampledPath {
    topology::GroupId group_id;
    std::string task_id;  // optional; empty = unspecified
    std::vector<SampledPoint> samples;

    SampledPath() = default;
    SampledPath(topology::GroupId group_id_, std::string task_id_ = {})
        : group_id(std::move(group_id_))
        , task_id(std::move(task_id_)) {}

    bool empty() const { return samples.empty(); }
    std::size_t size() const { return samples.size(); }
};

/// One group–group geometric conflict (`[V]` atomic evidence).
///
/// Sphere-centric: witnesses are two `SampledPoint`s.
/// `deficit` is geometric; `score` is a ranking cache filled by `ConflictAnalyzer::score`
/// (select reads it — does not recompute).
struct Conflict {
    topology::GroupId group_a;
    topology::GroupId group_b;

    /// Optional Task ids (`topology::make_task_id`); empty = unspecified.
    std::string task_a;
    std::string task_b;

    /// Witness spheres (may differ in space — radius clearance, not shared-cell MAPF).
    SampledPoint sample_a;
    SampledPoint sample_b;

    /// Penetration: `max(0, r_a + r_b - center_distance)`.
    float deficit = 0.f;

    /// Ranking cache (higher = more urgent). Filled by Analyzer; used by select_*.
    float score = 0.f;

    Conflict() = default;

    Conflict(topology::GroupId group_a_,
             topology::GroupId group_b_,
             SampledPoint sample_a_,
             SampledPoint sample_b_,
             float deficit_ = 0.f)
        : group_a(std::move(group_a_))
        , group_b(std::move(group_b_))
        , sample_a(std::move(sample_a_))
        , sample_b(std::move(sample_b_))
        , deficit(deficit_) {}

    float needed_clearance() const { return sample_a.radius + sample_b.radius; }

    bool involves(const topology::GroupId& group_id) const {
        return group_a == group_id || group_b == group_id;
    }

    bool involves_task(const std::string& task_id) const {
        if (task_id.empty()) {
            return false;
        }
        return task_a == task_id || task_b == task_id;
    }

    bool same_groups(const Conflict& other) const {
        return (group_a == other.group_a && group_b == other.group_b) ||
               (group_a == other.group_b && group_b == other.group_a);
    }

    void swap_sides() {
        using std::swap;
        swap(group_a, group_b);
        swap(task_a, task_b);
        swap(sample_a, sample_b);
    }
};

/// Connected set of atomic Conflicts (path-graph adjacency).
///
/// May involve **more than two** groups when hits form one connected component
/// (e.g. A–B and B–C bands merge). `groups` lists every group touched.
/// `score` is a ranking cache filled by Analyzer; select reads it.
struct ConflictCluster {
    std::vector<topology::GroupId> groups;
    std::vector<Conflict> conflicts;

    /// Ranking cache (higher = more urgent). Filled by Analyzer; used by select_*.
    float score = 0.f;

    ConflictCluster() = default;

    explicit ConflictCluster(std::vector<topology::GroupId> groups_)
        : groups(std::move(groups_)) {}

    bool empty() const { return conflicts.empty(); }
    std::size_t size() const { return conflicts.size(); }

    bool involves(const topology::GroupId& group_id) const {
        return std::find(groups.begin(), groups.end(), group_id) != groups.end();
    }

    float max_deficit() const {
        float m = 0.f;
        for (const Conflict& c : conflicts) {
            m = std::max(m, c.deficit);
        }
        return m;
    }

    void add(Conflict c) {
        auto ensure = [&](const topology::GroupId& g) {
            if (!involves(g)) {
                groups.push_back(g);
            }
        };
        ensure(c.group_a);
        ensure(c.group_b);
        conflicts.push_back(std::move(c));
    }
};

}  // namespace pipe_design::conflict
