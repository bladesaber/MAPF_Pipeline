#pragma once

#include "geom/Types.h"
#include "grid/Types.h"

#include <memory>
#include <vector>

/// TODO: change class to Dynamic Constraint Chain
/// std::vector<geom::SphereObstacle> spheres_, std::vector<geom::CapsuleObstacle> capsules_ is not enough,
/// we need new classes which contain SphereObstacle, type[positive, negative], step_length(the time axis in normal mapf)

namespace pipe_design::grid {

/// Immutable constraint list with parent sharing (CBS-friendly COW).
///
/// Each node stores only obstacles added at this CBS branch.
/// Children keep a shared_ptr to the parent; ancestors are never mutated.
///
/// Effective obstacles(node) = node.local ∪ parent.local ∪ ... ∪ root.local
class ObstacleChain : public std::enable_shared_from_this<ObstacleChain> {
public:
    ObstacleChain(const ObstacleChain&) = delete;
    ObstacleChain& operator=(const ObstacleChain&) = delete;

    /// Empty root chain (no parent, no obstacles).
    static std::shared_ptr<ObstacleChain> make_root();

    /// Fork a child that inherits this chain and appends one sphere.
    std::shared_ptr<ObstacleChain> extend(const geom::SphereObstacle& sphere) const;

    /// Fork a child that inherits this chain and appends one capsule.
    std::shared_ptr<ObstacleChain> extend(const geom::CapsuleObstacle& capsule) const;

    /// Fork a child that inherits this chain and appends several obstacles.
    std::shared_ptr<ObstacleChain> extend(
        const std::vector<geom::SphereObstacle>& spheres,
        const std::vector<geom::CapsuleObstacle>& capsules) const;

    const ObstacleChain* parent() const { return parent_.get(); }
    const std::vector<geom::SphereObstacle>& local_spheres() const { return spheres_; }
    const std::vector<geom::CapsuleObstacle>& local_capsules() const { return capsules_; }

    /// Number of obstacles stored on this node only (not ancestors).
    std::size_t local_count() const { return spheres_.size() + capsules_.size(); }

    /// Total obstacles along the chain (root → this).
    std::size_t total_count() const;

    /// Min signed clearance to any obstacle on this chain (and ancestors).
    /// Returns +inf if the chain is empty.
    float clearance(const geom::Vec3& point) const;

    /// True if a pipe of the given radius centered at `point` collides with any chain obstacle.
    /// Early-exits on the first obstacle with clearance < pipe_radius.
    bool collides(const geom::Vec3& point, float pipe_radius) const;

    /// True if segment `from`→`to` comes within `pipe_radius` of any chain obstacle.
    bool collides(
        const geom::Vec3& from,
        const geom::Vec3& to,
        float pipe_radius) const;

private:
    ObstacleChain() = default;

    explicit ObstacleChain(std::shared_ptr<const ObstacleChain> parent)
        : parent_(std::move(parent)) {}

    std::shared_ptr<const ObstacleChain> parent_;
    std::vector<geom::SphereObstacle> spheres_;
    std::vector<geom::CapsuleObstacle> capsules_;
};

}  // namespace pipe_design::grid
