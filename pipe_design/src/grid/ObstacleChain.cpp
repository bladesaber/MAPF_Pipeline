#include "grid/ObstacleChain.h"

#include "geom/Geometry.h"

#include <limits>

namespace pipe_design::grid {

std::shared_ptr<ObstacleChain> ObstacleChain::make_root() {
    return std::shared_ptr<ObstacleChain>(new ObstacleChain());
}

std::shared_ptr<ObstacleChain> ObstacleChain::extend(const geom::SphereObstacle& sphere) const {
    auto child = std::shared_ptr<ObstacleChain>(new ObstacleChain(shared_from_this()));
    child->spheres_.push_back(sphere);
    return child;
}

std::shared_ptr<ObstacleChain> ObstacleChain::extend(const geom::CapsuleObstacle& capsule) const {
    auto child = std::shared_ptr<ObstacleChain>(new ObstacleChain(shared_from_this()));
    child->capsules_.push_back(capsule);
    return child;
}

std::shared_ptr<ObstacleChain> ObstacleChain::extend(
    const std::vector<geom::SphereObstacle>& spheres,
    const std::vector<geom::CapsuleObstacle>& capsules) const {
    auto child = std::shared_ptr<ObstacleChain>(new ObstacleChain(shared_from_this()));
    child->spheres_ = spheres;
    child->capsules_ = capsules;
    return child;
}

std::size_t ObstacleChain::total_count() const {
    std::size_t count = 0;
    for (const ObstacleChain* node = this; node != nullptr; node = node->parent_.get()) {
        count += node->local_count();
    }
    return count;
}

float ObstacleChain::clearance(const geom::Vec3& point) const {
    float best = std::numeric_limits<float>::infinity();
    for (const ObstacleChain* node = this; node != nullptr; node = node->parent_.get()) {
        for (const auto& sphere : node->spheres_) {
            best = std::min(best, geom::signed_clearance(point, sphere));
        }
        for (const auto& capsule : node->capsules_) {
            best = std::min(best, geom::signed_clearance(point, capsule));
        }
    }
    return best;
}

bool ObstacleChain::collides(const geom::Vec3& point, float pipe_radius) const {
    for (const ObstacleChain* node = this; node != nullptr; node = node->parent_.get()) {
        for (const auto& sphere : node->spheres_) {
            if (geom::signed_clearance(point, sphere) < pipe_radius) {
                return true;
            }
        }
        for (const auto& capsule : node->capsules_) {
            if (geom::signed_clearance(point, capsule) < pipe_radius) {
                return true;
            }
        }
    }
    return false;
}

bool ObstacleChain::collides(
    const geom::Vec3& from,
    const geom::Vec3& to,
    float pipe_radius) const {
    for (const ObstacleChain* node = this; node != nullptr; node = node->parent_.get()) {
        for (const auto& sphere : node->spheres_) {
            if (geom::signed_clearance(from, to, sphere) < pipe_radius) {
                return true;
            }
        }
        for (const auto& capsule : node->capsules_) {
            if (geom::signed_clearance(from, to, capsule) < pipe_radius) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace pipe_design::grid
