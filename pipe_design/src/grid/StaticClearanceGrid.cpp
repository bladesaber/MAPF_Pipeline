#include "grid/StaticClearanceGrid.h"

#include "geom/Geometry.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pipe_design::grid {

StaticClearanceGrid::StaticClearanceGrid(
    int size_x,
    int size_y,
    int size_z,
    float resolution,
    float clearance_cap)
    : size_x_(size_x),
      size_y_(size_y),
      size_z_(size_z),
      resolution_(resolution),
      clearance_cap_(clearance_cap),
      slice_size_(size_x * size_y) {
    if (size_x <= 0 || size_y <= 0 || size_z <= 0) {
        throw std::invalid_argument("StaticClearanceGrid: sizes must be positive");
    }
    if (resolution <= 0.f) {
        throw std::invalid_argument("StaticClearanceGrid: resolution must be positive");
    }
    if (clearance_cap <= 0.f) {
        throw std::invalid_argument("StaticClearanceGrid: clearance_cap must be positive");
    }
    field_.assign(static_cast<std::size_t>(slice_size_ * size_z_), clearance_cap_);
}

bool StaticClearanceGrid::in_bounds(const CellCoord& cell) const {
    return cell.x >= 0 && cell.x < size_x_ && cell.y >= 0 && cell.y < size_y_ &&
           cell.z >= 0 && cell.z < size_z_;
}

bool StaticClearanceGrid::in_bounds(const geom::Vec3& point) const {
    const float max_x = static_cast<float>(size_x_) * resolution_;
    const float max_y = static_cast<float>(size_y_) * resolution_;
    const float max_z = static_cast<float>(size_z_) * resolution_;
    return point.x >= 0.f && point.x < max_x && point.y >= 0.f && point.y < max_y &&
           point.z >= 0.f && point.z < max_z;
}

geom::Vec3 StaticClearanceGrid::cell_center(const CellCoord& cell) const {
    return {
        (static_cast<float>(cell.x) + 0.5f) * resolution_,
        (static_cast<float>(cell.y) + 0.5f) * resolution_,
        (static_cast<float>(cell.z) + 0.5f) * resolution_,
    };
}

float StaticClearanceGrid::clearance(const CellCoord& cell) const {
    if (!in_bounds(cell)) {
        return 0.f;
    }
    return field_[flat_index(cell)];
}

float StaticClearanceGrid::clearance(const geom::Vec3& point) const {
    if (!in_bounds(point)) {
        return 0.f;
    }
    float best = clearance_cap_;
    for (const auto& sphere : spheres_) {
        best = std::min(best, geom::signed_clearance(point, sphere));
    }
    for (const auto& capsule : capsules_) {
        best = std::min(best, geom::signed_clearance(point, capsule));
    }
    return best;
}

bool StaticClearanceGrid::is_free(const geom::Vec3& point, float pipe_radius) const {
    if (!in_bounds(point)) {
        return false;
    }
    for (const auto& sphere : spheres_) {
        if (geom::signed_clearance(point, sphere) < pipe_radius) {
            return false;
        }
    }
    for (const auto& capsule : capsules_) {
        if (geom::signed_clearance(point, capsule) < pipe_radius) {
            return false;
        }
    }
    return true;
}

bool StaticClearanceGrid::is_move_free(
    const CellCoord& from,
    const CellCoord& to,
    float pipe_radius) const {
    if (from == to) {
        throw std::invalid_argument("StaticClearanceGrid::is_move_free: from == to");
    }

    const int32_t adx = std::abs(to.x - from.x);
    const int32_t ady = std::abs(to.y - from.y);
    const int32_t adz = std::abs(to.z - from.z);
    if (std::max(adx, std::max(ady, adz)) != 1) {
        throw std::invalid_argument(
            "StaticClearanceGrid::is_move_free: requires Chebyshev distance == 1");
    }

    const int32_t x0 = std::min(from.x, to.x);
    const int32_t x1 = std::max(from.x, to.x);
    const int32_t y0 = std::min(from.y, to.y);
    const int32_t y1 = std::max(from.y, to.y);
    const int32_t z0 = std::min(from.z, to.z);
    const int32_t z1 = std::max(from.z, to.z);

    for (int32_t z = z0; z <= z1; ++z) {
        for (int32_t y = y0; y <= y1; ++y) {
            for (int32_t x = x0; x <= x1; ++x) {
                if (x == from.x && y == from.y && z == from.z) {
                    continue;
                }
                if (!is_free(CellCoord{x, y, z}, pipe_radius)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool StaticClearanceGrid::is_move_free(
    const geom::Vec3& from,
    const geom::Vec3& to,
    float pipe_radius) const {
    if (!in_bounds(from) || !in_bounds(to)) {
        return false;
    }
    for (const auto& sphere : spheres_) {
        if (geom::signed_clearance(from, to, sphere) < pipe_radius) {
            return false;
        }
    }
    for (const auto& capsule : capsules_) {
        if (geom::signed_clearance(from, to, capsule) < pipe_radius) {
            return false;
        }
    }
    return true;
}

void StaticClearanceGrid::clear() {
    std::fill(field_.begin(), field_.end(), clearance_cap_);
    spheres_.clear();
    capsules_.clear();
}

void StaticClearanceGrid::bake(
    const std::vector<geom::SphereObstacle>& spheres,
    const std::vector<geom::CapsuleObstacle>& capsules) {
    clear();
    spheres_ = spheres;
    capsules_ = capsules;
    for (const auto& sphere : spheres_) {
        stamp_sphere(sphere);
    }
    for (const auto& capsule : capsules_) {
        stamp_capsule(capsule);
    }
}

std::size_t StaticClearanceGrid::flat_index(const CellCoord& cell) const {
    return static_cast<std::size_t>(
        cell.z * slice_size_ + cell.y * size_x_ + cell.x);
}

void StaticClearanceGrid::lower_clearance(const CellCoord& cell, float value) {
    if (!in_bounds(cell)) {
        return;
    }
    const std::size_t idx = flat_index(cell);
    if (value < field_[idx]) {
        field_[idx] = value;
    }
}

void StaticClearanceGrid::stamp_sphere(const geom::SphereObstacle& sphere) {
    // Only cells that can drop below clearance_cap need updates:
    // ||p - c|| - r < cap  <=>  ||p - c|| < r + cap
    const float influence = sphere.radius + clearance_cap_;

    const int min_x = static_cast<int>(std::floor((sphere.center.x - influence) / resolution_));
    const int max_x = static_cast<int>(std::floor((sphere.center.x + influence) / resolution_));
    const int min_y = static_cast<int>(std::floor((sphere.center.y - influence) / resolution_));
    const int max_y = static_cast<int>(std::floor((sphere.center.y + influence) / resolution_));
    const int min_z = static_cast<int>(std::floor((sphere.center.z - influence) / resolution_));
    const int max_z = static_cast<int>(std::floor((sphere.center.z + influence) / resolution_));

    for (int z = min_z; z <= max_z; ++z) {
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                const CellCoord cell{x, y, z};
                if (!in_bounds(cell)) {
                    continue;
                }
                const float clr = geom::signed_clearance(cell_center(cell), sphere);
                if (clr < clearance_cap_) {
                    lower_clearance(cell, std::min(clearance_cap_, std::max(0.f, clr)));
                }
            }
        }
    }
}

void StaticClearanceGrid::stamp_capsule(const geom::CapsuleObstacle& capsule) {
    const float influence = capsule.radius + clearance_cap_;

    const float min_wx = std::min(capsule.a.x, capsule.b.x) - influence;
    const float max_wx = std::max(capsule.a.x, capsule.b.x) + influence;
    const float min_wy = std::min(capsule.a.y, capsule.b.y) - influence;
    const float max_wy = std::max(capsule.a.y, capsule.b.y) + influence;
    const float min_wz = std::min(capsule.a.z, capsule.b.z) - influence;
    const float max_wz = std::max(capsule.a.z, capsule.b.z) + influence;

    const int min_x = static_cast<int>(std::floor(min_wx / resolution_));
    const int max_x = static_cast<int>(std::floor(max_wx / resolution_));
    const int min_y = static_cast<int>(std::floor(min_wy / resolution_));
    const int max_y = static_cast<int>(std::floor(max_wy / resolution_));
    const int min_z = static_cast<int>(std::floor(min_wz / resolution_));
    const int max_z = static_cast<int>(std::floor(max_wz / resolution_));

    for (int z = min_z; z <= max_z; ++z) {
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                const CellCoord cell{x, y, z};
                if (!in_bounds(cell)) {
                    continue;
                }
                const float clr = geom::signed_clearance(cell_center(cell), capsule);
                if (clr < clearance_cap_) {
                    lower_clearance(cell, std::min(clearance_cap_, std::max(0.f, clr)));
                }
            }
        }
    }
}

}  // namespace pipe_design::grid
