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
    if (bvh_.empty()) {
        return clearance_cap_;
    }

    geom::Aabb box{point, point};
    box.expand(clearance_cap_);

    std::vector<geom::Bvh::Ref> hits;
    bvh_.query(box, hits);

    float best = clearance_cap_;
    for (const geom::Bvh::Ref& ref : hits) {
        if (ref.kind == geom::ObstacleKind::Sphere) {
            best = std::min(
                best,
                geom::signed_clearance(point, spheres_[static_cast<std::size_t>(ref.index)]));
        } else if (ref.kind == geom::ObstacleKind::Capsule) {
            best = std::min(
                best,
                geom::signed_clearance(point, capsules_[static_cast<std::size_t>(ref.index)]));
        } else {
            best = std::min(
                best,
                geom::signed_clearance(point, aabbs_[static_cast<std::size_t>(ref.index)]));
        }
    }
    return best;
}

bool StaticClearanceGrid::is_free(const geom::Vec3& point, float pipe_radius) const {
    if (!in_bounds(point)) {
        return false;
    }
    if (bvh_.empty()) {
        return true;
    }

    geom::Aabb box{point, point};
    box.expand(pipe_radius);

    std::vector<geom::Bvh::Ref> hits;
    bvh_.query(box, hits);

    for (const geom::Bvh::Ref& ref : hits) {
        if (ref.kind == geom::ObstacleKind::Sphere) {
            if (geom::signed_clearance(point, spheres_[static_cast<std::size_t>(ref.index)]) <
                pipe_radius) {
                return false;
            }
        } else if (ref.kind == geom::ObstacleKind::Capsule) {
            if (geom::signed_clearance(point, capsules_[static_cast<std::size_t>(ref.index)]) <
                pipe_radius) {
                return false;
            }
        } else if (
            geom::signed_clearance(point, aabbs_[static_cast<std::size_t>(ref.index)]) <
            pipe_radius) {
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
    const int32_t cheb = std::max(adx, std::max(ady, adz));
    if (cheb > 1) {
        return is_move_free(cell_center(from), cell_center(to), pipe_radius);
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
    if (bvh_.empty()) {
        return true;
    }

    const geom::Vec3 delta = to - from;
    const float len = delta.norm();
    // Long segments get a huge query AABB; split so each piece stays local in the BVH.
    const float max_piece = std::max(4.f * resolution_, 1e-3f);
    const int pieces = (len <= max_piece)
        ? 1
        : std::max(2, static_cast<int>(std::ceil(static_cast<double>(len / max_piece))));

    std::vector<geom::Bvh::Ref> hits;
    for (int i = 0; i < pieces; ++i) {
        const float t0 = static_cast<float>(i) / static_cast<float>(pieces);
        const float t1 = static_cast<float>(i + 1) / static_cast<float>(pieces);
        const geom::Vec3 a = from + delta * t0;
        const geom::Vec3 b = from + delta * t1;

        hits.clear();
        bvh_.query(geom::aabb_of_segment(a, b, pipe_radius), hits);

        for (const geom::Bvh::Ref& ref : hits) {
            if (ref.kind == geom::ObstacleKind::Sphere) {
                if (geom::signed_clearance(a, b, spheres_[static_cast<std::size_t>(ref.index)]) <
                    pipe_radius) {
                    return false;
                }
            } else if (ref.kind == geom::ObstacleKind::Capsule) {
                if (geom::signed_clearance(a, b, capsules_[static_cast<std::size_t>(ref.index)]) <
                    pipe_radius) {
                    return false;
                }
            } else if (
                geom::signed_clearance(a, b, aabbs_[static_cast<std::size_t>(ref.index)]) <
                pipe_radius) {
                return false;
            }
        }
    }
    return true;
}

void StaticClearanceGrid::clear() {
    std::fill(field_.begin(), field_.end(), clearance_cap_);
    spheres_.clear();
    capsules_.clear();
    aabbs_.clear();
    bvh_.clear();
}

void StaticClearanceGrid::bake(
    const std::vector<geom::SphereObstacle>& spheres,
    const std::vector<geom::CapsuleObstacle>& capsules,
    const std::vector<geom::AabbObstacle>& aabbs) {
    clear();
    spheres_ = spheres;
    capsules_ = capsules;
    aabbs_ = aabbs;
    for (const auto& sphere : spheres_) {
        stamp_sphere(sphere);
    }
    for (const auto& capsule : capsules_) {
        stamp_capsule(capsule);
    }
    for (const auto& aabb : aabbs_) {
        stamp_aabb(aabb);
    }
    bvh_.build(spheres_, capsules_, aabbs_);
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

void StaticClearanceGrid::stamp_aabb(const geom::AabbObstacle& aabb) {
    const float influence = clearance_cap_;

    const int min_x =
        static_cast<int>(std::floor((aabb.mn.x - influence) / resolution_));
    const int max_x =
        static_cast<int>(std::floor((aabb.mx.x + influence) / resolution_));
    const int min_y =
        static_cast<int>(std::floor((aabb.mn.y - influence) / resolution_));
    const int max_y =
        static_cast<int>(std::floor((aabb.mx.y + influence) / resolution_));
    const int min_z =
        static_cast<int>(std::floor((aabb.mn.z - influence) / resolution_));
    const int max_z =
        static_cast<int>(std::floor((aabb.mx.z + influence) / resolution_));

    for (int z = min_z; z <= max_z; ++z) {
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                const CellCoord cell{x, y, z};
                if (!in_bounds(cell)) {
                    continue;
                }
                const float clr = geom::signed_clearance(cell_center(cell), aabb);
                if (clr < clearance_cap_) {
                    lower_clearance(cell, std::min(clearance_cap_, std::max(0.f, clr)));
                }
            }
        }
    }
}

}  // namespace pipe_design::grid
