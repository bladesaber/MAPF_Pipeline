#pragma once

#include "benchmark/GeneratorUtils.h"
#include "benchmark/Subdomain.h"
#include "benchmark/plane_slice/Types.h"
#include "grid/Types.h"

#include <vector>

namespace pipe_design::benchmark::plane_slice {

/// Cell lies on an axis-aligned slice plane.
inline bool on_plane(const grid::CellCoord& c, const Plane& plane) {
    switch (plane.axis) {
    case AxisKind::XY: return c.z == plane.fixed;
    case AxisKind::XZ: return c.y == plane.fixed;
    case AxisKind::YZ: return c.x == plane.fixed;
    }
    return false;
}

/// Fixed coordinate of `c` for the given axis kind (z / y / x).
inline int fixed_of(const grid::CellCoord& c, AxisKind axis) {
    switch (axis) {
    case AxisKind::XY: return c.z;
    case AxisKind::XZ: return c.y;
    case AxisKind::YZ: return c.x;
    }
    throw std::invalid_argument("fixed_of: invalid axis kind");
}

inline bool intersects_plane(const Subdomain& sd, const Plane& plane) {
    switch (plane.axis) {
    case AxisKind::XY:
        return plane.fixed >= sd.z_min && plane.fixed < sd.z_max;
    case AxisKind::XZ:
        return plane.fixed >= sd.y_min && plane.fixed < sd.y_max;
    case AxisKind::YZ:
        return plane.fixed >= sd.x_min && plane.fixed < sd.x_max;
    }
    return false;
}

/// Append domain-face cells of `sd` that lie on `plane`.
inline void append_face_cells_on_plane(const Subdomain& sd,
                                       const Plane& plane,
                                       int size_x, int size_y, int size_z,
                                       std::vector<grid::CellCoord>& out) {
    if (!intersects_plane(sd, plane)) {
        return;
    }

    switch (plane.axis) {
    case AxisKind::XY: {
        const int z = plane.fixed;
        for (int y = sd.y_min; y < sd.y_max; ++y) {
            for (int x = sd.x_min; x < sd.x_max; ++x) {
                const grid::CellCoord c = Subdomain::cell_at(x, y, z);
                if (benchmark::on_face(c, size_x, size_y, size_z)) {
                    out.push_back(c);
                }
            }
        }
        break;
    }
    case AxisKind::XZ: {
        const int y = plane.fixed;
        for (int z = sd.z_min; z < sd.z_max; ++z) {
            for (int x = sd.x_min; x < sd.x_max; ++x) {
                const grid::CellCoord c = Subdomain::cell_at(x, y, z);
                if (benchmark::on_face(c, size_x, size_y, size_z)) {
                    out.push_back(c);
                }
            }
        }
        break;
    }
    case AxisKind::YZ: {
        const int x = plane.fixed;
        for (int z = sd.z_min; z < sd.z_max; ++z) {
            for (int y = sd.y_min; y < sd.y_max; ++y) {
                const grid::CellCoord c = Subdomain::cell_at(x, y, z);
                if (benchmark::on_face(c, size_x, size_y, size_z)) {
                    out.push_back(c);
                }
            }
        }
        break;
    }
    }
}

}  // namespace pipe_design::benchmark::plane_slice
