#pragma once

#include "benchmark/GeneratorTypes.h"
#include "geom/Geometry.h"
#include "geom/Types.h"
#include "grid/ObstacleChain.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"
#include "planner/low/Path.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace pipe_design::benchmark {

/// Chebyshev distance between two cells: max(|dx|,|dy|,|dz|).
inline int chebyshev(const grid::CellCoord& a, const grid::CellCoord& b) {
    return std::max({std::abs(a.x - b.x), std::abs(a.y - b.y), std::abs(a.z - b.z)});
}

/// Chebyshev length of a cell step / delta.
inline int chebyshev(const grid::CellCoord& d) {
    return std::max({std::abs(d.x), std::abs(d.y), std::abs(d.z)});
}

/// Domain-boundary cell (any of x/y/z at 0 or size-1).
inline bool on_face(const grid::CellCoord& c,
                    int size_x, int size_y, int size_z) {
    return c.x == 0 || c.x == size_x - 1 || c.y == 0 || c.y == size_y - 1 ||
           c.z == 0 || c.z == size_z - 1;
}

/// Map core-grid corridors into the full domain: offset cells by `m`, then extend
/// each face terminal outward by `m` cells to the real surface (stub segments).
/// `inst.params.size_*` must already be the **full** domain; paths/terminals are
/// still in core coordinates `[0, size-2m)`.
inline void expand_shell(Instance& inst, int m) {
    if (m <= 0) {
        return;
    }
    const float res = GenParams::resolution;
    const int core_x = inst.params.size_x - 2 * m;
    const int core_y = inst.params.size_y - 2 * m;
    const int core_z = inst.params.size_z - 2 * m;

    for (GroupCorridors& g : inst.groups) {
        // Core → full frame: planning used origin at the inner box, so every
        // witness cell must be translated by +m (not a path-length extension).
        for (planner::Path& path : g.paths) {
            for (grid::CellCoord& c : path.cells) {
                c.x += m;
                c.y += m;
                c.z += m;
            }
        }
        for (topology::Terminal& t : g.terminals) {
            const grid::CellCoord core{
                static_cast<std::int32_t>(std::floor(t.pos.x / res)),
                static_cast<std::int32_t>(std::floor(t.pos.y / res)),
                static_cast<std::int32_t>(std::floor(t.pos.z / res))};
            int dx = 0, dy = 0, dz = 0;
            if (core.x == 0) {
                dx = -1;
            } else if (core.x == core_x - 1) {
                dx = 1;
            } else if (core.y == 0) {
                dy = -1;
            } else if (core.y == core_y - 1) {
                dy = 1;
            } else if (core.z == 0) {
                dz = -1;
            } else if (core.z == core_z - 1) {
                dz = 1;
            }
            const grid::CellCoord inner{core.x + m, core.y + m, core.z + m};
            const grid::CellCoord outer{inner.x + m * dx, inner.y + m * dy,
                                        inner.z + m * dz};
            t.pos = {(static_cast<float>(outer.x) + 0.5f) * res,
                     (static_cast<float>(outer.y) + 0.5f) * res,
                     (static_cast<float>(outer.z) + 0.5f) * res};
            if (dx == 0 && dy == 0 && dz == 0) {
                continue;
            }
            for (planner::Path& path : g.paths) {
                if (path.cells.empty()) {
                    continue;
                }
                if (path.cells.front() == inner) {
                    std::vector<grid::CellCoord> cells;
                    cells.reserve(path.cells.size() + static_cast<std::size_t>(m));
                    grid::CellCoord c = outer;
                    for (int i = 0; i < m; ++i) {
                        cells.push_back(c);
                        c.x -= dx;
                        c.y -= dy;
                        c.z -= dz;
                    }
                    cells.insert(cells.end(), path.cells.begin(), path.cells.end());
                    path.cells = std::move(cells);
                    path.cost += static_cast<double>(m);
                    break;
                }
                if (path.cells.back() == inner) {
                    grid::CellCoord c = inner;
                    for (int i = 0; i < m; ++i) {
                        c.x += dx;
                        c.y += dy;
                        c.z += dz;
                        path.cells.push_back(c);
                    }
                    path.cost += static_cast<double>(m);
                    break;
                }
            }
        }
    }
}

/// True if Chebyshev from `c` to every existing terminal is > `center_gap`
/// (pass `GenParams::terminal_center_gap()`).
inline bool terminal_clearance_ok(const grid::CellCoord& c,
                                  const std::vector<GroupCorridors>& groups,
                                  const grid::StaticClearanceGrid& grid,
                                  int center_gap) {
    for (const GroupCorridors& gr : groups) {
        for (const auto& t : gr.terminals) {
            const grid::CellCoord o = grid.world_to_cell(t.pos);
            if (chebyshev(c, o) < center_gap) {
                return false;
            }
        }
    }
    return true;
}

/// True if two pipe spheres at cell centers conflict: dist < radius_a + radius_b.
inline bool cells_radius_conflict(const grid::CellCoord& a, float radius_a,
                                  const grid::CellCoord& b, float radius_b,
                                  const grid::StaticClearanceGrid& grid) {
    const geom::Vec3 ca = grid.cell_center(a);
    const geom::Vec3 cb = grid.cell_center(b);
    const float dx = ca.x - cb.x;
    const float dy = ca.y - cb.y;
    const float dz = ca.z - cb.z;
    const float rsum = radius_a + radius_b;
    return (dx * dx + dy * dy + dz * dz) < (rsum * rsum);
}

/// True if a candidate terminal of `radius` does not radius-conflict with any
/// existing terminal (any group).
inline bool terminal_radius_ok(const grid::CellCoord& c, float radius,
                               const std::vector<GroupCorridors>& groups,
                               const grid::StaticClearanceGrid& grid) {
    for (const GroupCorridors& gr : groups) {
        for (const auto& t : gr.terminals) {
            const grid::CellCoord o = grid.world_to_cell(t.pos);
            if (cells_radius_conflict(c, radius, o, t.radius, grid)) {
                return false;
            }
        }
    }
    return true;
}

/// Append a discrete path into an ObstacleChain (sphere / capsule runs).
inline std::shared_ptr<const grid::ObstacleChain> bake_path(
    std::shared_ptr<const grid::ObstacleChain> chain,
    const grid::StaticClearanceGrid& sg,
    const planner::Path& path,
    float radius) {
    if (path.cells.empty()) {
        return chain;
    }
    if (!chain) {
        chain = grid::ObstacleChain::make_root();
    }
    std::vector<geom::SphereObstacle> spheres;
    std::vector<geom::CapsuleObstacle> capsules;
    if (path.cells.size() == 1) {
        spheres.push_back({sg.cell_center(path.cells[0]), radius});
        return chain->extend(spheres, capsules);
    }
    std::size_t i = 0;
    while (i + 1 < path.cells.size()) {
        const grid::CellCoord dir = path.cells[i + 1] - path.cells[i];
        std::size_t j = i + 1;
        while (j + 1 < path.cells.size() &&
               (path.cells[j + 1] - path.cells[j]) == dir) {
            ++j;
        }
        const geom::Vec3 a = sg.cell_center(path.cells[i]);
        const geom::Vec3 b = sg.cell_center(path.cells[j]);
        const grid::CellCoord d = path.cells[j] - path.cells[i];
        if (chebyshev(d) == 1) {
            spheres.push_back({b, radius});
        } else {
            capsules.push_back({a, b, radius});
        }
        i = j;
    }
    return chain->extend(spheres, capsules);
}

/// Mark Chebyshev-thickened witness corridor cells as blocked (shared by
/// PillarField / CorridorRack obstacle paint). `blocked` must already be sized
/// `size_x * size_y * size_z` (index `(z * size_y + y) * size_x + x`).
inline void mark_corridors_blocked(const std::vector<GroupCorridors>& groups,
                                   int margin,
                                   int size_x,
                                   int size_y,
                                   int size_z,
                                   std::vector<std::uint8_t>& blocked) {
    const auto in_bounds = [&](int x, int y, int z) {
        return x >= 0 && x < size_x && y >= 0 && y < size_y && z >= 0 &&
               z < size_z;
    };
    const auto flat = [&](int x, int y, int z) {
        return static_cast<std::size_t>((z * size_y + y) * size_x + x);
    };
    for (const GroupCorridors& g : groups) {
        for (const planner::Path& path : g.paths) {
            for (const grid::CellCoord& c : path.cells) {
                for (int dz = -margin; dz <= margin; ++dz) {
                    for (int dy = -margin; dy <= margin; ++dy) {
                        for (int dx = -margin; dx <= margin; ++dx) {
                            if (chebyshev(grid::CellCoord{dx, dy, dz}) > margin) {
                                continue;
                            }
                            const int x = c.x + dx;
                            const int y = c.y + dy;
                            const int z = c.z + dz;
                            if (in_bounds(x, y, z)) {
                                blocked[flat(x, y, z)] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Mark cells occupied by existing instance obstacles (shell AABBs, etc.).
/// Call after `mark_corridors_blocked` so witness tubes stay protected.
inline void mark_obstacles_blocked(const Instance& inst,
                                   int size_x,
                                   int size_y,
                                   int size_z,
                                   std::vector<std::uint8_t>& blocked) {
    const float res = GenParams::resolution;
    const auto in_bounds = [&](int x, int y, int z) {
        return x >= 0 && x < size_x && y >= 0 && y < size_y && z >= 0 &&
               z < size_z;
    };
    const auto flat = [&](int x, int y, int z) {
        return static_cast<std::size_t>((z * size_y + y) * size_x + x);
    };
    const auto mark_cell = [&](int x, int y, int z) {
        if (in_bounds(x, y, z)) {
            blocked[flat(x, y, z)] = 1;
        }
    };
    const auto cell_center = [&](int x, int y, int z) {
        return geom::Vec3{(static_cast<float>(x) + 0.5f) * res,
                          (static_cast<float>(y) + 0.5f) * res,
                          (static_cast<float>(z) + 0.5f) * res};
    };

    for (const geom::AabbObstacle& box : inst.aabbs) {
        const int x0 = std::max(
            0, static_cast<int>(std::floor(box.mn.x / res)));
        const int x1 = std::min(
            size_x - 1, static_cast<int>(std::ceil(box.mx.x / res)) - 1);
        const int y0 = std::max(
            0, static_cast<int>(std::floor(box.mn.y / res)));
        const int y1 = std::min(
            size_y - 1, static_cast<int>(std::ceil(box.mx.y / res)) - 1);
        const int z0 = std::max(
            0, static_cast<int>(std::floor(box.mn.z / res)));
        const int z1 = std::min(
            size_z - 1, static_cast<int>(std::ceil(box.mx.z / res)) - 1);
        for (int z = z0; z <= z1; ++z) {
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    if (geom::signed_clearance(cell_center(x, y, z), box) <=
                        0.f) {
                        mark_cell(x, y, z);
                    }
                }
            }
        }
    }

    for (const geom::SphereObstacle& s : inst.spheres) {
        const float reach = s.radius;
        const int x0 = std::max(
            0, static_cast<int>(std::floor((s.center.x - reach) / res)));
        const int x1 = std::min(
            size_x - 1,
            static_cast<int>(std::ceil((s.center.x + reach) / res)) - 1);
        const int y0 = std::max(
            0, static_cast<int>(std::floor((s.center.y - reach) / res)));
        const int y1 = std::min(
            size_y - 1,
            static_cast<int>(std::ceil((s.center.y + reach) / res)) - 1);
        const int z0 = std::max(
            0, static_cast<int>(std::floor((s.center.z - reach) / res)));
        const int z1 = std::min(
            size_z - 1,
            static_cast<int>(std::ceil((s.center.z + reach) / res)) - 1);
        for (int z = z0; z <= z1; ++z) {
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    if (geom::signed_clearance(cell_center(x, y, z), s) <= 0.f) {
                        mark_cell(x, y, z);
                    }
                }
            }
        }
    }

    for (const geom::CapsuleObstacle& c : inst.capsules) {
        const float reach = c.radius;
        const int x0 = std::max(
            0, static_cast<int>(
                   std::floor((std::min(c.a.x, c.b.x) - reach) / res)));
        const int x1 = std::min(
            size_x - 1,
            static_cast<int>(
                std::ceil((std::max(c.a.x, c.b.x) + reach) / res)) - 1);
        const int y0 = std::max(
            0, static_cast<int>(
                   std::floor((std::min(c.a.y, c.b.y) - reach) / res)));
        const int y1 = std::min(
            size_y - 1,
            static_cast<int>(
                std::ceil((std::max(c.a.y, c.b.y) + reach) / res)) - 1);
        const int z0 = std::max(
            0, static_cast<int>(
                   std::floor((std::min(c.a.z, c.b.z) - reach) / res)));
        const int z1 = std::min(
            size_z - 1,
            static_cast<int>(
                std::ceil((std::max(c.a.z, c.b.z) + reach) / res)) - 1);
        for (int z = z0; z <= z1; ++z) {
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    if (geom::signed_clearance(cell_center(x, y, z), c) <= 0.f) {
                        mark_cell(x, y, z);
                    }
                }
            }
        }
    }
}

/// Append shell keep-out as few AABBs.
/// Solid band = outer `shell_obstacle_cells()`; if that is 0 (`shell_margin==0`),
/// still use a 1-cell face layer (planar domain wall) so pipe radius cannot
/// leave the domain. Carve Chebyshev `carve_margin` (must be `>= 0`). Appends
/// to `out`.
inline void append_shell_aabbs(const std::vector<GroupCorridors>& groups,
                               const GenParams& params,
                               std::vector<geom::AabbObstacle>& out) {
    if (params.carve_margin < 0) {
        throw std::invalid_argument(
            "append_shell_aabbs: carve_margin must be >= 0");
    }
    // Floor open width by pipe radius (CLI carve_margin may still be larger).
    const int carve_margin = std::max(params.carve_margin, params.radius_cells());
    // m==0 → planar boundary: outermost cell layer (degraded shell).
    const int m = std::max(params.shell_obstacle_cells(), 1);
    const int size_x = params.size_x;
    const int size_y = params.size_y;
    const int size_z = params.size_z;
    const float resolution = GenParams::resolution;
    if (size_x <= 0 || size_y <= 0 || size_z <= 0) {
        return;
    }
    const std::size_t n = static_cast<std::size_t>(size_x) *
                          static_cast<std::size_t>(size_y) *
                          static_cast<std::size_t>(size_z);
    // 0 = shell cell still to cover; 1 = core / carved / already meshed.
    std::vector<std::uint8_t> remain(n, 1);
    const auto flat = [&](int x, int y, int z) {
        return static_cast<std::size_t>((z * size_y + y) * size_x + x);
    };
    for (int z = 0; z < size_z; ++z) {
        for (int y = 0; y < size_y; ++y) {
            for (int x = 0; x < size_x; ++x) {
                if (x < m || x >= size_x - m || y < m || y >= size_y - m ||
                    z < m || z >= size_z - m) {
                    remain[flat(x, y, z)] = 0;
                }
            }
        }
    }
    // Carve into remain directly (sets corridor cells to 1).
    mark_corridors_blocked(groups, carve_margin, size_x, size_y, size_z, remain);

    // TODO: improve the performance of this function in the future
    auto solid = [&](int x0, int x1, int y0, int y1, int z0, int z1) {
        for (int z = z0; z <= z1; ++z) {
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    if (remain[flat(x, y, z)]) {
                        return false;
                    }
                }
            }
        }
        return true;
    };

    // Greedy: seed → max x-run → grow y → grow z; mark covered as 1.
    // Single-cell extent on an axis → 1-cell-thick slab (planar keep-out).
    for (int z = 0; z < size_z; ++z) {
        for (int y = 0; y < size_y; ++y) {
            for (int x = 0; x < size_x; ++x) {
                if (remain[flat(x, y, z)]) {
                    continue;
                }
                int x1 = x;
                while (x1 + 1 < size_x && !remain[flat(x1 + 1, y, z)]) {
                    ++x1;
                }
                int y1 = y;
                while (y1 + 1 < size_y && solid(x, x1, y, y1 + 1, z, z)) {
                    ++y1;
                }
                int z1 = z;
                while (z1 + 1 < size_z && solid(x, x1, y, y1, z, z1 + 1)) {
                    ++z1;
                }
                for (int zz = z; zz <= z1; ++zz) {
                    for (int yy = y; yy <= y1; ++yy) {
                        for (int xx = x; xx <= x1; ++xx) {
                            remain[flat(xx, yy, zz)] = 1;
                        }
                    }
                }
                out.push_back(geom::AabbObstacle{
                    geom::Vec3{static_cast<float>(x) * resolution,
                               static_cast<float>(y) * resolution,
                               static_cast<float>(z) * resolution},
                    geom::Vec3{static_cast<float>(x1 + 1) * resolution,
                               static_cast<float>(y1 + 1) * resolution,
                               static_cast<float>(z1 + 1) * resolution}});
            }
        }
    }
}

}  // namespace pipe_design::benchmark
