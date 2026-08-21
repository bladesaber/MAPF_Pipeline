#pragma once

#include "benchmark/GeneratorTypes.h"
#include "geom/Types.h"
#include "grid/ObstacleChain.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"
#include "planner/low/Path.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
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

}  // namespace pipe_design::benchmark
