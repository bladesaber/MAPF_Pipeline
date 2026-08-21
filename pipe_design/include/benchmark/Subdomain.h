#pragma once

#include "grid/Types.h"

#include <cstddef>
#include <vector>

namespace pipe_design::benchmark {

/// One block of the N×N×N partition (cell AABB, half-open maxima).
/// Shared by generators that use subdomains (ROADMAP Method 1 / 2).
struct Subdomain {
    int ix = 0;
    int iy = 0;
    int iz = 0;
    int x_min = 0;
    int x_max = 0;  // exclusive
    int y_min = 0;
    int y_max = 0;
    int z_min = 0;
    int z_max = 0;

    static Subdomain make(int ix, int iy, int iz, int n,
                          int size_x, int size_y, int size_z);

    bool is_surface(int n) const;
    bool contains_cell(const grid::CellCoord& c) const;

    /// Cell at (x,y,z); must lie inside this box (unchecked).
    static grid::CellCoord cell_at(int x, int y, int z) {
        return {x, y, z};
    }
};

/// Full N×N×N partition of the domain (index order: ix fastest, then iy, iz).
std::vector<Subdomain> make_subdomains(int n, int size_x, int size_y, int size_z);

/// Index into `make_subdomains` result for cell `c` (same ix + n*(iy + n*iz) order).
std::size_t subdomain_index(int n, int size_x, int size_y, int size_z,
                            const grid::CellCoord& c);

}  // namespace pipe_design::benchmark
