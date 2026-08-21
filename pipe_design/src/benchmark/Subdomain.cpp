#include "benchmark/Subdomain.h"

#include <algorithm>
#include <cstddef>

namespace pipe_design::benchmark {

Subdomain Subdomain::make(int ix, int iy, int iz, int n,
                          int size_x, int size_y, int size_z) {
    Subdomain s;
    s.ix = ix;
    s.iy = iy;
    s.iz = iz;
    // Integer partition: (i * size) / n … ((i+1) * size) / n (half-open).
    // When size % n != 0, slab widths differ by at most 1; cells 0..size-1
    // are still fully covered. Do NOT use i * (size / n) — that drops a tail.
    s.x_min = ix * size_x / n;
    s.x_max = (ix + 1) * size_x / n;
    s.y_min = iy * size_y / n;
    s.y_max = (iy + 1) * size_y / n;
    s.z_min = iz * size_z / n;
    s.z_max = (iz + 1) * size_z / n;
    return s;
}

bool Subdomain::is_surface(int n) const {
    return ix == 0 || ix == n - 1 || iy == 0 || iy == n - 1 || iz == 0 ||
           iz == n - 1;
}

bool Subdomain::contains_cell(const grid::CellCoord& c) const {
    return c.x >= x_min && c.x < x_max && c.y >= y_min && c.y < y_max &&
           c.z >= z_min && c.z < z_max;
}

std::vector<Subdomain> make_subdomains(int n, int size_x, int size_y,
                                       int size_z) {
    std::vector<Subdomain> out;
    out.reserve(static_cast<std::size_t>(n * n * n));
    for (int iz = 0; iz < n; ++iz) {
        for (int iy = 0; iy < n; ++iy) {
            for (int ix = 0; ix < n; ++ix) {
                out.push_back(Subdomain::make(ix, iy, iz, n, size_x, size_y,
                                              size_z));
            }
        }
    }
    return out;
}

std::size_t subdomain_index(int n, int size_x, int size_y, int size_z,
                            const grid::CellCoord& c) {
    const int ix = std::min(n - 1, c.x * n / size_x);
    const int iy = std::min(n - 1, c.y * n / size_y);
    const int iz = std::min(n - 1, c.z * n / size_z);
    return static_cast<std::size_t>(ix + n * (iy + n * iz));
}

}  // namespace pipe_design::benchmark
