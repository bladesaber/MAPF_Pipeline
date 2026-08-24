#include "benchmark/corridor_rack/CorridorRackGenerator.h"

#include "benchmark/GeneratorUtils.h"

#include <cstdint>
#include <vector>

namespace pipe_design::benchmark::corridor_rack {

CorridorRackGenerator::CorridorRackGenerator(GenParams params)
    : params_(std::move(params)) {}

bool CorridorRackGenerator::generate(Instance& out) {
    out.obstacle_params = static_cast<const ObstacleGenParams&>(params_);
    density_ = 0.f;
    eligible0_ = 0;
    dilate_steps_ = 0;

    const int sx = out.params.size_x;
    const int sy = out.params.size_y;
    const int sz = out.params.size_z;
    const float res = benchmark::GenParams::resolution;
    if (sx <= 0 || sy <= 0 || sz <= 0) {
        out.reason = "corridor_rack: empty domain";
        return false;
    }

    dilate_steps_ = params_.dilate_steps;
    if (dilate_steps_ < 0) {
        dilate_steps_ = params_.corridor_margin;
    }
    if (dilate_steps_ < 0) {
        out.reason = "corridor_rack: dilate_steps < 0";
        return false;
    }

    const std::size_t n = static_cast<std::size_t>(sx) *
                          static_cast<std::size_t>(sy) *
                          static_cast<std::size_t>(sz);
    // 0 = obstacle still to mesh; 1 = free channel / existing obstacle / covered.
    std::vector<std::uint8_t> remain(n, 0);
    mark_corridors_blocked(out.groups, dilate_steps_, sx, sy, sz, remain);
    std::int64_t corridor_vol = 0;
    for (std::uint8_t cell : remain) {
        if (cell) {
            ++corridor_vol;
        }
    }
    mark_obstacles_blocked(out, sx, sy, sz, remain);
    eligible0_ = 0;
    for (std::uint8_t cell : remain) {
        if (!cell) {
            ++eligible0_;
        }
    }
    if (eligible0_ == 0) {
        density_ = static_cast<float>(n - corridor_vol) / static_cast<float>(n);
        out.obstacle_params.density_target = density_;
        return true;
    }
    
    const auto flat = [&](int x, int y, int z) {
        return static_cast<std::size_t>((z * sy + y) * sx + x);
    };
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

    for (int z = 0; z < sz; ++z) {
        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                if (remain[flat(x, y, z)]) {
                    continue;
                }
                int x1 = x;
                while (x1 + 1 < sx && !remain[flat(x1 + 1, y, z)]) {
                    ++x1;
                }
                int y1 = y;
                while (y1 + 1 < sy && solid(x, x1, y, y1 + 1, z, z)) {
                    ++y1;
                }
                int z1 = z;
                while (z1 + 1 < sz && solid(x, x1, y, y1, z, z1 + 1)) {
                    ++z1;
                }
                for (int zz = z; zz <= z1; ++zz) {
                    for (int yy = y; yy <= y1; ++yy) {
                        for (int xx = x; xx <= x1; ++xx) {
                            remain[flat(xx, yy, zz)] = 1;
                        }
                    }
                }
                out.aabbs.push_back(geom::AabbObstacle{
                    geom::Vec3{static_cast<float>(x) * res,
                               static_cast<float>(y) * res,
                               static_cast<float>(z) * res},
                    geom::Vec3{static_cast<float>(x1 + 1) * res,
                               static_cast<float>(y1 + 1) * res,
                               static_cast<float>(z1 + 1) * res}});
            }
        }
    }

    density_ = static_cast<float>(n - corridor_vol) / static_cast<float>(n);
    out.obstacle_params.density_target = density_;
    return true;
}

}  // namespace pipe_design::benchmark::corridor_rack
