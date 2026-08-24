#include "benchmark/pillar_field/PillarFieldGenerator.h"

#include "benchmark/GeneratorUtils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace pipe_design::benchmark::pillar_field {
namespace {

constexpr int kDistInf = std::numeric_limits<int>::max() / 4;

}  // namespace

PillarFieldGenerator::PillarFieldGenerator(GenParams params)
    : params_(std::move(params))
    , rng_(params_.seed) {}

bool PillarFieldGenerator::generate(Instance& out) {
    out.obstacle_params = static_cast<const ObstacleGenParams&>(params_);
    sx_ = out.params.size_x;
    sy_ = out.params.size_y;
    sz_ = out.params.size_z;
    res_ = benchmark::GenParams::resolution;
    density_ = 0.f;
    blocked_vol_ = 0;
    eligible0_ = 0;
    out.aabbs.clear();

    if (sx_ <= 0 || sy_ <= 0 || sz_ <= 0) {
        out.reason = "pillar_field: empty domain";
        return false;
    }
    if (params_.density_target < 0.f || params_.density_target > 1.f) {
        out.reason = "pillar_field: density_target out of [0,1]";
        return false;
    }
    if (params_.s_min < 1) {
        out.reason = "pillar_field: s_min < 1";
        return false;
    }

    const int margin = params_.corridor_margin >= 0
                           ? params_.corridor_margin
                           : out.params.radius_cells();

    const std::size_t n =
        static_cast<std::size_t>(sx_) * static_cast<std::size_t>(sy_) *
        static_cast<std::size_t>(sz_);
    blocked_.assign(n, 0);
    dist_.assign(n, 0);
    mark_corridors_blocked(out.groups, margin, sx_, sy_, sz_, blocked_);

    for (std::uint8_t v : blocked_) {
        if (v == 0) {
            ++eligible0_;
        }
    }
    if (eligible0_ == 0 || params_.density_target <= 0.f) {
        density_ = 0.f;
        return true;
    }

    rebuild_dist();

    int stagnant = 0;
    for (int round = 0; round < params_.max_rounds; ++round) {
        if (density_ >= params_.density_target) {
            break;
        }

        Box best{};
        int best_score = 0;
        for (int c = 0; c < params_.candidates_per_round; ++c) {
            grid::CellCoord seed;
            if (!sample_seed(seed)) {
                break;
            }
            Box box = greedy_expand(seed);
            if (box.min_edge() < params_.s_min) {
                continue;
            }
            const int score = box.volume();
            if (score > best_score) {
                best_score = score;
                best = box;
            }
        }

        if (best_score <= 0) {
            if (++stagnant >= 3) {
                break;
            }
            continue;
        }
        stagnant = 0;
        commit(best, out);
    }

    density_ = static_cast<float>(blocked_vol_) /
               static_cast<float>(eligible0_);
    return true;
}

void PillarFieldGenerator::rebuild_dist() {
    std::queue<grid::CellCoord> q;
    for (int z = 0; z < sz_; ++z) {
        for (int y = 0; y < sy_; ++y) {
            for (int x = 0; x < sx_; ++x) {
                const std::size_t i = static_cast<std::size_t>(idx(x, y, z));
                if (blocked_[i]) {
                    dist_[i] = 0;
                    q.push({x, y, z});
                } else {
                    dist_[i] = kDistInf;
                }
            }
        }
    }
    while (!q.empty()) {
        const grid::CellCoord c = q.front();
        q.pop();
        const int cd = dist_[static_cast<std::size_t>(idx(c.x, c.y, c.z))];
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    const int nx = c.x + dx;
                    const int ny = c.y + dy;
                    const int nz = c.z + dz;
                    if (!in_bounds(nx, ny, nz)) {
                        continue;
                    }
                    const std::size_t ni =
                        static_cast<std::size_t>(idx(nx, ny, nz));
                    const int nd = cd + 1;
                    if (nd < dist_[ni]) {
                        dist_[ni] = nd;
                        q.push({nx, ny, nz});
                    }
                }
            }
        }
    }
}

void PillarFieldGenerator::update_dist_from(
    const std::vector<grid::CellCoord>& seeds) {
    std::queue<grid::CellCoord> q;
    for (const grid::CellCoord& s : seeds) {
        const std::size_t i = static_cast<std::size_t>(idx(s.x, s.y, s.z));
        dist_[i] = 0;
        q.push(s);
    }
    while (!q.empty()) {
        const grid::CellCoord c = q.front();
        q.pop();
        const int cd = dist_[static_cast<std::size_t>(idx(c.x, c.y, c.z))];
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    const int nx = c.x + dx;
                    const int ny = c.y + dy;
                    const int nz = c.z + dz;
                    if (!in_bounds(nx, ny, nz)) {
                        continue;
                    }
                    const std::size_t ni =
                        static_cast<std::size_t>(idx(nx, ny, nz));
                    const int nd = cd + 1;
                    if (nd < dist_[ni]) {
                        dist_[ni] = nd;
                        q.push({nx, ny, nz});
                    }
                }
            }
        }
    }
}

float PillarFieldGenerator::sample_weight_at(int x, int y, int z) const {
    const int d = dist_[static_cast<std::size_t>(idx(x, y, z))];
    if (params_.dist_cap > 0) {
        return static_cast<float>(std::min(d, params_.dist_cap));
    }
    return static_cast<float>(d);
}

bool PillarFieldGenerator::sample_seed(grid::CellCoord& out) {
    // One pass builds weights; roulette reuses them (no second sample_weight_at).
    std::vector<float> w(dist_.size(), 0.f);
    double total = 0.0;
    for (int z = 0; z < sz_; ++z) {
        for (int y = 0; y < sy_; ++y) {
            for (int x = 0; x < sx_; ++x) {
                if (is_blocked(x, y, z)) {
                    continue;
                }
                const std::size_t i = static_cast<std::size_t>(idx(x, y, z));
                w[i] = sample_weight_at(x, y, z);
                total += static_cast<double>(w[i]);
            }
        }
    }
    if (total <= 0.0) {
        return false;
    }

    std::uniform_real_distribution<double> uni(0.0, total);
    double tick = uni(rng_);
    for (int z = 0; z < sz_; ++z) {
        for (int y = 0; y < sy_; ++y) {
            for (int x = 0; x < sx_; ++x) {
                if (is_blocked(x, y, z)) {
                    continue;
                }
                tick -= static_cast<double>(
                    w[static_cast<std::size_t>(idx(x, y, z))]);
                if (tick <= 0.0) {
                    out = {x, y, z};
                    return true;
                }
            }
        }
    }
    throw std::logic_error(
        "pillar_field: sample_seed roulette failed with total > 0");
}

bool PillarFieldGenerator::face_free(const Box& b, int face) const {
    if (face == 0) {
        const int x = b.x1 + 1;
        for (int z = b.z0; z <= b.z1; ++z) {
            for (int y = b.y0; y <= b.y1; ++y) {
                if (is_blocked(x, y, z)) {
                    return false;
                }
            }
        }
        return true;
    }
    if (face == 1) {
        const int x = b.x0 - 1;
        for (int z = b.z0; z <= b.z1; ++z) {
            for (int y = b.y0; y <= b.y1; ++y) {
                if (is_blocked(x, y, z)) {
                    return false;
                }
            }
        }
        return true;
    }
    if (face == 2) {
        const int y = b.y1 + 1;
        for (int z = b.z0; z <= b.z1; ++z) {
            for (int x = b.x0; x <= b.x1; ++x) {
                if (is_blocked(x, y, z)) {
                    return false;
                }
            }
        }
        return true;
    }
    if (face == 3) {
        const int y = b.y0 - 1;
        for (int z = b.z0; z <= b.z1; ++z) {
            for (int x = b.x0; x <= b.x1; ++x) {
                if (is_blocked(x, y, z)) {
                    return false;
                }
            }
        }
        return true;
    }
    if (face == 4) {
        const int z = b.z1 + 1;
        for (int y = b.y0; y <= b.y1; ++y) {
            for (int x = b.x0; x <= b.x1; ++x) {
                if (is_blocked(x, y, z)) {
                    return false;
                }
            }
        }
        return true;
    }
    const int z = b.z0 - 1;
    for (int y = b.y0; y <= b.y1; ++y) {
        for (int x = b.x0; x <= b.x1; ++x) {
            if (is_blocked(x, y, z)) {
                return false;
            }
        }
    }
    return true;
}

PillarFieldGenerator::Box PillarFieldGenerator::greedy_expand(
    const grid::CellCoord& seed) const {
    Box b{seed.x, seed.y, seed.z, seed.x, seed.y, seed.z};
    if (is_blocked(seed.x, seed.y, seed.z)) {
        return b;
    }
    if (params_.isometric) {
        // All six faces must advance together; stop when any face is blocked.
        while (true) {
            bool ok = true;
            for (int face = 0; face < 6; ++face) {
                if (!face_free(b, face)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                break;
            }
            ++b.x1;
            --b.x0;
            ++b.y1;
            --b.y0;
            ++b.z1;
            --b.z0;
        }
        return b;
    }
    // Maximal AABB: expand any free face independently.
    bool grew = true;
    while (grew) {
        grew = false;
        for (int face = 0; face < 6; ++face) {
            if (!face_free(b, face)) {
                continue;
            }
            if (face == 0) {
                ++b.x1;
            } else if (face == 1) {
                --b.x0;
            } else if (face == 2) {
                ++b.y1;
            } else if (face == 3) {
                --b.y0;
            } else if (face == 4) {
                ++b.z1;
            } else {
                --b.z0;
            }
            grew = true;
        }
    }
    return b;
}

void PillarFieldGenerator::commit(const Box& b, Instance& out) {
    std::vector<grid::CellCoord> newly;
    newly.reserve(static_cast<std::size_t>(b.volume()));
    for (int z = b.z0; z <= b.z1; ++z) {
        for (int y = b.y0; y <= b.y1; ++y) {
            for (int x = b.x0; x <= b.x1; ++x) {
                const std::size_t i = static_cast<std::size_t>(idx(x, y, z));
                if (blocked_[i] != 0) {
                    continue;
                }
                blocked_[i] = 1;
                newly.push_back({x, y, z});
            }
        }
    }
    blocked_vol_ += static_cast<std::int64_t>(newly.size());
    density_ = static_cast<float>(blocked_vol_) /
               static_cast<float>(eligible0_);

    update_dist_from(newly);

    out.aabbs.push_back(geom::AabbObstacle{
        geom::Vec3{static_cast<float>(b.x0) * res_,
                   static_cast<float>(b.y0) * res_,
                   static_cast<float>(b.z0) * res_},
        geom::Vec3{static_cast<float>(b.x1 + 1) * res_,
                   static_cast<float>(b.y1 + 1) * res_,
                   static_cast<float>(b.z1 + 1) * res_}});
}

}  // namespace pipe_design::benchmark::pillar_field
