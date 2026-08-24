#pragma once

#include "benchmark/GeneratorTypes.h"
#include "benchmark/pillar_field/Types.h"
#include "grid/Types.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace pipe_design::benchmark::pillar_field {

/// Fill eligible space (outside thickened corridors) with large AABB obstacles.
class PillarFieldGenerator {
public:
    explicit PillarFieldGenerator(GenParams params);

    /// Clears `out.aabbs` only, paints until `density_target` or no progress.
    /// Writes base knobs into `out.obstacle_params`. Returns false only on bad input.
    bool generate(Instance& out);

    float density() const { return density_; }
    std::int64_t eligible0() const { return eligible0_; }

private:
    struct Box {
        int x0 = 0, y0 = 0, z0 = 0;
        int x1 = 0, y1 = 0, z1 = 0;  // inclusive cell bounds
        int volume() const {
            return (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1);
        }
        int min_edge() const {
            const int dx = x1 - x0 + 1;
            const int dy = y1 - y0 + 1;
            const int dz = z1 - z0 + 1;
            return std::min({dx, dy, dz});
        }
    };

    GenParams params_;
    std::mt19937 rng_;
    int sx_ = 0, sy_ = 0, sz_ = 0;
    float res_ = 1.f;
    std::vector<std::uint8_t> blocked_;
    /// Chebyshev distance to nearest blocked cell (0 if blocked). Sampling weight.
    std::vector<int> dist_;
    std::int64_t eligible0_ = 0;
    std::int64_t blocked_vol_ = 0;
    float density_ = 0.f;

    int idx(int x, int y, int z) const { return (z * sy_ + y) * sx_ + x; }
    bool in_bounds(int x, int y, int z) const {
        return x >= 0 && x < sx_ && y >= 0 && y < sy_ && z >= 0 && z < sz_;
    }
    bool is_blocked(int x, int y, int z) const {
        return !in_bounds(x, y, z) ||
               blocked_[static_cast<std::size_t>(idx(x, y, z))] != 0;
    }
    float sample_weight_at(int x, int y, int z) const;

    /// Multi-source Chebyshev DT from all blocked cells.
    void rebuild_dist();
    /// Lower `dist_` after new blocked cells (incremental min-update BFS).
    void update_dist_from(const std::vector<grid::CellCoord>& seeds);
    bool sample_seed(grid::CellCoord& out);
    Box greedy_expand(const grid::CellCoord& seed) const;
    bool face_free(const Box& b, int face) const;
    void commit(const Box& b, Instance& out);
};

}  // namespace pipe_design::benchmark::pillar_field
