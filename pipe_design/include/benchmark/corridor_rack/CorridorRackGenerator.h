#pragma once

#include "benchmark/GeneratorTypes.h"
#include "benchmark/corridor_rack/Types.h"

#include <cstdint>

namespace pipe_design::benchmark::corridor_rack {

/// Dilate witness corridors; mesh the complement as few large AABBs.
class CorridorRackGenerator {
public:
    explicit CorridorRackGenerator(GenParams params);

    /// Appends meshed complement AABBs; preserves existing obstacles.
    /// Writes base knobs + derived density into `out.obstacle_params`. Returns false on
    /// bad input.
    bool generate(Instance& out);

    float density() const { return density_; }
    std::int64_t eligible0() const { return eligible0_; }
    int dilate_steps() const { return dilate_steps_; }

private:
    GenParams params_;
    int dilate_steps_ = 0;
    std::int64_t eligible0_ = 0;
    float density_ = 0.f;
};

}  // namespace pipe_design::benchmark::corridor_rack
