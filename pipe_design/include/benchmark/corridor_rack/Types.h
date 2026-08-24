#pragma once

#include "benchmark/GeneratorTypes.h"

namespace pipe_design::benchmark::corridor_rack {

/// CorridorRack: dilate witnesses by `dilate_steps`, complement → large AABBs
/// (ROADMAP §2.3). Freeness knob is `dilate_steps`; `density_target` is ignored
/// (derived ρ written back into `obstacle_params.density_target`).
struct GenParams : ObstacleGenParams {
    /// Chebyshev dilate of witness cells → free channel half-width.
    /// If < 0, use `corridor_margin` (or `radius_cells()` if that is also < 0).
    int dilate_steps = -1;
};

}  // namespace pipe_design::benchmark::corridor_rack
