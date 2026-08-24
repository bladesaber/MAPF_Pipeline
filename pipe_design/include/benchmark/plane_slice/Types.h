#pragma once

#include "benchmark/GeneratorTypes.h"

#include <cstdint>

namespace pipe_design::benchmark::plane_slice {

/// Axis-aligned slice: fix one coordinate (ROADMAP §1.A.1).
enum class AxisKind : std::uint8_t { XY = 0, XZ = 1, YZ = 2 };

struct Plane {
    AxisKind axis = AxisKind::XY;
    int fixed = 0;  // z for XY, y for XZ, x for YZ (cell units)
};

/// Plane-slice generator input: shared knobs + `plane_gap` slack.
struct GenParams : benchmark::GenParams {
    /// Extra cell slack beyond `diameter_cells()` for parallel planes.
    int plane_gap = 2;

    /// Effective min |fixed_i - fixed_j| (same group, same axis).
    /// Reject when `|fixed_i - fixed_j| <= plane_center_gap()`.
    int plane_center_gap() const {
        return diameter_cells() + plane_gap;
    }
};

}  // namespace pipe_design::benchmark::plane_slice
