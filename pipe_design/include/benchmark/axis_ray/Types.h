#pragma once

#include "benchmark/GeneratorTypes.h"

namespace pipe_design::benchmark::axis_ray {

/// Axis-aligned LOS ray tree (ROADMAP §1.C). No subdomains.
struct GenParams : benchmark::GenParams {
    int init_waypoints_num = 3;    // seed: interior segs = this-1, then face seg
    int attach_waypoints_num = 2;  // attach branch: same pattern
    /// Interior segment length as multiples of pipe diameter (cells via accessors).
    int min_path_size = 4;
    int max_path_size = 16;        // interior only; last face seg ignores max
    /// Extra Chebyshev slack beyond `diameter_cells()` vs occupied paths.
    int path_gap = 2;
    /// Extra Chebyshev slack beyond diameter vs waypoints when picking a split.
    /// Effective `branch_center_gap()` should be >= `path_center_gap()` (equal OK).
    int branch_gap = 3;
    int max_attempts = 64;         // retries per seed / attach

    /// Effective min steps (excluding start): diameter * min_path_size.
    int min_path_length() const {
        return diameter_cells() * min_path_size;
    }

    /// Effective max interior steps: diameter * max_path_size.
    int max_path_length() const {
        return diameter_cells() * max_path_size;
    }

    /// Effective min Chebyshev to occupied path cells.
    /// Reject when `chebyshev <= path_center_gap()`.
    int path_center_gap() const {
        return diameter_cells() + path_gap;
    }

    /// Effective min Chebyshev to existing waypoints when picking a split.
    /// Reject when `chebyshev <= branch_center_gap()`.
    int branch_center_gap() const {
        return diameter_cells() + branch_gap;
    }
};

}  // namespace pipe_design::benchmark::axis_ray
