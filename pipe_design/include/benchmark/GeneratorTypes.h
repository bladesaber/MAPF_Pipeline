#pragma once

#include "planner/low/Path.h"
#include "topology/Types.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace pipe_design::benchmark {

/// Shared generator knobs (all corridor methods). `resolution` is fixed at 1.0.
/// Method-specific fields extend this in each generator's `Types.h`.
struct GenParams {
    int size_x = 0;
    int size_y = 0;
    int size_z = 0;
    /// Fixed grid resolution for all generators (cell size in world units).
    static constexpr float resolution = 1.f;
    int subdomain_n = 3;       // 3, 4, or 5
    float radius = 0.5f;       // uniform terminal / pipe radius
    /// Extra world clearance beyond `radius` for the outer shell (see `shell_cells`).
    /// Planning uses core size `size_* - 2*shell_cells()`; terminals are extended
    /// to the full-domain face afterward.
    float shell_margin = 0.f;
    /// Extra Chebyshev slack (cells) beyond `2*radius/resolution`.
    int terminal_gap = 2;
    int num_groups = 1;
    /// Target face-terminal count per group (each entry >= 2). Size == num_groups.
    std::vector<int> terminals_per_group;
    std::uint32_t seed = 0;
    /// Cap on outer round-robin grow loops (seed / attach rounds).
    int max_iterations = 256;

    /// Shell thickness in cells: ceil((shell_margin + radius) / resolution).
    int shell_cells() const {
        return static_cast<int>(std::ceil((shell_margin + radius) / resolution));
    }

    /// ceil(2 * radius / resolution) — pipe diameter in cells.
    int radius_diameter_cells() const {
        return static_cast<int>(std::ceil(2.f * radius / resolution));
    }

    /// Effective min Chebyshev between terminal centers: diameter + slack.
    /// Call sites reject when `chebyshev <= terminal_center_gap()`.
    int terminal_center_gap() const {
        return radius_diameter_cells() + terminal_gap;
    }
};

/// One group's terminals + witness corridor segments (shared across methods).
struct GroupCorridors {
    topology::GroupId id;
    std::vector<topology::Terminal> terminals;
    std::vector<planner::Path> paths;  // witness segments in grow order
};

/// Constructive corridor result shared by all generators (before obstacle / YAML).
struct Instance {
    GenParams params;
    std::vector<GroupCorridors> groups;
    bool success = false;
    std::string reason;
};

}  // namespace pipe_design::benchmark
