#pragma once

#include "geom/Types.h"
#include "planner/low/Path.h"
#include "topology/Types.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
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
    /// Chebyshev carve of witness tunnels through shell AABBs (cells).
    /// Must be `>= 0` (`append_shell_aabbs` throws otherwise). Typical: `radius_cells()`.
    int carve_margin = 0;
    /// Extra Chebyshev slack (cells) beyond `2*radius/resolution`.
    int terminal_gap = 2;
    int num_groups = 1;
    /// Target face-terminal count per group (each entry >= 2). Size == num_groups.
    std::vector<int> terminals_per_group;
    std::uint32_t seed = 0;
    /// Cap on outer round-robin grow loops (seed / attach rounds).
    int max_iterations = 256;

    /// Full shell band for planning: ceil((shell_margin + radius) / resolution).
    int shell_cells() const {
        return static_cast<int>(std::ceil((shell_margin + radius) / resolution));
    }

    /// Solid shell obstacles export thickness (cells): `floor(shell_margin / resolution)`.
    /// Planning shell still uses `shell_cells()` (= margin + radius). When this is 0,
    /// `append_shell_aabbs` still emits a 1-cell face layer (planar domain wall).
    int shell_obstacle_cells() const {
        return static_cast<int>(std::floor(shell_margin / resolution));
    }

    /// ceil(radius / resolution) — pipe radius in cells (obstacle keep-out scale).
    int radius_cells() const {
        return static_cast<int>(std::ceil(radius / resolution));
    }

    /// ceil(2 * radius / resolution) — pipe diameter in cells (pipe–pipe gaps).
    int diameter_cells() const {
        return static_cast<int>(std::ceil(2.f * radius / resolution));
    }

    /// Effective min Chebyshev between terminal centers: diameter + slack.
    /// Call sites reject when `chebyshev <= terminal_center_gap()`.
    int terminal_center_gap() const {
        return diameter_cells() + terminal_gap;
    }
};

/// Shared knobs for all obstacle layout generators (PillarField, CorridorRack, …).
/// Family-specific fields extend this in each obstacle folder's `Types.h`.
struct ObstacleGenParams {
    float density_target = 0.4f;
    /// Chebyshev thicken of witness paths → ineligible for obstacles.
    /// Pipe center must keep clearance ≥ r to obstacle surfaces, so the floor is
    /// ~radius (not diameter). If < 0, use `GenParams::radius_cells()`.
    int corridor_margin = -1;
    std::uint32_t seed = 0;
    int max_rounds = 256;
};

/// One group's terminals + witness corridor segments (shared across methods).
struct GroupCorridors {
    topology::GroupId id;
    std::vector<topology::Terminal> terminals;
    std::vector<planner::Path> paths;  // witness segments in grow order
};

/// Constructive corridor result shared by all generators (before / after obstacle paint).
/// Obstacle paint may fill any of sphere / capsule / AABB (same set as `problem::Problem`).
struct Instance {
    GenParams params;
    ObstacleGenParams obstacle_params;
    std::vector<GroupCorridors> groups;
    std::vector<geom::SphereObstacle> spheres;
    std::vector<geom::CapsuleObstacle> capsules;
    std::vector<geom::AabbObstacle> aabbs;
    bool success = false;
    std::string reason;

    /// Write YAML instance (ROADMAP §3). `generator` e.g. "axis_ray";
    /// `generator_params` merged into `meta`.
    bool to_file(const std::string& path,
                 const std::string& generator,
                 const YAML::Node& generator_params = {}) const;

    /// Load YAML including witness `solution` paths (for vs-witness compare).
    static Instance from_file(const std::string& path);
};

}  // namespace pipe_design::benchmark
