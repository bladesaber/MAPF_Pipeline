#pragma once

#include "benchmark/GeneratorTypes.h"
#include "benchmark/axis_ray/Types.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"
#include "planner/low/Path.h"

#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace pipe_design::benchmark::axis_ray {

/// Axis-aligned ray-tree corridor growth (ROADMAP §1.C).
class AxisRayGenerator {
public:
    explicit AxisRayGenerator(GenParams params);

    bool generate(Instance& out);

private:
    GenParams params_;
    std::mt19937 rng_;
    std::unique_ptr<grid::StaticClearanceGrid> grid_;

    std::vector<GroupCorridors> groups_;
    std::vector<std::vector<grid::CellCoord>> waypoints_;  // per group
    std::vector<int> last_axis_;  // per group: -1 none, 0=x, 1=y, 2=z
    std::unordered_set<grid::CellCoord> occupied_;  // occupied cells (any group)

    bool seed_group(std::size_t g, std::string& reason);
    bool attach_branch(std::size_t g, std::string& reason);

    bool sample_face_terminal(grid::CellCoord& out);
    /// If `reason != nullptr` and growth fails, writes `interior_fail` or `face_seg_fail`.
    bool grow_chain(std::size_t g,
                    grid::CellCoord tip,
                    int forbidden_axis,
                    int interior_segs,
                    bool record_split_waypoint,
                    const grid::CellCoord* split,
                    std::string* reason = nullptr);

    /// Append up to `segs` interior segments; updates `cur` and `axis_forbid`.
    bool try_interior(std::size_t g,
                      int segs,
                      grid::CellCoord& cur,
                      int& axis_forbid);

    /// Last segment from `cur` to a domain face; adds face terminal on success.
    bool try_face_segment(std::size_t g, grid::CellCoord cur, int axis_forbid);

    /// Longest free steps along +dir from `from` (not counting `from`).
    /// `spine_axis`: 0/1/2 = ignore path_center_gap vs occupied on that axis-line
    /// through `from`; -1 = only `from` itself.
    int scan_los(const grid::CellCoord& from,
                  int dx, int dy, int dz,
                  int spine_axis) const;
    void commit_segment(std::size_t g, const planner::Path& path, bool add_end_waypoint);
    bool edge_axis_at(std::size_t g, const grid::CellCoord& p, int& axis) const;
};
}  // namespace pipe_design::benchmark::axis_ray
