#pragma once

#include "geom/Types.h"
#include "grid/StaticClearanceGrid.h"
#include "topology/Types.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pipe_design::problem {

/// Parameters for the search grid domain.
/// Occupancy truncation (`clearance_cap`) is derived as
/// `max_pipe_radius * clearance_relax` — not set directly.
struct GridParams {
    int size_x = 0;
    int size_y = 0;
    int size_z = 0;
    float resolution = 1.f;
    /// Multiplier on max terminal radius → StaticClearanceGrid clearance_cap.
    /// Must be >= 1 so the field covers every planned pipe_radius.
    /// NOTE: Use > 1 when running AdaptiveStepAStar — step size uses
    /// `clearance - pipe_radius` from the truncated field; cap == max pipe
    /// radius leaves little slack in open space.
    float clearance_relax = 1.f;

    GridParams() = default;
    GridParams(int size_x_, int size_y_, int size_z_, float resolution_,
               float clearance_relax_ = 1.f)
        : size_x(size_x_)
        , size_y(size_y_)
        , size_z(size_z_)
        , resolution(resolution_)
        , clearance_relax(clearance_relax_) {}
};

/// One pipe group: terminals (pipe endpoints / ports) that must form one network.
struct PipeGroup {
    topology::GroupId id;
    std::vector<topology::Terminal> terminals;

    PipeGroup() = default;
    explicit PipeGroup(topology::GroupId id_) : id(std::move(id_)) {}
    PipeGroup(topology::GroupId id_, std::vector<topology::Terminal> terminals_)
        : id(std::move(id_))
        , terminals(std::move(terminals_)) {}
};

/// Frozen user problem definition (ROADMAP §1).
///
/// Construct once with grid / obstacles / groups; `static_grid` is baked in the ctor.
/// `clearance_cap = max_terminal_radius * grid.clearance_relax`.
/// Afterwards only const accessors — no setters. Does **not** hold TopologyPlan,
/// Path, or `ObstacleChain` (solver state).
class Problem {
public:
    Problem(GridParams grid,
            std::vector<geom::SphereObstacle> spheres,
            std::vector<geom::CapsuleObstacle> capsules,
            std::vector<PipeGroup> groups,
            std::vector<geom::AabbObstacle> aabbs = {})
        : grid_(std::move(grid))
        , spheres_(std::move(spheres))
        , capsules_(std::move(capsules))
        , aabbs_(std::move(aabbs))
        , groups_(std::move(groups)) {
        max_pipe_radius_ = compute_max_pipe_radius(groups_);
        if (max_pipe_radius_ <= 0.f) {
            throw std::invalid_argument(
                "Problem: need at least one terminal with positive pipe radius");
        }
        if (grid_.clearance_relax < 1.f) {
            throw std::invalid_argument(
                "Problem: clearance_relax must be >= 1");
        }
        const float clearance_cap = max_pipe_radius_ * grid_.clearance_relax;
        static_grid_ = std::make_unique<grid::StaticClearanceGrid>(
            grid_.size_x, grid_.size_y, grid_.size_z,
            grid_.resolution, clearance_cap);
        static_grid_->bake(spheres_, capsules_, aabbs_);
    }

    Problem(const Problem&) = delete;
    Problem& operator=(const Problem&) = delete;
    Problem(Problem&&) noexcept = default;
    Problem& operator=(Problem&&) noexcept = default;

    const GridParams& grid() const { return grid_; }
    const std::vector<geom::SphereObstacle>& spheres() const { return spheres_; }
    const std::vector<geom::CapsuleObstacle>& capsules() const { return capsules_; }
    const std::vector<geom::AabbObstacle>& aabbs() const { return aabbs_; }
    const std::vector<PipeGroup>& groups() const { return groups_; }

    /// Max terminal radius over all groups (before clearance_relax).
    float max_pipe_radius() const { return max_pipe_radius_; }

    const grid::StaticClearanceGrid& static_grid() const { return *static_grid_; }

private:
    static float compute_max_pipe_radius(const std::vector<PipeGroup>& groups) {
        float best = 0.f;
        for (const PipeGroup& g : groups) {
            for (const topology::Terminal& t : g.terminals) {
                best = std::max(best, t.radius);
            }
        }
        return best;
    }

    GridParams grid_;
    std::vector<geom::SphereObstacle> spheres_;
    std::vector<geom::CapsuleObstacle> capsules_;
    std::vector<geom::AabbObstacle> aabbs_;
    std::vector<PipeGroup> groups_;
    float max_pipe_radius_ = 0.f;
    /// Owned by Problem; fixed after construction (CompositeGrid holds a const ref).
    std::unique_ptr<grid::StaticClearanceGrid> static_grid_;
};

}  // namespace pipe_design::problem
