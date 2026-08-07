#pragma once

#include "geom/Types.h"
#include "grid/StaticClearanceGrid.h"
#include "topology/Types.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pipe_design::problem {

/// Parameters for the search / clearance grid (matches `StaticClearanceGrid` ctor).
struct GridParams {
    int size_x = 0;
    int size_y = 0;
    int size_z = 0;
    float resolution = 1.f;
    float clearance_cap = 1.f;

    GridParams() = default;
    GridParams(int size_x_, int size_y_, int size_z_,
               float resolution_, float clearance_cap_)
        : size_x(size_x_)
        , size_y(size_y_)
        , size_z(size_z_)
        , resolution(resolution_)
        , clearance_cap(clearance_cap_) {}
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
/// Afterwards only const accessors — no setters. Does **not** hold TopologyPlan,
/// Path, or `ObstacleChain` (solver state).
class Problem {
public:
    Problem(GridParams grid,
            std::vector<geom::SphereObstacle> spheres,
            std::vector<geom::CapsuleObstacle> capsules,
            std::vector<PipeGroup> groups)
        : grid_(std::move(grid))
        , spheres_(std::move(spheres))
        , capsules_(std::move(capsules))
        , groups_(std::move(groups)) {
        static_grid_ = std::make_unique<grid::StaticClearanceGrid>(
            grid_.size_x, grid_.size_y, grid_.size_z,
            grid_.resolution, grid_.clearance_cap);
        static_grid_->bake(spheres_, capsules_);
    }

    Problem(const Problem&) = delete;
    Problem& operator=(const Problem&) = delete;
    Problem(Problem&&) noexcept = default;
    Problem& operator=(Problem&&) noexcept = default;

    const GridParams& grid() const { return grid_; }
    const std::vector<geom::SphereObstacle>& spheres() const { return spheres_; }
    const std::vector<geom::CapsuleObstacle>& capsules() const { return capsules_; }
    const std::vector<PipeGroup>& groups() const { return groups_; }

    const grid::StaticClearanceGrid& static_grid() const { return *static_grid_; }

private:
    GridParams grid_;
    std::vector<geom::SphereObstacle> spheres_;
    std::vector<geom::CapsuleObstacle> capsules_;
    std::vector<PipeGroup> groups_;
    /// Owned by Problem; fixed after construction (CompositeGrid holds a const ref).
    std::unique_ptr<grid::StaticClearanceGrid> static_grid_;
};

}  // namespace pipe_design::problem
