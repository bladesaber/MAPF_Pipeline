#pragma once

#include "benchmark/GeneratorTypes.h"
#include "benchmark/Subdomain.h"
#include "benchmark/plane_slice/Types.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"
#include "planner/low/Path.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace pipe_design::benchmark::plane_slice {

/// Plane-sliced multi-source corridor growth (ROADMAP §1.A).
class PlaneSlicedGenerator {
public:
    explicit PlaneSlicedGenerator(GenParams params);

    /// Round-robin: seed empty groups, then attach until targets met (or failure).
    /// Sets `out.success` / `out.reason`.
    bool generate(Instance& out);

private:
    GenParams params_;
    std::mt19937 rng_;
    std::unique_ptr<grid::StaticClearanceGrid> static_grid_;
    std::vector<Subdomain> subdomains_;

    std::vector<GroupCorridors> groups_;
    std::vector<std::vector<Plane>> planes_;

    /// Plane-gap OK and at least one free surface subdomain with face cells on
    /// `plane`. Fills `candidates` (subdomain indices) and `cells_per_sd`.
    bool collect_plane_candidates(
        std::size_t g,
        const Plane& plane,
        std::vector<std::size_t>& candidates,
        std::vector<std::vector<grid::CellCoord>>& cells_per_sd) const;

    /// Two face terminals in different candidate subdomains (seed).
    bool pick_seed_terminals(
        const std::vector<std::size_t>& candidates,
        std::vector<std::vector<grid::CellCoord>>& cells_per_sd,
        grid::CellCoord& s_cell,
        grid::CellCoord& g_cell);

    /// One new face terminal (attach).
    bool pick_attach_terminal(
        const std::vector<std::size_t>& candidates,
        std::vector<std::vector<grid::CellCoord>>& cells_per_sd,
        grid::CellCoord& g_cell);

    bool seed_group(std::size_t g, std::string& reason);
    bool attach_terminal(std::size_t g, std::string& reason);

    /// Planar A* on `plane` only (ROADMAP §1.A). Other groups' corridors = obstacles.
    /// If `reason != nullptr` and search fails, writes a short cause tag.
    bool planar_search(
        const Plane& plane,
        std::size_t self_group,
        const std::vector<grid::CellCoord>& starts,
        const std::vector<grid::CellCoord>& goals,
        planner::Path& out_path,
        std::string* reason = nullptr);
};

}  // namespace pipe_design::benchmark::plane_slice
