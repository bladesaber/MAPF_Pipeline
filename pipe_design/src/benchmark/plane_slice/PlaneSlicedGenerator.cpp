#include "benchmark/plane_slice/PlaneSlicedGenerator.h"
#include "benchmark/GeneratorUtils.h"
#include "benchmark/plane_slice/Utils.h"

#include "geom/Types.h"
#include "grid/CompositeGrid.h"
#include "grid/ObstacleChain.h"
#include "grid/Types.h"
#include "heuristic/Geometric.h"
#include "planner/low/AStar.h"
#include "planner/low/Path.h"
#include "planner/motion/ActionModel.h"
#include "planner/open/PairingHeap.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace pipe_design::benchmark::plane_slice {
namespace {

using grid::CellCoord;
using planner::AStar;
using planner::PairingHeap;
using planner::Path;
using planner::SearchNode;
using planner::SearchStatus;
using planner::motion::ActionModel;
using planner::motion::Successor;
using heuristic::Octile;

/// 8-connected moves that stay on an axis-aligned slice (no out-of-plane steps).
class PlaneEightConnected final : public ActionModel {
public:
    PlaneEightConnected(Plane plane, double step_cost = 1.0)
        : plane_(plane)
        , step_cost_(step_cost) {}

    using ActionModel::successors;

    void successors(
        const CellCoord& from,
        const geom::Heading& in_heading,
        std::vector<Successor>& out) const override {
        if (!on_plane(from, plane_)) {
            return;
        }
        // 4 face + 4 diagonal in the plane's two free axes.
        static constexpr int kUV[8][2] = {
            {1, 0}, {-1, 0}, {0, 1}, {0, -1},
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
        };
        constexpr double kSqrt2 = 1.4142135623730951;
        out.reserve(out.size() + 8);
        for (const auto& uv : kUV) {
            int dx = 0;
            int dy = 0;
            int dz = 0;
            switch (plane_.axis) {
            case AxisKind::XY:
                dx = uv[0];
                dy = uv[1];
                break;
            case AxisKind::XZ:
                dx = uv[0];
                dz = uv[1];
                break;
            case AxisKind::YZ:
                dy = uv[0];
                dz = uv[1];
                break;
            }
            const int axes = (dx != 0) + (dy != 0) + (dz != 0);
            const double length = step_cost_ * (axes == 2 ? kSqrt2 : 1.0);
            const geom::Heading heading =
                geom::Heading::from_grid_step(dx, dy, dz);
            if (!accept_turn(in_heading, heading)) {
                continue;
            }
            out.push_back(Successor{
                CellCoord{from.x + dx, from.y + dy, from.z + dz},
                heading,
                combine_edge_cost(in_heading, heading, length),
            });
        }
    }

private:
    Plane plane_;
    double step_cost_ = 1.0;
};

}  // namespace

PlaneSlicedGenerator::PlaneSlicedGenerator(GenParams params)
    : params_(std::move(params))
    , rng_(params_.seed) {
    if (params_.subdomain_n < 3 || params_.subdomain_n > 5) {
        throw std::invalid_argument(
            "PlaneSlicedGenerator: subdomain_n must be 3, 4, or 5");
    }
    if (params_.num_groups <= 0) {
        throw std::invalid_argument("PlaneSlicedGenerator: num_groups must be > 0");
    }
    if (static_cast<int>(params_.terminals_per_group.size()) != params_.num_groups) {
        throw std::invalid_argument(
            "PlaneSlicedGenerator: terminals_per_group size must match num_groups");
    }
    for (int n : params_.terminals_per_group) {
        if (n < 2) {
            throw std::invalid_argument(
                "PlaneSlicedGenerator: each group needs >= 2 terminals");
        }
    }
    if (params_.size_x < params_.subdomain_n ||
        params_.size_y < params_.subdomain_n ||
        params_.size_z < params_.subdomain_n) {
        throw std::invalid_argument(
            "PlaneSlicedGenerator: domain must be >= subdomain_n per axis");
    }
    if (params_.max_iterations <= 0) {
        throw std::invalid_argument(
            "PlaneSlicedGenerator: max_iterations must be > 0");
    }
}

bool PlaneSlicedGenerator::generate(Instance& out) {
    out = Instance{};

    const int full_x = params_.size_x;
    const int full_y = params_.size_y;
    const int full_z = params_.size_z;
    const int m = params_.shell_cells();
    const int core_x = full_x - 2 * m;
    const int core_y = full_y - 2 * m;
    const int core_z = full_z - 2 * m;
    if (core_x < params_.subdomain_n || core_y < params_.subdomain_n ||
        core_z < params_.subdomain_n) {
        out.params = params_;
        out.success = false;
        out.reason = "core domain too small for shell (need size-2*shell_cells >= subdomain_n)";
        return false;
    }
    params_.size_x = core_x;
    params_.size_y = core_y;
    params_.size_z = core_z;

    const std::size_t n = static_cast<std::size_t>(params_.num_groups);
    groups_.assign(n, GroupCorridors{});
    planes_.assign(n, {});
    for (std::size_t g = 0; g < n; ++g) {
        groups_[g].id = "g" + std::to_string(g);
    }

    static_grid_ = std::make_unique<grid::StaticClearanceGrid>(
        params_.size_x, params_.size_y, params_.size_z,
        GenParams::resolution, /*clearance_cap=*/params_.radius);
    static_grid_->clear();
    subdomains_ = make_subdomains(params_.subdomain_n, params_.size_x,
                                  params_.size_y, params_.size_z);

    bool ok = false;
    std::string reason =
        "max_iterations exceeded (" + std::to_string(params_.max_iterations) + ")";
    for (int iter = 0; iter < params_.max_iterations; ++iter) {
        bool any_needed = false;
        bool step_ok = true;
        for (std::size_t g = 0; g < n; ++g) {
            const int target = params_.terminals_per_group[g];
            if (static_cast<int>(groups_[g].terminals.size()) >= target) {
                continue;
            }
            any_needed = true;
            if (groups_[g].terminals.empty() ? !seed_group(g) : !attach_terminal(g)) {
                reason = groups_[g].terminals.empty()
                             ? ("seed_group failed for " + groups_[g].id)
                             : ("attach_terminal failed for " + groups_[g].id);
                step_ok = false;
                break;
            }
        }
        if (!step_ok) {
            break;
        }
        if (!any_needed) {
            ok = true;
            reason.clear();
            break;
        }
    }

    params_.size_x = full_x;
    params_.size_y = full_y;
    params_.size_z = full_z;
    out.params = params_;
    out.groups = groups_;
    out.success = ok;
    out.reason = reason;
    expand_shell(out, m);
    append_shell_aabbs(out.groups, out.params, out.aabbs);
    return ok;
}

bool PlaneSlicedGenerator::collect_plane_candidates(
    std::size_t g,
    const Plane& plane,
    std::vector<std::size_t>& candidates,
    std::vector<std::vector<CellCoord>>& cells_per_sd) const {
    candidates.clear();
    cells_per_sd.assign(subdomains_.size(), {});

    /// distance between planes must be greater than plane_gap
    for (const Plane& used : planes_[g]) {
        if (used.axis == plane.axis &&
            std::abs(used.fixed - plane.fixed) <= params_.plane_center_gap()) {
            return false;
        }
    }

    const int sx = static_grid_->size_x();
    const int sy = static_grid_->size_y();
    const int sz = static_grid_->size_z();
    for (std::size_t i = 0; i < subdomains_.size(); ++i) {
        const Subdomain& sd = subdomains_[i];
        if (!sd.is_surface(params_.subdomain_n) || !intersects_plane(sd, plane)) {
            continue;
        }
        bool occupied_same = false;
        /// no terminal in the same subdomain
        for (const auto& t : groups_[g].terminals) {
            if (sd.contains_cell(static_grid_->world_to_cell(t.pos))) {
                occupied_same = true;
                break;
            }
        }
        if (occupied_same) {
            continue;
        }
        append_face_cells_on_plane(sd, plane, sx, sy, sz, cells_per_sd[i]);
        if (!cells_per_sd[i].empty()) {
            candidates.push_back(i);
        }
    }
    return !candidates.empty();
}

bool PlaneSlicedGenerator::planar_search(
    const Plane& plane,
    std::size_t self_group,
    const std::vector<CellCoord>& starts,
    const std::vector<CellCoord>& goals,
    Path& out_path) {
    out_path = Path{};
    if (starts.empty() || goals.empty()) {
        return false;
    }
    for (const CellCoord& c : starts) {
        if (!on_plane(c, plane)) {
            return false;
        }
    }
    for (const CellCoord& c : goals) {
        if (!on_plane(c, plane)) {
            return false;
        }
    }

    const float r = params_.radius;
    std::shared_ptr<const grid::ObstacleChain> chain = grid::ObstacleChain::make_root();
    for (std::size_t og = 0; og < groups_.size(); ++og) {
        if (og == self_group) {
            continue;
        }
        for (const Path& path : groups_[og].paths) {
            chain = bake_path(std::move(chain), *static_grid_, path, r);
        }
    }
    grid::CompositeGrid cgrid(*static_grid_, chain);
    for (const CellCoord& c : starts) {
        if (!cgrid.is_free(c, r)) {
            return false;
        }
    }
    for (const CellCoord& c : goals) {
        if (!cgrid.is_free(c, r)) {
            return false;
        }
    }

    Octile h(static_grid_->resolution());
    PlaneEightConnected actions(plane, static_grid_->resolution());
    PairingHeap open;
    AStar<SearchNode> astar;
    astar.reset(cgrid, starts, goals, r, h, open, actions);
    if (astar.search() != SearchStatus::Success) {
        return false;
    }
    out_path = astar.path();
    return !out_path.empty();
}

bool PlaneSlicedGenerator::pick_seed_terminals(
    const std::vector<std::size_t>& candidates,  // <subdomain index>
    std::vector<std::vector<CellCoord>>& cells_per_sd,  // <subdomain, cells>
    CellCoord& s_cell,
    CellCoord& g_cell) {
    s_cell = {-1, -1, -1};
    g_cell = {-1, -1, -1};
    if (candidates.size() < 2) {
        return false;
    }
    std::vector<std::size_t> order = candidates;
    std::shuffle(order.begin(), order.end(), rng_);

    for (std::size_t a = 0; a + 1 < order.size(); ++a) {
        auto& ba = cells_per_sd[order[a]];
        std::shuffle(ba.begin(), ba.end(), rng_);
        for (const CellCoord& sc : ba) {
            if (!terminal_clearance_ok(sc, groups_, *static_grid_,
                                       params_.terminal_center_gap())) {
                continue;
            }
            for (std::size_t b = a + 1; b < order.size(); ++b) {
                auto& bb = cells_per_sd[order[b]];
                std::shuffle(bb.begin(), bb.end(), rng_);
                for (const CellCoord& gc : bb) {
                    if (gc == sc ||
                        !terminal_clearance_ok(gc, groups_, *static_grid_, params_.terminal_center_gap()) ||
                        chebyshev(sc, gc) < params_.terminal_center_gap()) {
                        continue;
                    }
                    s_cell = sc;
                    g_cell = gc;
                    return true;
                }
            }
        }
    }
    return false;
}

bool PlaneSlicedGenerator::seed_group(std::size_t g) {
    const float r = params_.radius;
    const int sx = static_grid_->size_x();
    const int sy = static_grid_->size_y();
    const int sz = static_grid_->size_z();
    constexpr int kAttempts = 64;

    for (int attempt = 0; attempt < kAttempts; ++attempt) {
        Plane plane;
        plane.axis = static_cast<AxisKind>(std::uniform_int_distribution<int>(0, 2)(rng_));
        int fixed_max = sz;
        if (plane.axis == AxisKind::XZ) {
            fixed_max = sy;
        } else if (plane.axis == AxisKind::YZ) {
            fixed_max = sx;
        }
        plane.fixed = std::uniform_int_distribution<int>(0, fixed_max - 1)(rng_);

        std::vector<std::size_t> candidates;
        std::vector<std::vector<CellCoord>> cells_per_sd;
        if (!collect_plane_candidates(g, plane, candidates, cells_per_sd)) {
            continue;
        }

        CellCoord s_cell;
        CellCoord g_cell;
        if (!pick_seed_terminals(candidates, cells_per_sd, s_cell, g_cell)) {
            continue;
        }

        Path path;
        if (!planar_search(plane, g, {s_cell}, {g_cell}, path)) {
            continue;
        }

        const std::string gid = groups_[g].id;
        groups_[g].terminals.push_back({gid + "_t0", static_grid_->cell_center(s_cell), r});
        groups_[g].terminals.push_back({gid + "_t1", static_grid_->cell_center(g_cell), r});
        groups_[g].paths.push_back(std::move(path));
        planes_[g].push_back(plane);
        return true;
    }
    return false;
}

bool PlaneSlicedGenerator::pick_attach_terminal(
    const std::vector<std::size_t>& candidates,
    std::vector<std::vector<CellCoord>>& cells_per_sd,
    CellCoord& g_cell) {
    g_cell = {-1, -1, -1};
    if (candidates.empty()) {
        return false;
    }
    std::vector<std::size_t> order = candidates;
    std::shuffle(order.begin(), order.end(), rng_);
    for (std::size_t idx : order) {
        auto& cells = cells_per_sd[idx];
        std::shuffle(cells.begin(), cells.end(), rng_);
        for (const CellCoord& c : cells) {
            if (!terminal_clearance_ok(c, groups_, *static_grid_,
                                       params_.terminal_center_gap())) {
                continue;
            }
            g_cell = c;
            return true;
        }
    }
    return false;
}

bool PlaneSlicedGenerator::attach_terminal(std::size_t g) {
    const float r = params_.radius;
    constexpr int kAttempts = 64;

    if (planes_[g].empty() || groups_[g].paths.empty()) {
        return false;
    }
    const AxisKind last_axis = planes_[g].back().axis;

    // Path cells as plane-definition candidates.
    std::vector<CellCoord> path_cells;
    for (const Path& path : groups_[g].paths) {
        path_cells.insert(path_cells.end(), path.cells.begin(), path.cells.end());
    }
    if (path_cells.empty()) {
        return false;
    }

    for (int attempt = 0; attempt < kAttempts; ++attempt) {
        // Axis ≠ last; sample a path cell → define plane.
        AxisKind axis = last_axis;
        while (axis == last_axis) {
            axis = static_cast<AxisKind>(std::uniform_int_distribution<int>(0, 2)(rng_));
        }
        const CellCoord& pivot =
            path_cells[std::uniform_int_distribution<std::size_t>(
                0, path_cells.size() - 1)(rng_)];
        Plane plane{axis, fixed_of(pivot, axis)};
        std::vector<std::size_t> candidates;
        std::vector<std::vector<CellCoord>> cells_per_sd;
        if (!collect_plane_candidates(g, plane, candidates, cells_per_sd)) {
            continue;
        }
        CellCoord g_cell;
        if (!pick_attach_terminal(candidates, cells_per_sd, g_cell)) {
            continue;
        }

        // Multi-source starts: unique path cells on the plane (terminals are
        // path endpoints already).
        std::unordered_set<CellCoord> start_set;
        for (const Path& path : groups_[g].paths) {
            for (const CellCoord& c : path.cells) {
                if (on_plane(c, plane)) {
                    start_set.insert(c);
                }
            }
        }
        if (start_set.empty()) {
            continue;
        }
        std::vector<CellCoord> starts(start_set.begin(), start_set.end());

        Path path;
        if (!planar_search(plane, g, starts, {g_cell}, path)) {
            continue;
        }

        const std::string tid = groups_[g].id + "_t" + std::to_string(groups_[g].terminals.size());
        groups_[g].terminals.push_back({tid, static_grid_->cell_center(g_cell), r});
        groups_[g].paths.push_back(std::move(path));
        planes_[g].push_back(plane);
        return true;
    }
    return false;
}

}  // namespace pipe_design::benchmark::plane_slice
