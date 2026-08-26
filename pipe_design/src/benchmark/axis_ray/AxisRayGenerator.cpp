#include "benchmark/axis_ray/AxisRayGenerator.h"
#include "benchmark/GeneratorUtils.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace pipe_design::benchmark::axis_ray {
namespace {

using grid::CellCoord;
using planner::Path;

// dx,dy,dz and axis 0/1/2 for ±x,±y,±z
constexpr int kDir[6][4] = {
    {1, 0, 0, 0}, {-1, 0, 0, 0},
    {0, 1, 0, 1}, {0, -1, 0, 1},
    {0, 0, 1, 2}, {0, 0, -1, 2},
};

Path make_segment(CellCoord from, int dx, int dy, int dz, int steps) {
    Path p;
    p.cells.push_back(from);
    CellCoord c = from;
    for (int i = 0; i < steps; ++i) {
        c = CellCoord{c.x + dx, c.y + dy, c.z + dz};
        p.cells.push_back(c);
    }
    p.cost = static_cast<double>(steps);
    return p;
}

}  // namespace

AxisRayGenerator::AxisRayGenerator(GenParams params)
    : params_(std::move(params))
    , rng_(params_.seed) {
    if (params_.num_groups <= 0) {
        throw std::invalid_argument("AxisRayGenerator: num_groups must be > 0");
    }
    if (static_cast<int>(params_.terminals_per_group.size()) != params_.num_groups) {
        throw std::invalid_argument(
            "AxisRayGenerator: terminals_per_group size must match num_groups");
    }
    for (int n : params_.terminals_per_group) {
        if (n < 2) {
            throw std::invalid_argument("AxisRayGenerator: each group needs >= 2 terminals");
        }
    }
    if (params_.init_waypoints_num < 2 || params_.attach_waypoints_num < 2) {
        throw std::invalid_argument("AxisRayGenerator: *_waypoints_num must be >= 2");
    }
    if (params_.min_path_size < 1 || params_.max_path_size < params_.min_path_size) {
        throw std::invalid_argument("AxisRayGenerator: invalid min/max_path_size");
    }
    if (params_.path_center_gap() < 1 || params_.branch_center_gap() < 1) {
        throw std::invalid_argument(
            "AxisRayGenerator: path_center_gap and branch_center_gap must be >= 1");
    }
    // Equal is OK. Strict branch > path was only a style preference for
    // "clearly interior" splits; it rejected many valid attach points on dense trees.
    if (params_.branch_center_gap() < params_.path_center_gap()) {
        throw std::invalid_argument(
            "AxisRayGenerator: branch_center_gap must be >= path_center_gap "
            "(increase branch_gap or decrease path_gap)");
    }
    if (params_.size_x < 3 || params_.size_y < 3 || params_.size_z < 3) {
        throw std::invalid_argument("AxisRayGenerator: domain too small");
    }
}

bool AxisRayGenerator::sample_face_terminal(CellCoord& out) {
    std::vector<CellCoord> faces;
    faces.reserve(static_cast<std::size_t>(
        2 * (params_.size_x * params_.size_y + params_.size_x * params_.size_z +
             params_.size_y * params_.size_z)));
    // z-faces (full); later faces shrink ranges so edges/corners appear once.
    for (int x = 0; x < params_.size_x; ++x) {
        for (int y = 0; y < params_.size_y; ++y) {
            faces.push_back({x, y, 0});
            faces.push_back({x, y, params_.size_z - 1});
        }
    }
    for (int x = 0; x < params_.size_x; ++x) {
        for (int z = 1; z < params_.size_z - 1; ++z) {
            faces.push_back({x, 0, z});
            faces.push_back({x, params_.size_y - 1, z});
        }
    }
    for (int y = 1; y < params_.size_y - 1; ++y) {
        for (int z = 1; z < params_.size_z - 1; ++z) {
            faces.push_back({0, y, z});
            faces.push_back({params_.size_x - 1, y, z});
        }
    }
    std::shuffle(faces.begin(), faces.end(), rng_);
    for (const CellCoord& c : faces) {
        if (occupied_.count(c)) {
            continue;
        }
        if (!terminal_clearance_ok(c, groups_, *grid_,
                                   params_.terminal_center_gap())) {
            continue;
        }
        out = c;
        return true;
    }
    return false;
}

int AxisRayGenerator::scan_los(const CellCoord& from,
                                int dx, int dy, int dz,
                                int spine_axis) const {
    int steps = 0;
    CellCoord c = from;
    while (true) {
        const CellCoord n{c.x + dx, c.y + dy, c.z + dz};
        if (!grid_->in_bounds(n)) {
            break;
        }
        bool clear = true;
        for (const CellCoord& o : occupied_) {
            // Ignore path_center_gap vs occupied on the spine through `from`
            // (or just `from` when spine_axis < 0).
            if (spine_axis == 0 && o.y == from.y && o.z == from.z) {
                continue;
            }
            if (spine_axis == 1 && o.x == from.x && o.z == from.z) {
                continue;
            }
            if (spine_axis == 2 && o.x == from.x && o.y == from.y) {
                continue;
            }
            if (spine_axis < 0 && o == from) {
                continue;
            }
            if (chebyshev(n, o) <= params_.path_center_gap()) {
                clear = false;
                break;
            }
        }
        if (!clear) {
            break;
        }
        ++steps;
        c = n;
    }
    return steps;
}

void AxisRayGenerator::commit_segment(std::size_t g,
                                      const Path& path,
                                      bool add_end_waypoint) {
    groups_[g].paths.push_back(path);
    for (std::size_t i = 1; i < path.cells.size(); ++i) {
        occupied_.insert(path.cells[i]);
    }

    if (add_end_waypoint && !path.cells.empty()) {
        waypoints_[g].push_back(path.cells.back());
    }
}

bool AxisRayGenerator::try_interior(std::size_t g,
                                      int segs,
                                      CellCoord& cur,
                                      int& axis_forbid) {
    for (int s = 0; s < segs; ++s) {
        const int spine = axis_forbid;

        std::vector<int> dirs;
        for (int d = 0; d < 6; ++d) {
            if (kDir[d][3] != axis_forbid) {
                dirs.push_back(d);
            }
        }
        std::shuffle(dirs.begin(), dirs.end(), rng_);

        bool placed = false;
        for (int d : dirs) {
            const int dx = kDir[d][0], dy = kDir[d][1], dz = kDir[d][2];
            const int L = scan_los(cur, dx, dy, dz, spine);
            if (L < params_.min_path_length()) {
                continue;
            }
            const int hi = std::min(L, params_.max_path_length());
            std::uniform_int_distribution<int> dist(params_.min_path_length(), hi);
            const int steps = dist(rng_);
            Path seg = make_segment(cur, dx, dy, dz, steps);
            commit_segment(g, seg, /*add_end_waypoint=*/true);
            cur = seg.cells.back();
            axis_forbid = kDir[d][3];
            last_axis_[g] = axis_forbid;
            placed = true;
            break;
        }
        if (!placed) {
            return false;
        }
    }
    return true;
}

bool AxisRayGenerator::try_face_segment(std::size_t g,
                                          CellCoord cur,
                                          int axis_forbid) {
    const int spine = axis_forbid;

    std::vector<int> dirs;
    for (int d = 0; d < 6; ++d) {
        if (kDir[d][3] != axis_forbid) {
            dirs.push_back(d);
        }
    }
    std::shuffle(dirs.begin(), dirs.end(), rng_);

    for (int d : dirs) {
        const int dx = kDir[d][0], dy = kDir[d][1], dz = kDir[d][2];
        const int L = scan_los(cur, dx, dy, dz, spine);
        if (L < 1) {
            continue;
        }
        // Last free LOS cell must lie on the domain face (no max/min length).
        const CellCoord end{cur.x + dx * L, cur.y + dy * L, cur.z + dz * L};
        if (!on_face(end, params_.size_x, params_.size_y, params_.size_z)) {
            continue;
        }
        // scan_los ignores spine occupied / path_gap only; terminal_center_gap
        // may be stricter. Chebyshev clearance implies no equal-r sphere overlap.
        if (!terminal_clearance_ok(end, groups_, *grid_,
                                   params_.terminal_center_gap())) {
            continue;
        }
        Path seg = make_segment(cur, dx, dy, dz, L);
        commit_segment(g, seg, /*add_end_waypoint=*/true);
        last_axis_[g] = kDir[d][3];

        topology::Terminal t;
        t.id = groups_[g].id + "_t" + std::to_string(groups_[g].terminals.size());
        t.pos = grid_->cell_center(end);
        t.radius = params_.radius;
        groups_[g].terminals.push_back(std::move(t));
        return true;
    }
    return false;
}

bool AxisRayGenerator::grow_chain(std::size_t g,
                                    CellCoord tip,
                                    int forbidden_axis,
                                    int interior_segs,
                                    bool record_split_waypoint,
                                    const CellCoord* split,
                                    std::string* reason) {
    if (record_split_waypoint && split != nullptr) {
        const bool already =
            std::find(waypoints_[g].begin(), waypoints_[g].end(), *split) !=
            waypoints_[g].end();
        if (!already) {
            waypoints_[g].push_back(*split);
        }
    }

    int axis_forbid = forbidden_axis;
    CellCoord cur = tip;
    if (!try_interior(g, interior_segs, cur, axis_forbid)) {
        if (reason != nullptr) {
            *reason = "interior_fail";
        }
        return false;
    }
    if (!try_face_segment(g, cur, axis_forbid)) {
        if (reason != nullptr) {
            *reason = "face_seg_fail";
        }
        return false;
    }
    if (reason != nullptr) {
        reason->clear();
    }
    return true;
}

bool AxisRayGenerator::seed_group(std::size_t g, std::string& reason) {
    int n_grow = 0;
    std::string last_grow;

    for (int attempt = 0; attempt < params_.max_attempts; ++attempt) {
        const auto occ_snap = occupied_;
        const auto wp_snap = waypoints_[g];
        const auto paths_snap = groups_[g].paths;
        const auto term_snap = groups_[g].terminals;
        const int axis_snap = last_axis_[g];

        CellCoord start;
        if (!sample_face_terminal(start)) {
            occupied_ = occ_snap;
            waypoints_[g] = wp_snap;
            groups_[g].paths = paths_snap;
            groups_[g].terminals = term_snap;
            last_axis_[g] = axis_snap;
            reason = "no_face_terminal attempt=" + std::to_string(attempt);
            return false;
        }
        occupied_.insert(start);
        waypoints_[g].push_back(start);
        topology::Terminal t0;
        t0.id = groups_[g].id + "_t0";
        t0.pos = grid_->cell_center(start);
        t0.radius = params_.radius;
        groups_[g].terminals.push_back(std::move(t0));
        last_axis_[g] = -1;

        // Waypoints on seed ≈ segments+1 → interior segs = init-2, then one face seg.
        const int interior = params_.init_waypoints_num - 2;
        std::string grow_reason;
        if (grow_chain(g, start, /*forbidden_axis=*/-1, interior,
                       /*record_split=*/false, nullptr, &grow_reason)) {
            reason.clear();
            return true;
        }
        ++n_grow;
        last_grow = grow_reason;

        /// roll back
        occupied_ = occ_snap;
        waypoints_[g] = wp_snap;
        groups_[g].paths = paths_snap;
        groups_[g].terminals = term_snap;
        last_axis_[g] = axis_snap;
    }
    reason = "attempts=" + std::to_string(params_.max_attempts) +
             " grow_fail=" + std::to_string(n_grow);
    if (!last_grow.empty()) {
        reason += " last_grow=" + last_grow;
    }
    return false;
}

bool AxisRayGenerator::edge_axis_at(std::size_t g, const CellCoord& p, int& axis) const {
    // Each Path is one axis-aligned straight segment (make_segment).
    for (const Path& path : groups_[g].paths) {
        if (path.cells.size() < 2) {
            throw std::logic_error("AxisRayGenerator: path with <2 cells");
        }
        const CellCoord& a = path.cells.front();
        const CellCoord& b = path.cells.back();
        // O(1): p on the axis-aligned segment a—b?
        if (a.y == b.y && a.z == b.z && p.y == a.y && p.z == a.z &&
            p.x >= std::min(a.x, b.x) && p.x <= std::max(a.x, b.x)) {
            axis = 0;
            return true;
        }
        if (a.x == b.x && a.z == b.z && p.x == a.x && p.z == a.z &&
            p.y >= std::min(a.y, b.y) && p.y <= std::max(a.y, b.y)) {
            axis = 1;
            return true;
        }
        if (a.x == b.x && a.y == b.y && p.x == a.x && p.y == a.y &&
            p.z >= std::min(a.z, b.z) && p.z <= std::max(a.z, b.z)) {
            axis = 2;
            return true;
        }
    }
    throw std::logic_error("AxisRayGenerator: p not on any axis-aligned segment");
}

bool AxisRayGenerator::attach_branch(std::size_t g, std::string& reason) {
    const int interior = params_.attach_waypoints_num - 2;

    int n_empty_cand = 0;
    int n_grow = 0;
    int n_split_tried = 0;
    std::string last_grow;

    for (int attempt = 0; attempt < params_.max_attempts; ++attempt) {
        std::vector<CellCoord> candidates;
        for (const Path& path : groups_[g].paths) {
            for (const CellCoord& c : path.cells) {
                bool far = true;
                for (const CellCoord& w : waypoints_[g]) {
                    if (chebyshev(c, w) <= params_.branch_center_gap()) {
                        far = false;
                        break;
                    }
                }
                if (far) {
                    candidates.push_back(c);
                }
            }
        }
        // No split far enough from waypoints — further attempts cannot help.
        if (candidates.empty()) {
            ++n_empty_cand;
            reason = "no_branch_candidates attempt=" + std::to_string(attempt) +
                     " empty_cand_rounds=" + std::to_string(n_empty_cand);
            return false;
        }
        std::shuffle(candidates.begin(), candidates.end(), rng_);

        const auto occ_snap = occupied_;
        const auto wp_snap = waypoints_[g];
        const auto paths_snap = groups_[g].paths;
        const auto term_snap = groups_[g].terminals;
        const int axis_snap = last_axis_[g];

        // Try every candidate this attempt (not only candidates.front()).
        // Candidates are already > branch_center_gap from every waypoint, so each
        // split lies on an edge interior — forbid that edge's axis (side exit).
        for (const CellCoord& split : candidates) {
            int forbid = -1;
            edge_axis_at(g, split, forbid);
            ++n_split_tried;

            std::string grow_reason;
            if (grow_chain(g, split, forbid, interior, /*record_split=*/true,
                           &split, &grow_reason)) {
                reason.clear();
                return true;
            }
            ++n_grow;
            last_grow = grow_reason;

            occupied_ = occ_snap;
            waypoints_[g] = wp_snap;
            groups_[g].paths = paths_snap;
            groups_[g].terminals = term_snap;
            last_axis_[g] = axis_snap;
        }
    }
    reason = "attempts=" + std::to_string(params_.max_attempts) +
             " split_tried=" + std::to_string(n_split_tried) +
             " grow_fail=" + std::to_string(n_grow);
    if (!last_grow.empty()) {
        reason += " last_grow=" + last_grow;
    }
    return false;
}

bool AxisRayGenerator::generate(Instance& out) {
    out = Instance{};

    const int full_x = params_.size_x;
    const int full_y = params_.size_y;
    const int full_z = params_.size_z;
    const int m = params_.shell_cells();
    if (full_x < 2 * m + 3 || full_y < 2 * m + 3 || full_z < 2 * m + 3) {
        out.params = params_;
        out.success = false;
        out.reason = "domain too small for shell (need size >= 2*shell_cells+3)";
        return false;
    }
    params_.size_x = full_x - 2 * m;
    params_.size_y = full_y - 2 * m;
    params_.size_z = full_z - 2 * m;

    const std::size_t n = static_cast<std::size_t>(params_.num_groups);
    groups_.assign(n, GroupCorridors{});
    waypoints_.assign(n, {});
    last_axis_.assign(n, -1);
    occupied_.clear();
    for (std::size_t g = 0; g < n; ++g) {
        groups_[g].id = "g" + std::to_string(g);
    }

    grid_ = std::make_unique<grid::StaticClearanceGrid>(
        params_.size_x, params_.size_y, params_.size_z,
        GenParams::resolution, /*clearance_cap=*/params_.radius);
    grid_->clear();

    bool ok = false;
    std::string reason = "max_iterations exceeded before all groups reached terminal targets";
    for (int iter = 0; iter < params_.max_iterations; ++iter) {
        bool any_needed = false;
        bool step_ok = true;
        for (std::size_t g = 0; g < n; ++g) {
            const int target = params_.terminals_per_group[g];
            if (static_cast<int>(groups_[g].terminals.size()) >= target) {
                continue;
            }
            any_needed = true;
            std::string step_reason;
            const bool need_seed = groups_[g].terminals.empty();
            const bool step_ok_g =
                need_seed ? seed_group(g, step_reason) : attach_branch(g, step_reason);
            if (!step_ok_g) {
                reason = (need_seed ? "seed_group" : "attach_branch") +
                         std::string(" failed for ") + groups_[g].id +
                         " (have_terms=" +
                         std::to_string(groups_[g].terminals.size()) + "/" +
                         std::to_string(target) + "): " + step_reason;
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

}  // namespace pipe_design::benchmark::axis_ray
