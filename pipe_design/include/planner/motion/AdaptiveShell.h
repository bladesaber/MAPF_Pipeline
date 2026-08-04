#pragma once

#include "planner/motion/ActionModel.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace pipe_design::planner::motion {

/// Chebyshev shell motion (caller / planner chooses step size).
///
/// For step size `s`, successors are all cells with
/// `max(|dx|,|dy|,|dz|) == s` (outer shell), including e.g. `{2,1,0}` when `s=2`
/// — not only axis-scaled 6/26 rays. Edge length is Euclidean · `step_cost`.
///
/// `offsets_for(s)` / `interior_offsets_for(s)` cache shell and ball interiors.
/// Clearance → `s` mapping belongs in the planner, not here.
class AdaptiveShell final : public ActionModel {
public:
    struct Offset {
        int dx = 0;
        int dy = 0;
        int dz = 0;
    };

    explicit AdaptiveShell(double step_cost = 1.0, int max_step = 4)
        : step_cost_(step_cost), max_step_(std::max(1, max_step)) {}

    using ActionModel::successors;

    double step_cost() const { return step_cost_; }
    int max_step() const { return max_step_; }

    void set_max_step(int max_step) {
        max_step_ = std::max(1, max_step);
        step_size_ = std::clamp(step_size_, 1, max_step_);
    }

    /// Shell radius in cells, clamped to `[1, max_step]`.
    void set_step_size(int step_size) { step_size_ = std::clamp(step_size, 1, max_step_); }
    int step_size() const { return step_size_; }

    /// Chebyshev outer shell for `step_size` (cached).
    const std::vector<Offset>& offsets_for(int step_size) const {
        const int s = std::max(1, step_size);
        const auto it = shell_cache_.find(s);
        if (it != shell_cache_.end()) {
            return it->second;
        }

        std::vector<Offset> offs;
        // Surface only: |d|∞ == s → 24 s² + 2 cells (no interior scan).
        offs.reserve(static_cast<std::size_t>(24 * s * s + 2));
        for (const int sx : {-s, s}) {
            for (int dy = -s; dy <= s; ++dy) {
                for (int dz = -s; dz <= s; ++dz) {
                    offs.push_back(Offset{sx, dy, dz});
                }
            }
        }
        for (const int sy : {-s, s}) {
            for (int dx = -s + 1; dx <= s - 1; ++dx) {
                for (int dz = -s; dz <= s; ++dz) {
                    offs.push_back(Offset{dx, sy, dz});
                }
            }
        }
        for (const int sz : {-s, s}) {
            for (int dx = -s + 1; dx <= s - 1; ++dx) {
                for (int dy = -s + 1; dy <= s - 1; ++dy) {
                    offs.push_back(Offset{dx, dy, sz});
                }
            }
        }
        return shell_cache_.emplace(s, std::move(offs)).first->second;
    }

    /// Chebyshev open-ball interior for shell radius `s`: `|d|∞ < s` (cached).
    /// Empty when `s <= 1`.
    const std::vector<Offset>& interior_offsets_for(int step_size) const {
        const auto it = interior_cache_.find(step_size);
        if (it != interior_cache_.end()) {
            return it->second;
        }

        std::vector<Offset> offs;
        if (step_size > 1) {
            const int r = step_size - 1;
            offs.reserve(static_cast<std::size_t>((2 * r + 1) * (2 * r + 1) * (2 * r + 1)));
            for (int dz = -r; dz <= r; ++dz) {
                for (int dy = -r; dy <= r; ++dy) {
                    for (int dx = -r; dx <= r; ++dx) {
                        offs.push_back(Offset{dx, dy, dz});
                    }
                }
            }
        }
        return interior_cache_.emplace(step_size, std::move(offs)).first->second;
    }

    void successors(
        const grid::CellCoord& from,
        const geom::Heading& in_heading,
        std::vector<Successor>& out) const override {
        const std::vector<Offset>& offs = offsets_for(step_size_);
        out.reserve(out.size() + offs.size());

        for (const Offset& o : offs) {
            const double length =
                step_cost_ * std::sqrt(static_cast<double>(o.dx * o.dx + o.dy * o.dy + o.dz * o.dz));
            const geom::Heading heading = geom::Heading::from_grid_step(o.dx, o.dy, o.dz);
            if (!accept_turn(in_heading, heading)) {
                continue;
            }
            out.push_back(Successor{
                grid::CellCoord{from.x + o.dx, from.y + o.dy, from.z + o.dz},
                heading,
                combine_edge_cost(in_heading, heading, length),
            });
        }
    }

    /// If `goal` has Chebyshev distance in `(0, step_size)`, append a direct link.
    /// `cheb == 0` (goal == from) is a logic error — caller must not expand a goal.
    void append_interior_goal(
        const grid::CellCoord& from,
        const grid::CellCoord& goal,
        const geom::Heading& in_heading,
        std::vector<Successor>& out) const {
        const int dx = goal.x - from.x;
        const int dy = goal.y - from.y;
        const int dz = goal.z - from.z;
        const int cheb = std::max(std::abs(dx), std::max(std::abs(dy), std::abs(dz)));
        if (cheb == 0) {
            throw std::logic_error(
                "AdaptiveShell::append_interior_goal: goal coincides with from");
        }
        if (cheb >= step_size_) {
            return;
        }
        const geom::Heading heading = geom::Heading::from_grid_step(dx, dy, dz);
        if (!accept_turn(in_heading, heading)) {
            return;
        }
        const double length =
            step_cost_ * std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz));
        out.push_back(Successor{
            goal,
            heading,
            combine_edge_cost(in_heading, heading, length),
        });
    }

private:
    double step_cost_ = 1.0;
    int max_step_ = 4;
    int step_size_ = 1;
    mutable std::unordered_map<int, std::vector<Offset>> shell_cache_;
    mutable std::unordered_map<int, std::vector<Offset>> interior_cache_;
};

}  // namespace pipe_design::planner::motion
