#pragma once

#include "planner/motion/ActionModel.h"

namespace pipe_design::planner::motion {

/// All 26 neighbors in a 3×3×3 cube (excluding self).
/// Edge base lengths match Octile / 3D diagonal metric:
///   face  (1 nonzero axis)  → step_cost · 1
///   edge  (2 nonzero axes)  → step_cost · √2
///   corner (3 nonzero axes) → step_cost · √3
class TwentySixConnected final : public ActionModel {
public:
    explicit TwentySixConnected(double step_cost = 1.0) : step_cost_(step_cost) {}

    using ActionModel::successors;

    double step_cost() const { return step_cost_; }

    void successors(
        const grid::CellCoord& from,
        const geom::Heading& in_heading,
        std::vector<Successor>& out) const override {
        constexpr double kSqrt2 = 1.4142135623730951;
        constexpr double kSqrt3 = 1.7320508075688772;

        out.reserve(out.size() + 26);
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    const int axes = (dx != 0) + (dy != 0) + (dz != 0);
                    double scale = 1.0;
                    if (axes == 2) {
                        scale = kSqrt2;
                    } else if (axes == 3) {
                        scale = kSqrt3;
                    }
                    const geom::Heading heading = geom::Heading::from_grid_step(dx, dy, dz);
                    out.push_back(Successor{
                        grid::CellCoord{from.x + dx, from.y + dy, from.z + dz},
                        heading,
                        combine_edge_cost(in_heading, heading, step_cost_ * scale),
                    });
                }
            }
        }
    }

private:
    double step_cost_ = 1.0;
};

}  // namespace pipe_design::planner::motion
