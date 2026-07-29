#pragma once

#include "planner/motion/ActionModel.h"

namespace pipe_design::planner::motion {

/// Planar 8-connect (same z): 4 face + 4 diagonal neighbors.
///   face     → step_cost · 1
///   diagonal → step_cost · √2
/// Pairs with 2D Octile (dz = 0). For full 3D diagonals use TwentySixConnected.
class EightConnected final : public ActionModel {
public:
    explicit EightConnected(double step_cost = 1.0) : step_cost_(step_cost) {}

    using ActionModel::successors;

    double step_cost() const { return step_cost_; }

    void successors(
        const grid::CellCoord& from,
        const geom::Heading& in_heading,
        std::vector<Successor>& out) const override {
        constexpr double kSqrt2 = 1.4142135623730951;
        static constexpr int kOffsets[8][2] = {
            {1, 0},
            {-1, 0},
            {0, 1},
            {0, -1},
            {1, 1},
            {1, -1},
            {-1, 1},
            {-1, -1},
        };

        out.reserve(out.size() + 8);
        for (const auto& o : kOffsets) {
            const int axes = (o[0] != 0) + (o[1] != 0);
            const double length = step_cost_ * (axes == 2 ? kSqrt2 : 1.0);
            const geom::Heading heading = geom::Heading::from_grid_step(o[0], o[1], 0);
            out.push_back(Successor{
                grid::CellCoord{from.x + o[0], from.y + o[1], from.z},
                heading,
                combine_edge_cost(in_heading, heading, length),
            });
        }
    }

private:
    double step_cost_ = 1.0;
};

}  // namespace pipe_design::planner::motion
