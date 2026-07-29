#pragma once

#include "planner/motion/ActionModel.h"

namespace pipe_design::planner::motion {

/// Face-adjacent 6-connect; each move base length is `step_cost` (usually resolution).
class SixConnected final : public ActionModel {
public:
    explicit SixConnected(double step_cost = 1.0) : step_cost_(step_cost) {}

    using ActionModel::successors;

    double step_cost() const { return step_cost_; }

    void successors(
        const grid::CellCoord& from,
        const geom::Heading& in_heading,
        std::vector<Successor>& out) const override {
        static constexpr int kOffsets[6][3] = {
            {1, 0, 0},
            {-1, 0, 0},
            {0, 1, 0},
            {0, -1, 0},
            {0, 0, 1},
            {0, 0, -1},
        };
        out.reserve(out.size() + 6);
        for (const auto& o : kOffsets) {
            const geom::Heading heading = geom::Heading::from_grid_step(o[0], o[1], o[2]);
            out.push_back(Successor{
                grid::CellCoord{from.x + o[0], from.y + o[1], from.z + o[2]},
                heading,
                combine_edge_cost(in_heading, heading, step_cost_),
            });
        }
    }

private:
    double step_cost_ = 1.0;
};

}  // namespace pipe_design::planner::motion
