#pragma once

#include "heuristic/Heuristic.h"

#include <cmath>

namespace pipe_design::heuristic {

/// Euclidean distance in cell units, scaled by step_cost.
class Euclidean final : public Heuristic {
public:
    explicit Euclidean(double step_cost = 1.0) : step_cost_(step_cost) {}

    double step_cost() const { return step_cost_; }

    double estimate(const grid::CellCoord& from, const grid::CellCoord& to) const override {
        const double dx = static_cast<double>(from.x - to.x);
        const double dy = static_cast<double>(from.y - to.y);
        const double dz = static_cast<double>(from.z - to.z);
        return step_cost_ * std::sqrt(dx * dx + dy * dy + dz * dz);
    }

private:
    double step_cost_ = 1.0;
};

}  // namespace pipe_design::heuristic
