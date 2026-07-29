#pragma once

#include "heuristic/Heuristic.h"

#include <cstdlib>

namespace pipe_design::heuristic {

/// 6-connect grid: step_cost * (|dx|+|dy|+|dz|).
class Manhattan final : public Heuristic {
public:
    explicit Manhattan(double step_cost = 1.0) : step_cost_(step_cost) {}

    double step_cost() const { return step_cost_; }

    double estimate(const grid::CellCoord& from, const grid::CellCoord& to) const override {
        const int dx = std::abs(from.x - to.x);
        const int dy = std::abs(from.y - to.y);
        const int dz = std::abs(from.z - to.z);
        return step_cost_ * static_cast<double>(dx + dy + dz);
    }

private:
    double step_cost_ = 1.0;
};

}  // namespace pipe_design::heuristic
