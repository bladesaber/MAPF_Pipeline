#pragma once

#include "heuristic/Heuristic.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace pipe_design::heuristic {

/// 26-connect (3D octile) distance, scaled by step_cost.
///
/// With sorted |Δ| as a ≤ b ≤ c:
///   cost = step_cost * (a·√3 + (b−a)·√2 + (c−b))
/// In the xy-plane (dz = 0) this reduces to classic 2D octile:
///   (√2−1)·min(dx,dy) + max(dx,dy).
/// Admissible for face/edge/corner moves with lengths 1 / √2 / √3.
class Octile final : public Heuristic {
public:
    explicit Octile(double step_cost = 1.0) : step_cost_(step_cost) {}

    double step_cost() const { return step_cost_; }

    double estimate(const grid::CellCoord& from, const grid::CellCoord& to) const override {
        int d[3] = {
            std::abs(from.x - to.x),
            std::abs(from.y - to.y),
            std::abs(from.z - to.z),
        };
        std::sort(d, d + 3);
        const double a = static_cast<double>(d[0]);
        const double b = static_cast<double>(d[1]);
        const double c = static_cast<double>(d[2]);
        constexpr double kSqrt2 = 1.4142135623730951;
        constexpr double kSqrt3 = 1.7320508075688772;
        return step_cost_ * (a * kSqrt3 + (b - a) * kSqrt2 + (c - b));
    }

private:
    double step_cost_ = 1.0;
};

}  // namespace pipe_design::heuristic
