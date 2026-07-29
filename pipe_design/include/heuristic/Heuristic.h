#pragma once

#include "grid/Types.h"

namespace pipe_design::heuristic {

/// Admissible remaining-cost estimate for best-first search.
///
/// Edge costs and heuristic must use the same scale (usually grid resolution).
/// Stateful heuristics (true-distance, landmarks) may add prepare(goal) later;
/// A* holds a non-owning pointer — keep the object alive for the search.
class Heuristic {
public:
    virtual ~Heuristic() = default;

    virtual double estimate(
        const grid::CellCoord& from,
        const grid::CellCoord& to) const = 0;
};

}  // namespace pipe_design::heuristic
