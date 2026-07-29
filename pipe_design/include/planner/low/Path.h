#pragma once

#include "grid/Types.h"

#include <vector>

namespace pipe_design::planner {

/// Discrete path on the search grid (start → … → goal).
struct Path {
    std::vector<grid::CellCoord> cells;
    double cost = 0.0;

    bool empty() const { return cells.empty(); }
    std::size_t length() const { return cells.size(); }
};

}  // namespace pipe_design::planner
