#pragma once

#include "grid/Types.h"

namespace pipe_design::planner {

enum class NodeStatus {
    New,      // not yet discovered
    Open,     // in the open set
    Closed,   // already expanded
};

/// Search graph node. No open-set handle lives here — heaps keep their own side maps.
struct SearchNode {
    grid::CellCoord coord;
    double g = 0.0;
    double f = 0.0;
    SearchNode* parent = nullptr;
    NodeStatus status = NodeStatus::New;

    SearchNode() = default;
    explicit SearchNode(grid::CellCoord coord_) : coord(coord_) {}
};

}  // namespace pipe_design::planner
