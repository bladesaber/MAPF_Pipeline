#pragma once

#include "planner/low/AStar.h"

namespace pipe_design::planner {

/// Greedy best-first search: open priority uses only the heuristic (`f = h`).
/// Same loop / expand as A*; path cost `g` is still tracked for reporting.
class GreedyBestFirst : public AStar<SearchNode> {
protected:
    double combine_key(double /*g*/, double h) const override { return h; }
};

}  // namespace pipe_design::planner
