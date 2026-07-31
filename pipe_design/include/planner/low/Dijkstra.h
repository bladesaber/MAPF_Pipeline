#pragma once

#include "planner/low/AStar.h"

namespace pipe_design::planner {

/// Dijkstra / uniform-cost search: open priority uses only path cost (`f = g`).
/// Distinct from `heuristic::Dijkstra` (true-distance field bake).
/// Heuristic is still accepted by `reset` but ignored for ranking.
class Dijkstra : public AStar<SearchNode> {
protected:
    double combine_key(double g, double /*h*/) const override { return g; }
};

}  // namespace pipe_design::planner
