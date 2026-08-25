#pragma once

#include "benchmark/GeneratorTypes.h"

namespace pipe_design::benchmark::pillar_field {

/// PillarField AABB fill (ROADMAP §2.2). Extends shared `ObstacleGenParams`.
struct GenParams : ObstacleGenParams {
    /// Reject grown boxes with any edge shorter than this (cells).
    int min_edge = 2;
    /// Cap any edge length during greedy expand (cells). Must be >= min_edge.
    int max_edge = 16;
    int candidates_per_round = 32;
    /// false: each face expands independently (maximal AABB).
    /// true: all six faces grow together one layer at a time (isometric).
    bool isometric = false;
    /// Cap on Chebyshev distance used as sampling weight (`<= 0` = uncapped).
    /// Prefer moderate values so far-field cells share the same weight.
    int dist_cap = 16;
};

}  // namespace pipe_design::benchmark::pillar_field
