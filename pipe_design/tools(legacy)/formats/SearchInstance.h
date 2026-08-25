#pragma once

#include "geom/Types.h"
#include "grid/StaticClearanceGrid.h"
#include "grid/Types.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace pipe_design::tools {

/// One multi-start / multi-goal query (aligns with AStar::reset(starts, goals, …)).
struct SearchQuery {
    int query_id = 0;
    std::vector<grid::CellCoord> starts;
    std::vector<grid::CellCoord> goals;
    std::optional<double> reference_cost;
};

/// Low-level file-bridge input (`SearchInstance`). Not for high-level design —
/// see ROADMAP §5.2 `DesignInstance` (future).
/// Python writes this; C++ loads it. Does not modify planner APIs.
///
/// Default `step_cost` for action/heuristic = `resolution` (world cell length);
/// override via *_kwargs if needed.
///
/// Schema (version 1), abbreviated:
/// {
///   "version": 1,
///   "instance_id": "dao/arena/q0-batch",
///   "map_id": "dao/arena",
///   "size": [sx, sy, sz],
///   "resolution": 1.0,
///   "clearance_cap": 1.0,
///   "pipe_radius": 0.1,
///   "action_model": "EightConnected",
///   "action_model_kwargs": {"step_cost": 1.0, "turn_weight": 0.0},
///   "heuristic": "manhattan",
///   "heuristic_kwargs": {"step_cost": 1.0},
///   "heuristic_id": "",
///   "heap": "LazyBinaryHeap",
///   "heap_kwargs": {},
///   "heap_id": "",
///   "model": "astar",
///   "model_kwargs": {"heuristic_weight": 1.0},
///   // also: "directed_astar", "greedy", "dijkstra",
///   //       "theta_star", "lazy_theta_star" (+ optional step_cost)
///   "model_id": "",
///   "max_expansions": null,
///   "max_time_ms": 0,
///   "output_dir": "runs/smoke",
///   "export_closed_set": false,
///   "export_posthoc": false,
///   "obstacles": [
///     {"mode": "sphere", "cell_radius": 0.1, "cells": [[x,y,z], ...]},
///     {"mode": "capsules", "cell_radius": 0.1,
///      "from_point": [x,y,z], "to_point": [x,y,z]}
///   ],
///   "queries": [
///     {"query_id": 0,
///      "start": [[x,y,z], ...],   // or shorthand [x,y,z]
///      "goal":  [[x,y,z], ...],
///      "reference_cost": 3.2}
///   ]
/// }
struct SearchInstance {
    static constexpr int kSchemaVersion = 1;

    int version = kSchemaVersion;
    std::string instance_id;
    std::string map_id;

    int size_x = 0;
    int size_y = 0;
    int size_z = 1;

    float resolution = 1.f;
    float clearance_cap = 1.f;
    float pipe_radius = 0.1f;

    std::string action_model = "EightConnected";
    nlohmann::json action_model_kwargs = nlohmann::json::object();

    std::string heuristic = "manhattan";
    nlohmann::json heuristic_kwargs = nlohmann::json::object();
    /// Preset provenance id (e.g. from search_presets); empty if unused.
    std::string heuristic_id;

    /// "LazyBinaryHeap" | "BinaryHeap" | "PairingHeap" | "BucketHeap"
    std::string heap = "LazyBinaryHeap";
    nlohmann::json heap_kwargs = nlohmann::json::object();
    std::string heap_id;

    std::string model = "astar";
    nlohmann::json model_kwargs = nlohmann::json::object();
    std::string model_id;

    /// Cap on A* expansions; default = unlimited.
    std::size_t max_expansions = std::numeric_limits<std::size_t>::max();

    /// Cap on accumulated planner.step() time per query (ms); <= 0 means unlimited.
    double max_time_ms = 0.0;

    /// Directory for reports / closed / posthoc dumps (empty = no files written).
    std::string output_dir;

    /// If true, write closed-set expansion order under output_dir/closed_set/.
    bool export_closed_set = false;

    /// If true, write Posthoc-ready traces under output_dir/posthoc/ (requires size_z==1).
    bool export_posthoc = false;

    std::vector<geom::SphereObstacle> spheres;
    std::vector<geom::CapsuleObstacle> capsules;
    std::vector<geom::AabbObstacle> aabbs;
    std::vector<SearchQuery> queries;

    static SearchInstance from_file(const std::string& path);
    static SearchInstance from_json_string(const std::string& text);

    void to_file(const std::string& path) const;
    std::string to_json_string(int indent = 2) const;

    grid::StaticClearanceGrid make_static_grid() const;

    const std::vector<geom::SphereObstacle>& bake_spheres() const { return spheres; }
    const std::vector<geom::CapsuleObstacle>& bake_capsules() const { return capsules; }
    const std::vector<geom::AabbObstacle>& bake_aabbs() const { return aabbs; }
};

}  // namespace pipe_design::tools
