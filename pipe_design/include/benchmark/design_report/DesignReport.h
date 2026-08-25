#pragma once

#include "planner/low/Path.h"
#include "topology/Types.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace pipe_design::benchmark {

/// One Task's final path in a DesignReport group result.
struct DesignTaskResult {
    std::string task_id;
    std::string edge_id;
    bool success = false;
    planner::Path path;  // cells + cost
};

/// One group's final routing result (paths, not summary-only).
struct DesignGroupResult {
    topology::GroupId group_id;
    bool success = false;
    std::vector<DesignTaskResult> tasks;
};

/// One high-level design solve → one JSON object (batch = JSONL).
///
/// Experiment focus: fixed budget mode (`time` or `iterations`) → success rate.
/// Provenance: `instance_id` + `instance_path` (+ optional run `meta` from CLI).
///
/// Schema (version 1), abbreviated:
/// {
///   "version": 1,
///   "instance_id": "axis_ray/easy/instance_0",
///   "instance_path": "/path/to/instance_0.yaml",
///   "meta": {"generator": "axis_ray", "level": "easy"},
///   "algorithm": "cbs",
///   "topology": "pure_mst",
///   "low_model": "astar",
///   "heuristic": "octile",
///   "action_model": "TwentySixConnected",
///   "conflict_analyzer": "near_terminal",
///   "algorithm_kwargs": {},
///   "budget_mode": "time",
///   "max_time_ms": 60000,
///   "max_iterations": 10000,
///   "success": true,
///   "status": "Success",
///   "fail_reason": "",
///   "total_time_ms": 1234.5,
///   "iterations": 42,
///   "total_cost": 180.5,
///   "total_length": 180,
///   "num_tasks": 4,
///   "num_tasks_success": 4,
///   "witness_compared": true,
///   "reference_cost": 160.0,
///   "cost_gap": 20.5,
///   "cost_ratio": 1.128,
///   "groups": [
///     {"group_id": "G0", "success": true,
///      "tasks": [{"task_id": "G0:e0", "edge_id": "e0", "success": true,
///                "cost": 10.0, "cells": [[x,y,z], ...]}]}
///   ]
/// }
struct DesignReport {
    static constexpr int kSchemaVersion = 1;

    int version = kSchemaVersion;

    /// Provenance (instance YAML holds generator params / size / …).
    std::string instance_id;
    std::string instance_path;

    /// Free-form run tags from CLI `--meta` (JSON object). Not solver config.
    nlohmann::json meta = nlohmann::json::object();

    /// Solver config (echo from the run).
    std::string algorithm;
    std::string topology;
    std::string low_model;
    std::string heuristic;
    std::string action_model;
    std::string conflict_analyzer;
    nlohmann::json algorithm_kwargs = nlohmann::json::object();

    /// Budget: exactly one active axis via `budget_mode`.
    /// "time" → enforce `max_time_ms` only; "iterations" → enforce `max_iterations` only.
    std::string budget_mode = "time";
    double max_time_ms = 0.0;  // used when budget_mode == "time"; <= 0 → unlimited
    std::size_t max_iterations = std::numeric_limits<std::size_t>::max();  // used when budget_mode == "iterations"

    /// Outcome.
    bool success = false;
    /// "Success" | "Failure" | "Running" (hit time / iteration cap).
    std::string status = "Failure";
    /// "" | "timeout" | "iteration_cap" | "no_solution" | …
    std::string fail_reason;
    double total_time_ms = 0.0;
    std::size_t iterations = 0;

    /// Aggregates over `groups` (runner may call `recompute_totals()`).
    double total_cost = 0.0;
    std::size_t total_length = 0;
    std::size_t num_tasks = 0;
    std::size_t num_tasks_success = 0;

    /// Vs witness (constructive YAML `solution`). Filled by `compare_witness()`.
    bool witness_compared = false;
    double reference_cost = 0.0;
    double cost_gap = 0.0;     // total_cost - reference_cost (success only)
    double cost_ratio = 0.0;   // total_cost / reference_cost (success only)

    /// Per-group final paths.
    std::vector<DesignGroupResult> groups;

    /// Fill `groups` from a high-level `solution()` Task list (grouped by `group_id`).
    void set_groups_from_tasks(const std::vector<topology::Task>& tasks);

    /// Recompute aggregate fields from `groups`.
    void recompute_totals();

    /// Load witness from `instance_path` via `Instance::from_file`; fill reference_*.
    void compare_witness();

    void to_file(const std::string& path) const;
    std::string to_json_string(int indent = -1) const;
};

/// Write one JSON object per line (no pretty indent).
void write_design_reports_jsonl(
    const std::string& path,
    const std::vector<DesignReport>& reports);

}  // namespace pipe_design::benchmark
