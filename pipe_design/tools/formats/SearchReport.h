#pragma once

#include "grid/Types.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace pipe_design::tools {

/// One query result from a low/mid search run (ROADMAP §5.3).
/// Python / viz consume this; C++ runner fills it.
struct SearchReport {
    static constexpr int kSchemaVersion = 1;

    int version = kSchemaVersion;
    std::string instance_id;
    std::string map_id;
    int query_id = 0;

    bool success = false;
    /// "Success" | "Failure" | "Running" (e.g. hit expansion / time cap)
    std::string status = "Failure";

    double total_time_ms = 0.0;  // sum of planner.step() only
    std::size_t expansions = 0;
    /// Nodes created in the search map (proxy for generated when no finer counter).
    std::size_t generated = 0;
    /// Not instrumented in A* yet — left 0 until planner exposes reopen count.
    std::size_t reexpansions = 0;
    std::size_t open_peak = 0;
    std::size_t open_final = 0;
    /// Nodes with Closed status; falls back to expansions when not counted.
    std::size_t closed = 0;
    std::size_t nodes_stored = 0;
    /// Rough bytes: nodes_stored * bytes_per_node + open_final * 16.
    std::size_t bytes_estimate = 0;

    double path_cost = 0.0;
    std::size_t path_length = 0;
    std::size_t path_turns = 0;
    std::vector<grid::CellCoord> path;

    std::string model;
    std::string heuristic;
    std::string action_model;
    std::string heap;
    /// Preset provenance ids copied from SearchInstance (empty if unused).
    std::string model_id;
    std::string heap_id;
    std::string heuristic_id;

    std::optional<double> opt_cost;
    std::optional<double> reference_cost;

    void to_file(const std::string& path) const;
    std::string to_json_string(int indent = -1) const;
};

/// Write one JSON object per line (no pretty indent).
void write_search_reports_jsonl(
    const std::string& path,
    const std::vector<SearchReport>& reports);

}  // namespace pipe_design::tools
