#include "runners/LowLevelRunner.h"

#include "formats/SearchReport.h"
#include "grid/CompositeGrid.h"
#include "grid/StaticClearanceGrid.h"
#include "heuristic/Dijkstra.h"
#include "heuristic/Euclidean.h"
#include "heuristic/Manhattan.h"
#include "heuristic/Octile.h"
#include "planner/low/AStar.h"
#include "planner/low/Dijkstra.h"
#include "planner/low/DirectedAStar.h"
#include "planner/low/GreedyBestFirst.h"
#include "planner/low/LazyThetaStar.h"
#include "planner/low/SearchNode.h"
#include "planner/low/ThetaStar.h"
#include "planner/motion/EightConnected.h"
#include "planner/motion/SixConnected.h"
#include "planner/motion/TwentySixConnected.h"
#include "planner/open/BinaryHeap.h"
#include "planner/open/BucketHeap.h"
#include "planner/open/LazyBinaryHeap.h"
#include "planner/open/PairingHeap.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pipe_design::tools {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

using pipe_design::grid::CompositeGrid;
using pipe_design::grid::StaticClearanceGrid;
using pipe_design::planner::AStar;
using pipe_design::planner::Dijkstra;
using pipe_design::planner::DirectedAStar;
using pipe_design::planner::GreedyBestFirst;
using pipe_design::planner::LazyThetaStar;
using pipe_design::planner::OpenSet;
using pipe_design::planner::SearchNode;
using pipe_design::planner::SearchStatus;
using pipe_design::planner::ThetaStar;
using pipe_design::planner::heading_between;

void ensure_output_dir(const std::string& dir) {
    if (dir.empty()) {
        return;
    }
    const fs::path path(dir);
    if (fs::exists(path)) {
        return;
    }
    fs::create_directories(path);
}

double kw_double(const nlohmann::json& kw, const char* key, double def) {
    if (kw.contains(key) && !kw[key].is_null()) {
        return kw[key].get<double>();
    }
    return def;
}

int kw_int(const nlohmann::json& kw, const char* key, int def) {
    if (kw.contains(key) && !kw[key].is_null()) {
        return kw[key].get<int>();
    }
    return def;
}

std::unique_ptr<planner::motion::ActionModel> make_action_model(const SearchInstance& instance) {
    const double step =
        kw_double(instance.action_model_kwargs, "step_cost", instance.resolution);
    std::unique_ptr<planner::motion::ActionModel> actions;
    if (instance.action_model == "SixConnected") {
        actions = std::make_unique<planner::motion::SixConnected>(step);
    } else if (instance.action_model == "EightConnected") {
        actions = std::make_unique<planner::motion::EightConnected>(step);
    } else if (instance.action_model == "TwentySixConnected") {
        actions = std::make_unique<planner::motion::TwentySixConnected>(step);
    } else {
        throw std::runtime_error("LowLevelRunner: unknown action_model " + instance.action_model);
    }
    actions->set_turn_weight(kw_double(instance.action_model_kwargs, "turn_weight", 0.0));
    return actions;
}

}  // namespace

/// Per-query factory: geometric heuristics ignore `q`; Dijkstra prepares on `q.goals`.
std::unique_ptr<heuristic::Heuristic> make_heuristic(
    const SearchInstance& instance,
    const grid::StaticClearanceGrid& static_grid,
    const SearchQuery& q) {
    const double step =
        kw_double(instance.heuristic_kwargs, "step_cost", instance.resolution);
    if (instance.heuristic == "manhattan") {
        return std::make_unique<heuristic::Manhattan>(step);
    }
    if (instance.heuristic == "euclidean") {
        return std::make_unique<heuristic::Euclidean>(step);
    }
    if (instance.heuristic == "octile") {
        return std::make_unique<heuristic::Octile>(step);
    }
    if (instance.heuristic == "dijkstra") {
        const float bake_radius = static_cast<float>(
            kw_double(instance.heuristic_kwargs, "bake_radius", 0.0));
        auto h = std::make_unique<heuristic::Dijkstra>(static_grid, step, bake_radius);
        h->prepare(q.goals);
        return h;
    }
    throw std::runtime_error("LowLevelRunner: unknown heuristic " + instance.heuristic);
}

namespace {

std::unique_ptr<OpenSet> make_heap(const SearchInstance& instance) {
    if (instance.heap == "LazyBinaryHeap") {
        return std::make_unique<planner::LazyBinaryHeap>();
    }
    if (instance.heap == "BinaryHeap") {
        return std::make_unique<planner::BinaryHeap>();
    }
    if (instance.heap == "PairingHeap") {
        return std::make_unique<planner::PairingHeap>();
    }
    if (instance.heap == "BucketHeap") {
        return std::make_unique<planner::BucketHeap>(
            kw_double(instance.heap_kwargs, "delta", 1.0),
            kw_double(
                instance.heap_kwargs, "max_cost", std::numeric_limits<double>::infinity()));
    }
    throw std::runtime_error("LowLevelRunner: unknown heap " + instance.heap);
}

// Report Utilities
std::size_t count_path_turns(const std::vector<grid::CellCoord>& cells) {
    if (cells.size() < 3) {
        return 0;
    }
    std::size_t turns = 0;
    for (std::size_t i = 2; i < cells.size(); ++i) {
        const auto h0 = heading_between(cells[i - 2], cells[i - 1]);
        const auto h1 = heading_between(cells[i - 1], cells[i]);
        if (!h0.is_none() && !h1.is_none() && !h0.nearly_equal(h1)) {
            ++turns;
        }
    }
    return turns;
}

template <typename Planner>
std::size_t count_closed(const Planner& planner) {
    std::size_t n = 0;
    for (const auto& kv : planner.nodes()) {
        if (kv.second && kv.second->status == planner::NodeStatus::Closed) {
            ++n;
        }
    }
    return n;
}

template <typename Planner>
SearchReport fill_report(
    const SearchInstance& instance,
    const SearchQuery& q,
    Planner& planner,
    OpenSet& open,
    SearchStatus status,
    double time_ms,
    std::size_t open_peak) {
    SearchReport r;
    r.instance_id = instance.instance_id;
    r.map_id = instance.map_id;
    r.query_id = q.query_id;
    r.success = (status == SearchStatus::Success);
    if (status == SearchStatus::Success) {
        r.status = "Success";
    } else if (status == SearchStatus::Running) {
        r.status = "Running";
    } else {
        r.status = "Failure";
    }
    r.total_time_ms = time_ms;
    r.expansions = planner.expansions();
    r.nodes_stored = planner.nodes().size();
    r.generated = r.nodes_stored;
    r.reexpansions = 0;
    r.open_peak = open_peak;
    r.open_final = open.size();
    r.closed = count_closed(planner);
    constexpr std::size_t kBytesPerNode = 64;
    r.bytes_estimate = r.nodes_stored * kBytesPerNode + r.open_final * 16;

    if (status == SearchStatus::Success) {
        const auto path = planner.path();
        r.path = path.cells;
        r.path_cost = path.cost;
        r.path_length = path.length();
        r.path_turns = count_path_turns(path.cells);
    }

    r.model = instance.model;
    r.heuristic = instance.heuristic;
    r.action_model = instance.action_model;
    r.heap = instance.heap;
    r.model_id = instance.model_id;
    r.heap_id = instance.heap_id;
    r.heuristic_id = instance.heuristic_id;
    r.reference_cost = q.reference_cost;
    return r;
}

void write_closed_set(
    const SearchInstance& instance,
    int query_id,
    const std::vector<std::pair<grid::CellCoord, std::size_t>>& closed_order) {
    json j;
    j["instance_id"] = instance.instance_id;
    j["map_id"] = instance.map_id;
    j["query_id"] = query_id;
    j["size"] = json::array({instance.size_x, instance.size_y, instance.size_z});
    json closed = json::array();
    for (const auto& [cell, t] : closed_order) {
        closed.push_back({{"x", cell.x}, {"y", cell.y}, {"z", cell.z}, {"t", t}});
    }
    j["closed"] = std::move(closed);

    const fs::path dir = fs::path(instance.output_dir) / "closed_set";
    ensure_output_dir(dir.string());
    const fs::path path = dir / ("closed_q" + std::to_string(query_id) + ".json");
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("LowLevelRunner: cannot write " + path.string());
    }
    out << j.dump(2);
}

template <typename Planner>
void write_posthoc_trace(
    const SearchInstance& instance,
    int query_id,
    Planner& planner) {
    if (instance.size_z != 1) {
        throw std::runtime_error(
            "LowLevelRunner: export_posthoc requires size_z == 1 (Posthoc is 2D); got size_z=" +
            std::to_string(instance.size_z));
    }

    // Assign a stable integer id to each cell that appears in the search map.
    std::unordered_map<grid::CellCoord, int> cell_id;
    cell_id.reserve(planner.nodes().size());
    int next_id = 0;
    for (const auto& kv : planner.nodes()) {
        if (!kv.second) {
            continue;
        }
        const grid::CellCoord& c = kv.second->coord;
        if (cell_id.find(c) == cell_id.end()) {
            cell_id.emplace(c, next_id++);
        }
    }

    json events = json::array();
    for (const auto& kv : planner.nodes()) {
        if (!kv.second) {
            continue;
        }
        const auto& n = *kv.second;
        const int id = cell_id.at(n.coord);
        const bool is_end = planner.is_goal(n.coord);

        if (!n.parent) {
            events.push_back({
                {"type", "source"},
                {"id", id},
                {"x", n.coord.x},
                {"y", n.coord.y},
            });
            events.push_back({
                {"type", "expand"},
                {"id", id},
                {"f", n.f},
                {"g", n.g},
                {"x", n.coord.x},
                {"y", n.coord.y},
            });
        } else {
            const int pid = cell_id.at(n.parent->coord);
            events.push_back({
                {"type", "expand"},
                {"id", id},
                {"f", n.f},
                {"g", n.g},
                {"x", n.coord.x},
                {"y", n.coord.y},
                {"pId", pid},
            });
        }

        if (is_end) {
            events.push_back({
                {"type", "end"},
                {"id", id},
                {"f", n.f},
                {"g", n.g},
                {"x", n.coord.x},
                {"y", n.coord.y},
            });
        }
    }

    // Real solution path: walk goal_node parent chain (start → goal), then end.
    // Posthoc draws the path via id/pId; type "path" gets a distinct color in viz.
    if (const auto* gn = planner.goal_node()) {
        std::vector<const SearchNode*> chain;
        for (const SearchNode* n = gn; n != nullptr; n = n->parent) {
            chain.push_back(n);
        }
        std::reverse(chain.begin(), chain.end());

        for (std::size_t i = 0; i < chain.size(); ++i) {
            const SearchNode& n = *chain[i];
            const bool is_last = (i + 1 == chain.size());
            json e = {
                {"type", is_last ? "end" : "path"},
                {"id", cell_id.at(n.coord)},
                {"f", n.f},
                {"g", n.g},
                {"x", n.coord.x},
                {"y", n.coord.y},
            };
            if (n.parent) {
                e["pId"] = cell_id.at(n.parent->coord);
            }
            events.push_back(std::move(e));
        }
    }

    json j;
    j["version"] = "1.4.0";
    j["instance_id"] = instance.instance_id;
    j["map_id"] = instance.map_id;
    j["query_id"] = query_id;
    j["events"] = std::move(events);

    const fs::path dir = fs::path(instance.output_dir) / "posthoc";
    ensure_output_dir(dir.string());
    const fs::path path = dir / ("trace_q" + std::to_string(query_id) + ".json");
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("LowLevelRunner: cannot write " + path.string());
    }
    out << j.dump(2);
}

// Main Runner
template <typename Planner>
std::vector<SearchReport> run_queries(
    const SearchInstance& instance,
    const CompositeGrid& grid,
    const StaticClearanceGrid& static_grid,
    OpenSet& open,
    const planner::motion::ActionModel& actions,
    Planner& planner) {
    std::vector<SearchReport> reports;
    reports.reserve(instance.queries.size());

    for (const SearchQuery& q : instance.queries) {
        auto heuristic = make_heuristic(instance, static_grid, q);
        planner.reset(
            grid, q.starts, q.goals, instance.pipe_radius, *heuristic, open, actions);

        SearchStatus status = SearchStatus::Running;
        std::size_t open_peak = open.size();
        double step_time_ms = 0.0;  // sum of planner.step() only (excludes export / limits)
        std::vector<std::pair<grid::CellCoord, std::size_t>> closed_order;
        if (instance.export_closed_set) {
            closed_order.reserve(1024);
        }

        while (status == SearchStatus::Running) {
            open_peak = std::max(open_peak, open.size());
            if (instance.max_time_ms > 0.0 && step_time_ms >= instance.max_time_ms) {
                break;
            }
            if (planner.expansions() >= instance.max_expansions) {
                break;
            }
            const auto step_t0 = std::chrono::steady_clock::now();
            status = planner.step();
            step_time_ms += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - step_t0)
                                .count();
            if (instance.export_closed_set && planner.last_expanded() != nullptr) {
                closed_order.emplace_back(
                    planner.last_expanded()->coord, planner.expansions());
            }
        }

        SearchReport report =
            fill_report(instance, q, planner, open, status, step_time_ms, open_peak);
        std::cout << "query_id=" << report.query_id << " status=" << report.status
                  << " expansions=" << report.expansions << " path_cost=" << report.path_cost
                  << " path_length=" << report.path_length << " time_ms=" << report.total_time_ms
                  << '\n';

        if (!instance.output_dir.empty()) {
            if (instance.export_closed_set) {
                write_closed_set(instance, q.query_id, closed_order);
            }
            if (instance.export_posthoc) {
                write_posthoc_trace(instance, q.query_id, planner);
            }
        }

        reports.push_back(std::move(report));
    }

    return reports;
}

}  // namespace

std::vector<SearchReport> LowLevelRunner::run(const SearchInstance& instance) const {
    auto static_grid = instance.make_static_grid();
    static_grid.bake(instance.bake_spheres(), instance.bake_capsules());
    CompositeGrid grid(static_grid);

    auto actions = make_action_model(instance);
    auto open = make_heap(instance);
    const double weight = kw_double(instance.model_kwargs, "heuristic_weight", 1.0);

    ensure_output_dir(instance.output_dir);

    std::vector<SearchReport> reports;
    if (instance.model == "astar") {
        AStar<SearchNode> planner;
        planner.set_heuristic_weight(weight);
        reports = run_queries(instance, grid, static_grid, *open, *actions, planner);
    } else if (instance.model == "directed_astar") {
        DirectedAStar planner;
        planner.set_heading_bins(
            kw_int(instance.model_kwargs, "num_yaw_bins", 8),
            kw_int(instance.model_kwargs, "num_pitch_bins", 1));
        planner.set_heuristic_weight(weight);
        reports = run_queries(instance, grid, static_grid, *open, *actions, planner);
    } else if (instance.model == "greedy") {
        GreedyBestFirst planner;
        reports = run_queries(instance, grid, static_grid, *open, *actions, planner);
    } else if (instance.model == "dijkstra") {
        Dijkstra planner;
        reports = run_queries(instance, grid, static_grid, *open, *actions, planner);
    } else if (instance.model == "theta_star") {
        ThetaStar planner;
        planner.set_heuristic_weight(weight);
        planner.set_step_cost(
            kw_double(instance.model_kwargs, "step_cost", instance.resolution));
        reports = run_queries(instance, grid, static_grid, *open, *actions, planner);
    } else if (instance.model == "lazy_theta_star") {
        LazyThetaStar planner;
        planner.set_heuristic_weight(weight);
        planner.set_step_cost(
            kw_double(instance.model_kwargs, "step_cost", instance.resolution));
        reports = run_queries(instance, grid, static_grid, *open, *actions, planner);
    } else {
        throw std::runtime_error("LowLevelRunner: unsupported model " + instance.model);
    }

    if (!instance.output_dir.empty()) {
        const std::filesystem::path report_path =
            std::filesystem::path(instance.output_dir) / "reports.jsonl";
        write_search_reports_jsonl(report_path.string(), reports);
    }

    return reports;
}

}  // namespace pipe_design::tools
