#include "formats/SearchReport.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace pipe_design::tools {
namespace {

using json = nlohmann::json;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("SearchReport: " + msg);
}

json cell_to_json(const grid::CellCoord& c) {
    return json::array({c.x, c.y, c.z});
}

json dump(const SearchReport& r) {
    json j = {
        {"version", r.version},
        {"instance_id", r.instance_id},
        {"map_id", r.map_id},
        {"query_id", r.query_id},
        {"success", r.success},
        {"status", r.status},
        {"total_time_ms", r.total_time_ms},
        {"expansions", r.expansions},
        {"generated", r.generated},
        {"reexpansions", r.reexpansions},
        {"open_peak", r.open_peak},
        {"open_final", r.open_final},
        {"closed", r.closed},
        {"nodes_stored", r.nodes_stored},
        {"bytes_estimate", r.bytes_estimate},
        {"path_cost", r.path_cost},
        {"path_length", r.path_length},
        {"path_turns", r.path_turns},
        {"model", r.model},
        {"heuristic", r.heuristic},
        {"action_model", r.action_model},
        {"heap", r.heap},
        {"model_id", r.model_id},
        {"heap_id", r.heap_id},
        {"heuristic_id", r.heuristic_id},
    };
    if (r.reference_cost) {
        j["reference_cost"] = *r.reference_cost;
    }
    json path = json::array();
    for (const grid::CellCoord& c : r.path) {
        path.push_back(cell_to_json(c));
    }
    j["path"] = std::move(path);
    return j;
}

}  // namespace

void SearchReport::to_file(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        fail("cannot write file: " + path);
    }
    out << to_json_string(2);
}

std::string SearchReport::to_json_string(int indent) const {
    return dump(*this).dump(indent);
}

void write_search_reports_jsonl(
    const std::string& path,
    const std::vector<SearchReport>& reports) {
    std::ofstream out(path);
    if (!out) {
        fail("cannot write file: " + path);
    }
    for (const SearchReport& r : reports) {
        out << r.to_json_string(-1) << '\n';
    }
}

}  // namespace pipe_design::tools
