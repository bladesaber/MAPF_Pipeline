#include "formats/SearchInstance.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace pipe_design::tools {
namespace {

using json = nlohmann::json;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("SearchInstance: " + msg);
}

grid::CellCoord read_cell(const json& a) {
    return {
        a.at(0).get<int>(),
        a.at(1).get<int>(),
        a.size() >= 3 ? a.at(2).get<int>() : 0,
    };
}

/// `[x,y,z]` or `[[x,y,z], ...]`.
std::vector<grid::CellCoord> read_cells(const json& j) {
    if (!j.is_array() || j.empty()) {
        fail("cell list empty");
    }
    if (j.front().is_number()) {
        return {read_cell(j)};
    }
    std::vector<grid::CellCoord> out;
    out.reserve(j.size());
    for (const json& item : j) {
        out.push_back(read_cell(item));
    }
    return out;
}

geom::Vec3 read_vec3(const json& a) {
    return {
        a.at(0).get<float>(),
        a.at(1).get<float>(),
        a.size() >= 3 ? a.at(2).get<float>() : 0.f,
    };
}

SearchInstance parse(const json& j) {
    SearchInstance inst;
    inst.version = j.value("version", SearchInstance::kSchemaVersion);
    inst.instance_id = j.value("instance_id", std::string{});
    inst.map_id = j.value("map_id", std::string{});

    const json& size = j.at("size");
    inst.size_x = size.at(0).get<int>();
    inst.size_y = size.at(1).get<int>();
    inst.size_z = size.size() >= 3 ? size.at(2).get<int>() : 1;

    inst.resolution = j.value("resolution", 1.f);
    inst.clearance_cap = j.value("clearance_cap", 1.f);
    inst.pipe_radius = j.value("pipe_radius", 0.1f);
    inst.action_model = j.value("action_model", std::string{"EightConnected"});
    if (j.contains("action_model_kwargs") && j["action_model_kwargs"].is_object()) {
        inst.action_model_kwargs = j["action_model_kwargs"];
    }
    inst.heuristic = j.value("heuristic", std::string{"manhattan"});
    if (j.contains("heuristic_kwargs") && j["heuristic_kwargs"].is_object()) {
        inst.heuristic_kwargs = j["heuristic_kwargs"];
    }
    inst.heuristic_id = j.value("heuristic_id", std::string{});
    inst.heap = j.value("heap", std::string{"LazyBinaryHeap"});
    if (j.contains("heap_kwargs") && j["heap_kwargs"].is_object()) {
        inst.heap_kwargs = j["heap_kwargs"];
    }
    inst.heap_id = j.value("heap_id", std::string{});
    inst.model = j.value("model", std::string{"astar"});
    if (j.contains("model_kwargs") && j["model_kwargs"].is_object()) {
        inst.model_kwargs = j["model_kwargs"];
    }
    inst.model_id = j.value("model_id", std::string{});
    if (j.contains("max_expansions") && !j["max_expansions"].is_null()) {
        inst.max_expansions = j["max_expansions"].get<std::size_t>();
    }
    if (j.contains("max_time_ms") && !j["max_time_ms"].is_null()) {
        inst.max_time_ms = j["max_time_ms"].get<double>();
    }
    inst.output_dir = j.value("output_dir", std::string{});
    inst.export_closed_set = j.value("export_closed_set", false);
    inst.export_posthoc = j.value("export_posthoc", false);

    const grid::StaticClearanceGrid grid = inst.make_static_grid();

    if (j.contains("obstacles")) {
        for (const json& item : j.at("obstacles")) {
            const std::string mode = item.at("mode").get<std::string>();
            const float radius = item.value("cell_radius", 0.1f);

            if (mode == "sphere") {
                for (const grid::CellCoord& cell : read_cells(item.at("cells"))) {
                    if (!grid.in_bounds(cell)) {
                        fail("sphere cell out of bounds");
                    }
                    inst.spheres.emplace_back(grid.cell_center(cell), radius);
                }
            } else if (mode == "capsules") {
                inst.capsules.emplace_back(
                    read_vec3(item.at("from_point")),
                    read_vec3(item.at("to_point")),
                    radius);
            } else {
                fail("obstacles.mode must be \"sphere\" or \"capsules\"");
            }
        }
    }

    if (j.contains("queries")) {
        int auto_id = 0;
        for (const json& q : j.at("queries")) {
            SearchQuery query;
            query.query_id = q.value("query_id", auto_id++);
            query.starts = read_cells(q.at("start"));
            query.goals = read_cells(q.at("goal"));
            if (q.contains("opt_cost") && !q["opt_cost"].is_null()) {
                query.opt_cost = q["opt_cost"].get<double>();
            }
            if (q.contains("reference_cost") && !q["reference_cost"].is_null()) {
                query.reference_cost = q["reference_cost"].get<double>();
            }
            for (const grid::CellCoord& c : query.starts) {
                if (!grid.in_bounds(c)) {
                    fail("query start out of bounds");
                }
            }
            for (const grid::CellCoord& c : query.goals) {
                if (!grid.in_bounds(c)) {
                    fail("query goal out of bounds");
                }
            }
            inst.queries.push_back(std::move(query));
        }
    }

    return inst;
}

json dump(const SearchInstance& inst) {
    json j = {
        {"version", inst.version},
        {"instance_id", inst.instance_id},
        {"map_id", inst.map_id},
        {"size", json::array({inst.size_x, inst.size_y, inst.size_z})},
        {"resolution", inst.resolution},
        {"clearance_cap", inst.clearance_cap},
        {"pipe_radius", inst.pipe_radius},
        {"action_model", inst.action_model},
        {"action_model_kwargs", inst.action_model_kwargs},
        {"heuristic", inst.heuristic},
        {"heuristic_kwargs", inst.heuristic_kwargs},
        {"heuristic_id", inst.heuristic_id},
        {"heap", inst.heap},
        {"heap_kwargs", inst.heap_kwargs},
        {"heap_id", inst.heap_id},
        {"model", inst.model},
        {"model_kwargs", inst.model_kwargs},
        {"model_id", inst.model_id},
        {"max_time_ms", inst.max_time_ms},
        {"output_dir", inst.output_dir},
        {"export_closed_set", inst.export_closed_set},
        {"export_posthoc", inst.export_posthoc},
    };
    if (inst.max_expansions != std::numeric_limits<std::size_t>::max()) {
        j["max_expansions"] = inst.max_expansions;
    } else {
        j["max_expansions"] = nullptr;
    }

    json obs = json::array();
    std::map<float, json> sphere_cells_by_r;
    for (const geom::SphereObstacle& s : inst.spheres) {
        const grid::CellCoord cell{
            static_cast<int>(std::floor(s.center.x / inst.resolution)),
            static_cast<int>(std::floor(s.center.y / inst.resolution)),
            static_cast<int>(std::floor(s.center.z / inst.resolution)),
        };
        sphere_cells_by_r[s.radius].push_back(json::array({cell.x, cell.y, cell.z}));
    }
    for (auto& [radius, cells] : sphere_cells_by_r) {
        obs.push_back(
            {{"mode", "sphere"}, {"cell_radius", radius}, {"cells", std::move(cells)}});
    }
    for (const geom::CapsuleObstacle& c : inst.capsules) {
        obs.push_back({
            {"mode", "capsules"},
            {"cell_radius", c.radius},
            {"from_point", json::array({c.a.x, c.a.y, c.a.z})},
            {"to_point", json::array({c.b.x, c.b.y, c.b.z})},
        });
    }
    j["obstacles"] = std::move(obs);

    json queries = json::array();
    for (const SearchQuery& q : inst.queries) {
        json starts = json::array();
        json goals = json::array();
        for (const grid::CellCoord& c : q.starts) {
            starts.push_back(json::array({c.x, c.y, c.z}));
        }
        for (const grid::CellCoord& c : q.goals) {
            goals.push_back(json::array({c.x, c.y, c.z}));
        }
        json jq = {{"query_id", q.query_id}, {"start", starts}, {"goal", goals}};
        if (q.opt_cost) {
            jq["opt_cost"] = *q.opt_cost;
        }
        if (q.reference_cost) {
            jq["reference_cost"] = *q.reference_cost;
        }
        queries.push_back(std::move(jq));
    }
    j["queries"] = std::move(queries);
    return j;
}

}  // namespace

SearchInstance SearchInstance::from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        fail("cannot open file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return from_json_string(ss.str());
}

SearchInstance SearchInstance::from_json_string(const std::string& text) {
    try {
        return parse(json::parse(text));
    } catch (const json::exception& e) {
        fail(std::string("JSON parse error: ") + e.what());
    }
}

void SearchInstance::to_file(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        fail("cannot write file: " + path);
    }
    out << to_json_string(2);
}

std::string SearchInstance::to_json_string(int indent) const {
    return dump(*this).dump(indent);
}

grid::StaticClearanceGrid SearchInstance::make_static_grid() const {
    return grid::StaticClearanceGrid(size_x, size_y, size_z, resolution, clearance_cap);
}

}  // namespace pipe_design::tools
