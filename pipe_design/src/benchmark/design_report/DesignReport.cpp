#include "benchmark/design_report/DesignReport.h"

#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace pipe_design::benchmark {
namespace {

using json = nlohmann::json;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("DesignReport: " + msg);
}

json cell_to_json(const grid::CellCoord& c) {
    return json::array({c.x, c.y, c.z});
}

json path_to_json(const planner::Path& path) {
    json cells = json::array();
    for (const grid::CellCoord& c : path.cells) {
        cells.push_back(cell_to_json(c));
    }
    return json{
        {"cost", path.cost},
        {"cells", std::move(cells)},
    };
}

json dump_task(const DesignTaskResult& t) {
    json j = path_to_json(t.path);
    j["task_id"] = t.task_id;
    j["edge_id"] = t.edge_id;
    j["success"] = t.success;
    return j;
}

json dump_group(const DesignGroupResult& g) {
    json tasks = json::array();
    for (const DesignTaskResult& t : g.tasks) {
        tasks.push_back(dump_task(t));
    }
    return json{
        {"group_id", g.group_id},
        {"success", g.success},
        {"tasks", std::move(tasks)},
    };
}

json dump(const DesignReport& r) {
    json j = {
        {"version", r.version},
        {"instance_id", r.instance_id},
        {"instance_path", r.instance_path},
        {"algorithm", r.algorithm},
        {"topology", r.topology},
        {"low_model", r.low_model},
        {"heuristic", r.heuristic},
        {"action_model", r.action_model},
        {"conflict_analyzer", r.conflict_analyzer},
        {"algorithm_kwargs", r.algorithm_kwargs},
        {"max_time_ms", r.max_time_ms},
        {"success", r.success},
        {"status", r.status},
        {"fail_reason", r.fail_reason},
        {"total_time_ms", r.total_time_ms},
        {"iterations", r.iterations},
        {"total_cost", r.total_cost},
        {"total_length", r.total_length},
        {"num_tasks", r.num_tasks},
        {"num_tasks_success", r.num_tasks_success},
    };

    if (r.max_iterations == std::numeric_limits<std::size_t>::max()) {
        j["max_iterations"] = nullptr;
    } else {
        j["max_iterations"] = r.max_iterations;
    }

    json groups = json::array();
    for (const DesignGroupResult& g : r.groups) {
        groups.push_back(dump_group(g));
    }
    j["groups"] = std::move(groups);
    return j;
}

}  // namespace

void DesignReport::set_groups_from_tasks(const std::vector<topology::Task>& tasks) {
    groups.clear();
    std::unordered_map<std::string, std::size_t> index;

    for (const topology::Task& task : tasks) {
        auto it = index.find(task.group_id);
        if (it == index.end()) {
            index.emplace(task.group_id, groups.size());
            DesignGroupResult g;
            g.group_id = task.group_id;
            groups.push_back(std::move(g));
            it = index.find(task.group_id);
        }

        DesignGroupResult& g = groups[it->second];
        DesignTaskResult tr;
        tr.task_id = task.task_id;
        tr.edge_id = task.edge_id;
        tr.success = task.succeeded();
        tr.path = task.path;
        g.tasks.push_back(std::move(tr));
    }

    for (DesignGroupResult& g : groups) {
        g.success = !g.tasks.empty();
        for (const DesignTaskResult& t : g.tasks) {
            if (!t.success) {
                g.success = false;
                break;
            }
        }
    }

    recompute_totals();
}

void DesignReport::recompute_totals() {
    total_cost = 0.0;
    total_length = 0;
    num_tasks = 0;
    num_tasks_success = 0;

    for (const DesignGroupResult& g : groups) {
        for (const DesignTaskResult& t : g.tasks) {
            ++num_tasks;
            if (t.success) {
                ++num_tasks_success;
            }
            total_cost += t.path.cost;
            total_length += t.path.length();
        }
    }
}

void DesignReport::to_file(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        fail("cannot write file: " + path);
    }
    out << to_json_string(2);
}

std::string DesignReport::to_json_string(int indent) const {
    return dump(*this).dump(indent);
}

void write_design_reports_jsonl(
    const std::string& path,
    const std::vector<DesignReport>& reports) {
    std::ofstream out(path);
    if (!out) {
        fail("cannot write file: " + path);
    }
    for (const DesignReport& r : reports) {
        out << r.to_json_string(-1) << '\n';
    }
}

}  // namespace pipe_design::benchmark
