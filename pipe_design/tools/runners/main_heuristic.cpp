#include "runners/LowLevelRunner.h"

#include "formats/SearchInstance.h"
#include "grid/CompositeGrid.h"
#include "heuristic/Dijkstra.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using pipe_design::tools::SearchInstance;
using pipe_design::tools::make_heuristic;

double h_to_goals(
    const pipe_design::heuristic::Heuristic& h,
    const pipe_design::grid::CellCoord& from,
    const std::vector<pipe_design::grid::CellCoord>& goals) {
    // Dijkstra::estimate ignores `to` once prepared (multi-source field).
    if (dynamic_cast<const pipe_design::heuristic::Dijkstra*>(&h) != nullptr) {
        return h.estimate(from, from);
    }
    double best = std::numeric_limits<double>::infinity();
    for (const auto& g : goals) {
        best = std::min(best, h.estimate(from, g));
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    std::string instance_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--instance" || arg == "-i") && i + 1 < argc) {
            instance_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: pipe_dump_heuristic --instance <file.json>\n"
                << "  Independent of search. Writes heuristic_q*.json under\n"
                << "  output_dir/heuristic/ (h = Heuristic::estimate to each\n"
                << "  query's goals, free cells only).\n";
            return 0;
        } else {
            std::cerr << "Unknown arg: " << arg << '\n';
            return 1;
        }
    }

    if (instance_path.empty()) {
        std::cerr << "pipe_dump_heuristic: --instance is required\n";
        return 1;
    }

    try {
        const auto instance = SearchInstance::from_file(instance_path);
        if (instance.output_dir.empty()) {
            throw std::runtime_error("pipe_dump_heuristic: output_dir is empty in SearchInstance");
        }
        if (instance.queries.empty()) {
            throw std::runtime_error("pipe_dump_heuristic: no queries (need goals)");
        }

        const fs::path out_dir = fs::path(instance.output_dir) / "heuristic";
        fs::create_directories(out_dir);

        auto static_grid = instance.make_static_grid();
        static_grid.bake(instance.bake_spheres(), instance.bake_capsules());
        pipe_design::grid::CompositeGrid grid(static_grid);

        for (const auto& q : instance.queries) {
            auto heuristic = make_heuristic(instance, static_grid, q);

            json j;
            j["instance_id"] = instance.instance_id;
            j["map_id"] = instance.map_id;
            j["query_id"] = q.query_id;
            j["heuristic"] = instance.heuristic;
            j["size"] = json::array({instance.size_x, instance.size_y, instance.size_z});
            json goals = json::array();
            for (const auto& g : q.goals) {
                goals.push_back(json::array({g.x, g.y, g.z}));
            }
            j["goals"] = std::move(goals);

            json cells = json::array();
            for (int z = 0; z < instance.size_z; ++z) {
                for (int y = 0; y < instance.size_y; ++y) {
                    for (int x = 0; x < instance.size_x; ++x) {
                        const pipe_design::grid::CellCoord c{x, y, z};
                        if (!grid.is_free(c, instance.pipe_radius)) {
                            continue;
                        }
                        cells.push_back({
                            {"x", x},
                            {"y", y},
                            {"z", z},
                            {"h", h_to_goals(*heuristic, c, q.goals)},
                        });
                    }
                }
            }
            j["cells"] = std::move(cells);

            const fs::path path =
                out_dir / ("heuristic_q" + std::to_string(q.query_id) + ".json");
            std::ofstream out(path);
            if (!out) {
                throw std::runtime_error("cannot write " + path.string());
            }
            out << j.dump(2);
            std::cout << "wrote " << path.string() << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "pipe_dump_heuristic: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
