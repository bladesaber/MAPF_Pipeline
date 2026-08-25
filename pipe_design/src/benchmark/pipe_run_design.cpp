#include "benchmark/design_report/DesignReport.h"
#include "conflict/ConflictAnalyzer.h"
#include "conflict/NearTerminalAnalyzer.h"
#include "grid/CompositeGrid.h"
#include "heuristic/Euclidean.h"
#include "heuristic/Heuristic.h"
#include "heuristic/Manhattan.h"
#include "heuristic/Octile.h"
#include "planner/high/CBS.h"
#include "planner/high/PP.h"
#include "planner/high/PBS.h"
#include "planner/high/sequence/GroupLayerSwapReplanner.h"
#include "planner/low/AStar.h"
#include "planner/low/SearchNode.h"
#include "planner/motion/ActionModel.h"
#include "planner/motion/EightConnected.h"
#include "planner/motion/SixConnected.h"
#include "planner/motion/TwentySixConnected.h"
#include "planner/multi/GroupSession.h"
#include "planner/multi/PartialOrderSequencer.h"
#include "planner/multi/PrioritySequencer.h"
#include "planner/open/PairingHeap.h"
#include "problem/Problem.h"
#include "topology/PureMST.h"
#include "topology/RadiusPriorityMergeBuilder.h"
#include "topology/RadiusSpineMSTBuilder.h"
#include "topology/Types.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

using pipe_design::benchmark::DesignReport;
using pipe_design::benchmark::write_design_reports_jsonl;
using pipe_design::conflict::ConflictAnalyzer;
using pipe_design::conflict::NearTerminalAnalyzer;
using pipe_design::grid::CompositeGrid;
using pipe_design::heuristic::Euclidean;
using pipe_design::heuristic::Heuristic;
using pipe_design::heuristic::Manhattan;
using pipe_design::heuristic::Octile;
using pipe_design::planner::AStar;
using pipe_design::planner::CBS;
using pipe_design::planner::GroupLayerSwapReplanner;
using pipe_design::planner::GroupSession;
using pipe_design::planner::PairingHeap;
using pipe_design::planner::PartialOrderSequencer;
using pipe_design::planner::PP;
using pipe_design::planner::PBS;
using pipe_design::planner::PBSSearchMode;
using pipe_design::planner::PrioritySequencer;
using pipe_design::planner::SearchNode;
using pipe_design::planner::SearchStatus;
using pipe_design::planner::motion::ActionModel;
using pipe_design::planner::motion::EightConnected;
using pipe_design::planner::motion::SixConnected;
using pipe_design::planner::motion::TwentySixConnected;
using pipe_design::problem::Problem;
using pipe_design::topology::CostType;
using pipe_design::topology::GlobalTopologyPlan;
using pipe_design::topology::PureMST;
using pipe_design::topology::RadiusMerge;
using pipe_design::topology::RadiusPriorityMergeBuilder;
using pipe_design::topology::RadiusSpineMSTBuilder;

struct Cli {
    fs::path instance;
    fs::path instance_dir;
    fs::path export_path = "design_reports.jsonl";
    std::string algorithm = "cbs";
    // Topology section
    std::string topology = "pure_mst";
    std::string radius_merge = "min";
    std::string cost_type = "euclidean";
    // Low-level section
    std::string heuristic = "octile";
    std::string action_model = "TwentySixConnected";
    // CBS / PBS
    std::string conflict_analyzer = "near_terminal";
    // PBS
    std::string pbs_mode = "depth_first";
    // Budget section: exactly one active axis via budget_mode
    std::string budget_mode = "time";
    double max_time_ms = 60000.0;
    std::size_t max_iterations = 10000;
    // Free-form tags copied into DesignReport.meta (JSON object).
    nlohmann::json meta = nlohmann::json::object();
};

static void usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " --instance <file.yaml> [options]\n"
        << "  " << prog << " --instance_dir <dir> [options]\n"
        << "\nI/O:\n"
        << "  --instance PATH\n"
        << "  --instance_dir DIR\n"
        << "  --export PATH                          output JSONL file (default: design_reports.jsonl)\n"
        << "  --meta JSON_OBJECT                     free-form tags → report.meta\n"
        << "                                         e.g. '{\"generator\":\"axis_ray\",\"level\":\"middle\"}'\n"
        << "\nHigh-level:\n"
        << "  --algorithm cbs|pp|pbs                 (default: cbs)\n"
        << "\nTopology:\n"
        << "  --topology pure_mst|radius_spine|radius_priority_merge\n"
        << "                                         (default: pure_mst)\n"
        << "  --radius_merge min|average             (default: min)\n"
        << "  --cost_type euclidean|dijkstra|volume  (default: euclidean)\n"
        << "\nLow-level:\n"
        << "  --heuristic manhattan|euclidean|octile (default: octile)\n"
        << "  --action_model SixConnected|EightConnected|TwentySixConnected\n"
        << "                                         (default: TwentySixConnected)\n"
        << "\nCBS:\n"
        << "  --conflict_analyzer near_terminal      (default: near_terminal)\n"
        << "\nPBS:\n"
        << "  --conflict_analyzer near_terminal      (same flag as CBS)\n"
        << "  --pbs_mode depth_first|best_first      (default: depth_first)\n"
        << "\nPP:\n"
        << "  (no algo-specific flags yet; uses GroupLayerSwapReplanner)\n"
        << "\nBudget (exactly one mode):\n"
        << "  --budget_mode time|iterations          (default: time)\n"
        << "  --max_time_ms N                        used if mode=time (default: 60000)\n"
        << "  --max_iterations N                     used if mode=iterations (default: 10000)\n";
}

static bool parse_cli(int argc, char** argv, Cli& cli) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << '\n';
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--instance") {
            const char* v = need("--instance");
            if (v == nullptr) {
                return false;
            }
            cli.instance = v;
        } else if (arg == "--instance_dir") {
            const char* v = need("--instance_dir");
            if (v == nullptr) {
                return false;
            }
            cli.instance_dir = v;
        } else if (arg == "--export") {
            const char* v = need("--export");
            if (v == nullptr) {
                return false;
            }
            cli.export_path = v;
        } else if (arg == "--export_dir") {
            std::cerr << "--export_dir is removed; use --export <file.jsonl>\n";
            return false;
        } else if (arg == "--meta") {
            const char* v = need("--meta");
            if (v == nullptr) {
                return false;
            }
            try {
                cli.meta = nlohmann::json::parse(v);
            } catch (const nlohmann::json::exception& e) {
                std::cerr << "--meta must be valid JSON: " << e.what() << '\n';
                return false;
            }
            if (!cli.meta.is_object()) {
                std::cerr << "--meta must be a JSON object (dict), got "
                          << cli.meta.type_name() << '\n';
                return false;
            }
        } else if (arg == "--algorithm") {
            const char* v = need("--algorithm");
            if (v == nullptr) {
                return false;
            }
            cli.algorithm = v;
        } else if (arg == "--topology") {
            const char* v = need("--topology");
            if (v == nullptr) {
                return false;
            }
            cli.topology = v;
        } else if (arg == "--radius_merge") {
            const char* v = need("--radius_merge");
            if (v == nullptr) {
                return false;
            }
            cli.radius_merge = v;
        } else if (arg == "--cost_type") {
            const char* v = need("--cost_type");
            if (v == nullptr) {
                return false;
            }
            cli.cost_type = v;
        } else if (arg == "--heuristic") {
            const char* v = need("--heuristic");
            if (v == nullptr) {
                return false;
            }
            cli.heuristic = v;
        } else if (arg == "--action_model") {
            const char* v = need("--action_model");
            if (v == nullptr) {
                return false;
            }
            cli.action_model = v;
        } else if (arg == "--conflict_analyzer") {
            const char* v = need("--conflict_analyzer");
            if (v == nullptr) {
                return false;
            }
            cli.conflict_analyzer = v;
        } else if (arg == "--pbs_mode") {
            const char* v = need("--pbs_mode");
            if (v == nullptr) {
                return false;
            }
            cli.pbs_mode = v;
        } else if (arg == "--budget_mode") {
            const char* v = need("--budget_mode");
            if (v == nullptr) {
                return false;
            }
            cli.budget_mode = v;
        } else if (arg == "--max_time_ms") {
            const char* v = need("--max_time_ms");
            if (v == nullptr) {
                return false;
            }
            cli.max_time_ms = std::atof(v);
        } else if (arg == "--max_iterations") {
            const char* v = need("--max_iterations");
            if (v == nullptr) {
                return false;
            }
            const long long n = std::atoll(v);
            if (n <= 0) {
                std::cerr << "--max_iterations must be > 0\n";
                return false;
            }
            cli.max_iterations = static_cast<std::size_t>(n);
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "unknown arg: " << arg << '\n';
            return false;
        }
    }

    const bool has_file = !cli.instance.empty();
    const bool has_dir = !cli.instance_dir.empty();
    if (has_file == has_dir) {
        std::cerr << "exactly one of --instance or --instance_dir is required\n";
        return false;
    }
    return true;
}

static std::vector<fs::path> collect_instances(const fs::path& dir) {
    std::vector<fs::path> out;
    for (const fs::directory_entry& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) {
            continue;
        }
        const fs::path p = e.path();
        const std::string name = p.filename().string();
        if (name.rfind("instance_", 0) == 0 && p.extension() == ".yaml") {
            out.push_back(p);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static std::string make_instance_id(
    const fs::path& yaml_path,
    const fs::path& instance_dir) {
    if (!instance_dir.empty()) {
        fs::path base = instance_dir.parent_path().parent_path();
        if (base.empty() || base == instance_dir.parent_path()) {
            base = instance_dir.parent_path();
        }
        std::error_code ec;
        const fs::path rel = fs::relative(yaml_path, base, ec);
        if (!ec) {
            fs::path id = rel;
            id.replace_extension("");
            return id.generic_string();
        }
    }
    return yaml_path.stem().string();
}

static GlobalTopologyPlan build_topology(
    const Problem& problem,
    const std::string& name,
    const std::string& radius_merge_name,
    const std::string& cost_type_name) {
    RadiusMerge radius_merge = RadiusMerge::Min;
    if (radius_merge_name == "min") {
        radius_merge = RadiusMerge::Min;
    } else if (radius_merge_name == "average") {
        radius_merge = RadiusMerge::Average;
    } else {
        throw std::runtime_error("unsupported radius_merge: " + radius_merge_name);
    }

    CostType cost_type = CostType::Dijkstra;
    if (cost_type_name == "euclidean") {
        cost_type = CostType::Euclidean;
    } else if (cost_type_name == "dijkstra") {
        cost_type = CostType::Dijkstra;
    } else if (cost_type_name == "volume") {
        cost_type = CostType::Volume;
    } else {
        throw std::runtime_error("unsupported cost_type: " + cost_type_name);
    }

    if (name == "pure_mst") {
        PureMST builder(radius_merge, cost_type);
        builder.build(problem);
        return builder.plan();
    }
    if (name == "radius_spine") {
        RadiusSpineMSTBuilder builder(radius_merge, cost_type);
        builder.build(problem);
        return builder.plan();
    }
    if (name == "radius_priority_merge") {
        RadiusPriorityMergeBuilder builder(radius_merge, cost_type);
        builder.build(problem);
        return builder.plan();
    }
    throw std::runtime_error("unsupported topology: " + name);
}

static std::unique_ptr<Heuristic> build_heuristic(const std::string& name, float step) {
    if (name == "manhattan") {
        return std::make_unique<Manhattan>(step);
    }
    if (name == "euclidean") {
        return std::make_unique<Euclidean>(step);
    }
    if (name == "octile") {
        return std::make_unique<Octile>(step);
    }
    throw std::runtime_error("unsupported heuristic: " + name);
}

static std::unique_ptr<ActionModel> build_action_model(const std::string& name, float step) {
    if (name == "SixConnected") {
        return std::make_unique<SixConnected>(step);
    }
    if (name == "EightConnected") {
        return std::make_unique<EightConnected>(step);
    }
    if (name == "TwentySixConnected") {
        return std::make_unique<TwentySixConnected>(step);
    }
    throw std::runtime_error("unsupported action_model: " + name);
}

static std::unique_ptr<ConflictAnalyzer> build_conflict_analyzer(
    const std::string& name,
    const Problem& problem) {
    if (name == "near_terminal") {
        std::unordered_map<std::string, std::vector<pipe_design::geom::Vec3>> terminals;
        for (const auto& g : problem.groups()) {
            std::vector<pipe_design::geom::Vec3> pos;
            pos.reserve(g.terminals.size());
            for (const auto& t : g.terminals) {
                pos.push_back(t.pos);
            }
            terminals.emplace(g.id, std::move(pos));
        }
        return std::make_unique<NearTerminalAnalyzer>(std::move(terminals));
    }
    throw std::runtime_error("unsupported conflict_analyzer: " + name);
}

static DesignReport run_one(
    const fs::path& yaml_path,
    const fs::path& instance_dir,
    const Cli& cli) {
    DesignReport report;
    report.instance_path = fs::absolute(yaml_path).string();
    report.instance_id = make_instance_id(yaml_path, instance_dir);
    report.meta = cli.meta;
    report.algorithm = cli.algorithm;
    report.topology = cli.topology;
    report.low_model = "astar";  // TODO: add low-level model name in the future
    report.heuristic = cli.heuristic;
    report.action_model = cli.action_model;

    report.conflict_analyzer = (cli.algorithm == "pp") ? "" : cli.conflict_analyzer;
    report.algorithm_kwargs = {
        {"radius_merge", cli.radius_merge},
        {"cost_type", cli.cost_type},
    };
    if (cli.algorithm == "pbs") {
        report.algorithm_kwargs["pbs_mode"] = cli.pbs_mode;
    }

    report.budget_mode = cli.budget_mode;
    report.max_time_ms = cli.max_time_ms;
    report.max_iterations = cli.max_iterations;

    const Problem problem = Problem::from_instance_file(yaml_path.string());
    const GlobalTopologyPlan plans =
        build_topology(problem, cli.topology, cli.radius_merge, cli.cost_type);
    const float step = problem.grid().resolution;

    CompositeGrid grid(problem.static_grid());
    std::unique_ptr<Heuristic> heuristic = build_heuristic(cli.heuristic, step);
    std::unique_ptr<ActionModel> actions = build_action_model(cli.action_model, step);
    PairingHeap open;
    AStar<SearchNode> low;

    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t iter_cap = (cli.budget_mode == "iterations")
        ? cli.max_iterations
        : std::numeric_limits<std::size_t>::max();
    auto time_exceeded = [&]() -> bool {
        if (cli.budget_mode != "time" || cli.max_time_ms <= 0.0) {
            return false;
        }
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
                   .count() >= cli.max_time_ms;
    };

    if (cli.algorithm == "cbs") {
        GroupSession session;
        std::unique_ptr<ConflictAnalyzer> analyzer =
            build_conflict_analyzer(cli.conflict_analyzer, problem);
        CBS cbs;
        cbs.reset(plans, grid, low, *heuristic, open, *actions, session, *analyzer);
        while (cbs.status() == SearchStatus::Running &&
               cbs.expansions() < iter_cap &&
               !time_exceeded()) {
            cbs.step();
        }
        report.iterations = cbs.expansions();
        report.set_groups_from_tasks(cbs.solution());
        if (cbs.status() == SearchStatus::Success) {
            report.success = true;
            report.status = "Success";
        } else if (cbs.status() == SearchStatus::Running) {
            report.status = "Running";
            report.fail_reason =
                (cli.budget_mode == "time") ? "timeout" : "iteration_cap";
        } else {
            report.status = "Failure";
            report.fail_reason = "no_solution";
        }

    } else if (cli.algorithm == "pp") {
        PrioritySequencer sequencer;
        GroupLayerSwapReplanner replanner;
        PP pp;
        pp.reset(plans, grid, low, *heuristic, open, *actions, sequencer, replanner);
        while (pp.status() == SearchStatus::Running &&
               pp.expansions() < iter_cap &&
               !time_exceeded()) {
            pp.step();
        }
        report.iterations = pp.expansions();
        report.set_groups_from_tasks(pp.solution());
        if (pp.status() == SearchStatus::Success) {
            report.success = true;
            report.status = "Success";
        } else if (pp.status() == SearchStatus::Running) {
            report.status = "Running";
            report.fail_reason =
                (cli.budget_mode == "time") ? "timeout" : "iteration_cap";
        } else {
            report.status = "Failure";
            report.fail_reason = "no_solution";
        }

    } else if (cli.algorithm == "pbs") {
        PartialOrderSequencer sequencer;
        std::unique_ptr<ConflictAnalyzer> analyzer =
            build_conflict_analyzer(cli.conflict_analyzer, problem);
        PBS pbs;
        if (cli.pbs_mode == "best_first") {
            pbs.set_search_mode(PBSSearchMode::BestFirst);
        } else if (cli.pbs_mode == "depth_first") {
            pbs.set_search_mode(PBSSearchMode::DepthFirst);
        } else {
            throw std::runtime_error("unsupported pbs_mode: " + cli.pbs_mode);
        }
        pbs.reset(plans, grid, low, *heuristic, open, *actions, sequencer, *analyzer);
        while (pbs.status() == SearchStatus::Running &&
               pbs.expansions() < iter_cap &&
               !time_exceeded()) {
            pbs.step();
        }
        report.iterations = pbs.expansions();
        report.set_groups_from_tasks(pbs.solution());
        if (pbs.status() == SearchStatus::Success) {
            report.success = true;
            report.status = "Success";
        } else if (pbs.status() == SearchStatus::Running) {
            report.status = "Running";
            report.fail_reason =
                (cli.budget_mode == "time") ? "timeout" : "iteration_cap";
        } else {
            report.status = "Failure";
            report.fail_reason = "no_solution";
        }
        
    } else {
        throw std::runtime_error("unsupported algorithm: " + cli.algorithm);
    }

    report.total_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    report.compare_witness();
    return report;
}

int main(int argc, char** argv) {
    Cli cli;
    if (!parse_cli(argc, argv, cli)) {
        usage(argv[0]);
        return 1;
    }

    if (cli.algorithm != "cbs" && cli.algorithm != "pp" && cli.algorithm != "pbs") {
        std::cerr << "unsupported algorithm: " << cli.algorithm << '\n';
        return 1;
    }
    if (cli.topology != "pure_mst" && cli.topology != "radius_spine" &&
        cli.topology != "radius_priority_merge") {
        std::cerr << "unsupported topology: " << cli.topology << '\n';
        return 1;
    }
    if (cli.radius_merge != "min" && cli.radius_merge != "average") {
        std::cerr << "unsupported radius_merge: " << cli.radius_merge << '\n';
        return 1;
    }
    if (cli.cost_type != "euclidean" && cli.cost_type != "dijkstra" &&
        cli.cost_type != "volume") {
        std::cerr << "unsupported cost_type: " << cli.cost_type << '\n';
        return 1;
    }
    if (cli.heuristic != "manhattan" && cli.heuristic != "euclidean" &&
        cli.heuristic != "octile") {
        std::cerr << "unsupported heuristic: " << cli.heuristic << '\n';
        return 1;
    }
    if (cli.action_model != "SixConnected" && cli.action_model != "EightConnected" &&
        cli.action_model != "TwentySixConnected") {
        std::cerr << "unsupported action_model: " << cli.action_model << '\n';
        return 1;
    }
    if (cli.conflict_analyzer != "near_terminal") {
        std::cerr << "unsupported conflict_analyzer: " << cli.conflict_analyzer << '\n';
        return 1;
    }
    if (cli.pbs_mode != "depth_first" && cli.pbs_mode != "best_first") {
        std::cerr << "unsupported pbs_mode: " << cli.pbs_mode << '\n';
        return 1;
    }
    if (cli.budget_mode != "time" && cli.budget_mode != "iterations") {
        std::cerr << "unsupported budget_mode: " << cli.budget_mode << '\n';
        return 1;
    }
    if (cli.budget_mode == "time" && cli.max_time_ms <= 0.0) {
        std::cerr << "budget_mode=time requires --max_time_ms > 0\n";
        return 1;
    }

    try {
        if (cli.export_path.has_parent_path() && !cli.export_path.parent_path().empty()) {
            fs::create_directories(cli.export_path.parent_path());
        }

        std::vector<fs::path> instances;
        if (!cli.instance.empty()) {
            instances.push_back(cli.instance);
        } else {
            instances = collect_instances(cli.instance_dir);
            if (instances.empty()) {
                std::cerr << "no instance_*.yaml in " << cli.instance_dir << '\n';
                return 1;
            }
        }

        std::vector<DesignReport> reports;
        reports.reserve(instances.size());
        for (const fs::path& p : instances) {
            std::cout << "run " << p.filename().string() << "  algo=" << cli.algorithm << '\n';
            reports.push_back(run_one(p, cli.instance_dir, cli));
        }

        write_design_reports_jsonl(cli.export_path.string(), reports);
        std::cout << "wrote " << reports.size() << " report(s) -> " << cli.export_path << '\n';
    } catch (const std::exception& e) {
        std::cerr << "pipe_run_design: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
