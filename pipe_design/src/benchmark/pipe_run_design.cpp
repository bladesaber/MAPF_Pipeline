#include "benchmark/design_report/DesignReport.h"
#include "conflict/NearTerminalAnalyzer.h"
#include "grid/CompositeGrid.h"
#include "heuristic/Geometric.h"
#include "planner/high/CBS.h"
#include "planner/high/PP.h"
#include "planner/high/PBS.h"
#include "planner/high/sequence/GroupLayerSwapReplanner.h"
#include "planner/low/AStar.h"
#include "planner/low/SearchNode.h"
#include "planner/motion/SixConnected.h"
#include "planner/multi/GroupSession.h"
#include "planner/multi/PartialOrderSequencer.h"
#include "planner/multi/PrioritySequencer.h"
#include "planner/open/LazyBinaryHeap.h"
#include "problem/Problem.h"
#include "topology/PureMST.h"
#include "topology/Types.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

using pipe_design::benchmark::DesignReport;
using pipe_design::benchmark::write_design_reports_jsonl;
using pipe_design::conflict::NearTerminalAnalyzer;
using pipe_design::grid::CompositeGrid;
using pipe_design::heuristic::Manhattan;
using pipe_design::planner::AStar;
using pipe_design::planner::CBS;
using pipe_design::planner::GroupLayerSwapReplanner;
using pipe_design::planner::GroupSession;
using pipe_design::planner::HighLevelPlanner;
using pipe_design::planner::LazyBinaryHeap;
using pipe_design::planner::PartialOrderSequencer;
using pipe_design::planner::PP;
using pipe_design::planner::PBS;
using pipe_design::planner::PrioritySequencer;
using pipe_design::planner::SearchNode;
using pipe_design::planner::SearchStatus;
using pipe_design::planner::motion::SixConnected;
using pipe_design::problem::Problem;
using pipe_design::topology::CostType;
using pipe_design::topology::GlobalTopologyPlan;
using pipe_design::topology::PureMST;
using pipe_design::topology::RadiusMerge;

struct Cli {
    fs::path instance;
    fs::path instance_dir;
    fs::path export_dir = "runs";
    std::string algorithm = "cbs";
    std::string topology = "pure_mst";
    double max_time_ms = 0.0;
    std::size_t max_iterations = std::numeric_limits<std::size_t>::max();
};

static void usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " --instance <file.yaml> [options]\n"
        << "  " << prog << " --instance_dir <dir> --export_dir <dir> [options]\n"
        << "\nOptions:\n"
        << "  --algorithm cbs|pp|pbs     (default: cbs)\n"
        << "  --topology pure_mst        (default: pure_mst)\n"
        << "  --max_time_ms N            0 = unlimited (default: 0)\n"
        << "  --max_iterations N         0 = unlimited (default: unlimited)\n";
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
        } else if (arg == "--export_dir") {
            const char* v = need("--export_dir");
            if (v == nullptr) {
                return false;
            }
            cli.export_dir = v;
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
            cli.max_iterations = (n <= 0)
                ? std::numeric_limits<std::size_t>::max()
                : static_cast<std::size_t>(n);
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

static GlobalTopologyPlan build_topology(const Problem& problem, const std::string& name) {
    if (name != "pure_mst") {
        throw std::runtime_error("unsupported topology: " + name);
    }
    PureMST builder(RadiusMerge::Min, CostType::Dijkstra);
    builder.build(problem);
    return builder.plan();
}

static std::unordered_map<std::string, std::vector<pipe_design::geom::Vec3>>
terminal_map(const Problem& problem) {
    std::unordered_map<std::string, std::vector<pipe_design::geom::Vec3>> out;
    for (const auto& g : problem.groups()) {
        std::vector<pipe_design::geom::Vec3> pos;
        pos.reserve(g.terminals.size());
        for (const auto& t : g.terminals) {
            pos.push_back(t.pos);
        }
        out.emplace(g.id, std::move(pos));
    }
    return out;
}

static DesignReport run_one(
    const fs::path& yaml_path,
    const fs::path& instance_dir,
    const Cli& cli) {
    DesignReport report;
    report.instance_path = fs::absolute(yaml_path).string();
    report.instance_id = make_instance_id(yaml_path, instance_dir);
    report.algorithm = cli.algorithm;
    report.topology = cli.topology;
    report.low_model = "astar";
    report.heuristic = "manhattan";
    report.action_model = "SixConnected";
    report.max_time_ms = cli.max_time_ms;
    report.max_iterations = cli.max_iterations;

    const Problem problem = Problem::from_instance_file(yaml_path.string());
    const GlobalTopologyPlan plans = build_topology(problem, cli.topology);
    const float step = problem.grid().resolution;

    CompositeGrid grid(problem.static_grid());
    Manhattan heuristic(step);
    SixConnected actions(step);
    LazyBinaryHeap open;
    AStar<SearchNode> low;

    const auto t0 = std::chrono::steady_clock::now();

    if (cli.algorithm == "cbs") {
        report.conflict_analyzer = "near_terminal";
        GroupSession session;
        NearTerminalAnalyzer analyzer(terminal_map(problem));
        CBS cbs;
        cbs.reset(plans, grid, low, heuristic, open, actions, session, analyzer);
        while (cbs.status() == SearchStatus::Running &&
               cbs.expansions() < cli.max_iterations) {
            if (cli.max_time_ms > 0.0) {
                const double elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                if (elapsed_ms >= cli.max_time_ms) {
                    break;
                }
            }
            cbs.step();
        }
        report.iterations = cbs.expansions();
        report.set_groups_from_tasks(cbs.solution());
        if (cbs.status() == SearchStatus::Success) {
            report.success = true;
            report.status = "Success";
        } else if (cbs.status() == SearchStatus::Running) {
            report.status = "Running";
            report.fail_reason = (cli.max_time_ms > 0.0 &&
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count() >= cli.max_time_ms)
                ? "timeout"
                : "iteration_cap";
        } else {
            report.status = "Failure";
            report.fail_reason = "no_solution";
        }
    } else if (cli.algorithm == "pp") {
        report.conflict_analyzer = "";
        PrioritySequencer sequencer;
        GroupLayerSwapReplanner replanner;
        PP pp;
        pp.reset(plans, grid, low, heuristic, open, actions, sequencer, replanner);
        while (pp.status() == SearchStatus::Running &&
               pp.expansions() < cli.max_iterations) {
            if (cli.max_time_ms > 0.0) {
                const double elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                if (elapsed_ms >= cli.max_time_ms) {
                    break;
                }
            }
            pp.step();
        }
        report.iterations = pp.expansions();
        report.set_groups_from_tasks(pp.solution());
        if (pp.status() == SearchStatus::Success) {
            report.success = true;
            report.status = "Success";
        } else if (pp.status() == SearchStatus::Running) {
            report.status = "Running";
            report.fail_reason = (cli.max_time_ms > 0.0 &&
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count() >= cli.max_time_ms)
                ? "timeout"
                : "iteration_cap";
        } else {
            report.status = "Failure";
            report.fail_reason = "no_solution";
        }
    } else if (cli.algorithm == "pbs") {
        report.conflict_analyzer = "near_terminal";
        PartialOrderSequencer sequencer;
        NearTerminalAnalyzer analyzer(terminal_map(problem));
        PBS pbs;
        pbs.reset(plans, grid, low, heuristic, open, actions, sequencer, analyzer);
        while (pbs.status() == SearchStatus::Running &&
               pbs.expansions() < cli.max_iterations) {
            if (cli.max_time_ms > 0.0) {
                const double elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                if (elapsed_ms >= cli.max_time_ms) {
                    break;
                }
            }
            pbs.step();
        }
        report.iterations = pbs.expansions();
        report.set_groups_from_tasks(pbs.solution());
        if (pbs.status() == SearchStatus::Success) {
            report.success = true;
            report.status = "Success";
        } else if (pbs.status() == SearchStatus::Running) {
            report.status = "Running";
            report.fail_reason = (cli.max_time_ms > 0.0 &&
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count() >= cli.max_time_ms)
                ? "timeout"
                : "iteration_cap";
        } else {
            report.status = "Failure";
            report.fail_reason = "no_solution";
        }
    } else {
        throw std::runtime_error("unsupported algorithm: " + cli.algorithm);
    }

    report.total_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
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

    try {
        fs::create_directories(cli.export_dir);

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

        const fs::path jsonl = cli.export_dir / "design_reports.jsonl";
        write_design_reports_jsonl(jsonl.string(), reports);
        std::cout << "wrote " << reports.size() << " report(s) -> " << jsonl << '\n';
    } catch (const std::exception& e) {
        std::cerr << "pipe_run_design: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
