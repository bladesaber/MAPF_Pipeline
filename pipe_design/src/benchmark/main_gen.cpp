#include "benchmark/YamlUtils.h"
#include "benchmark/axis_ray/AxisRayGenerator.h"
#include "benchmark/plane_slice/PlaneSlicedGenerator.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace {

void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --generator <axis_ray|plane_slice> --out <dir> [options]\n"
        << "\nRequired:\n"
        << "  --generator <name>     axis_ray | plane_slice\n"
        << "  --out <dir>            output folder (created if missing)\n"
        << "\nCommon:\n"
        << "  --num <int>            instances to generate (default 1)\n"
        << "  --seed <uint>          base seed; instance i uses seed+i (default 0)\n"
        << "  --size <int>           cubic domain size_x=y=z (default 32)\n"
        << "  --size_x/y/z <int>     override one axis\n"
        << "  --radius <float>       pipe radius (default 0.5)\n"
        << "  --shell_margin <float> extra shell beyond radius (default 0;\n"
        << "                         shell_cells=ceil((shell_margin+radius)/res))\n"
        << "  --terminal_gap <int>   slack beyond diameter (default 2)\n"
        << "  --num_groups <int>     (default 1)\n"
        << "  --terminals <list>     per-group counts, e.g. 3 or 2,2,3 (default 2 each)\n"
        << "  --max_iterations <int> (default 256)\n"
        << "  --subdomain_n <int>    plane_slice only: 3|4|5 (default 3)\n"
        << "  --keep_failed          still write YAML when generate() fails\n"
        << "\naxis_ray:\n"
        << "  --init_waypoints_num --attach_waypoints_num\n"
        << "  --min_path_size --max_path_size --path_gap --branch_gap --max_attempts\n"
        << "\nplane_slice:\n"
        << "  --plane_gap <int>\n";
}

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        out.push_back(std::stoi(item));
    }
    return out;
}

const char* require_arg(int argc, char** argv, int& i, const char* flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
    }
    return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
    std::string generator;
    std::string out_dir;
    int num = 1;
    std::uint32_t seed0 = 0;
    int size = 32;
    int size_x = -1, size_y = -1, size_z = -1;
    float radius = 0.5f;
    float shell_margin = 0.f;
    int terminal_gap = 2;
    int num_groups = 1;
    std::vector<int> terminals;
    int max_iterations = 256;
    int subdomain_n = 3;
    bool keep_failed = false;

    // axis_ray
    int init_waypoints_num = 3;
    int attach_waypoints_num = 2;
    int min_path_size = 4;
    int max_path_size = 16;
    int path_gap = 2;
    int branch_gap = 3;
    int max_attempts = 64;

    // plane_slice
    int plane_gap = 2;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "-h" || a == "--help") {
                usage(argv[0]);
                return 0;
            }
            if (a == "--generator") {
                generator = require_arg(argc, argv, i, a.c_str());
            } else if (a == "--out") {
                out_dir = require_arg(argc, argv, i, a.c_str());
            } else if (a == "--num") {
                num = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--seed") {
                seed0 = static_cast<std::uint32_t>(
                    std::stoul(require_arg(argc, argv, i, a.c_str())));
            } else if (a == "--size") {
                size = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--size_x") {
                size_x = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--size_y") {
                size_y = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--size_z") {
                size_z = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--radius") {
                radius = std::stof(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--shell_margin") {
                shell_margin = std::stof(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--terminal_gap") {
                terminal_gap = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--num_groups") {
                num_groups = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--terminals") {
                terminals = parse_int_list(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--max_iterations") {
                max_iterations = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--subdomain_n") {
                subdomain_n = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--keep_failed") {
                keep_failed = true;
            } else if (a == "--init_waypoints_num") {
                init_waypoints_num = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--attach_waypoints_num") {
                attach_waypoints_num = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--min_path_size") {
                min_path_size = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--max_path_size") {
                max_path_size = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--path_gap") {
                path_gap = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--branch_gap") {
                branch_gap = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--max_attempts") {
                max_attempts = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else if (a == "--plane_gap") {
                plane_gap = std::stoi(require_arg(argc, argv, i, a.c_str()));
            } else {
                throw std::runtime_error("unknown argument: " + a);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        usage(argv[0]);
        return 1;
    }

    if (generator.empty() || out_dir.empty()) {
        std::cerr << "error: --generator and --out are required\n";
        usage(argv[0]);
        return 1;
    }
    if (num <= 0) {
        std::cerr << "error: --num must be > 0\n";
        return 1;
    }

    if (size_x < 0) {
        size_x = size;
    }
    if (size_y < 0) {
        size_y = size;
    }
    if (size_z < 0) {
        size_z = size;
    }

    if (terminals.empty()) {
        terminals.assign(static_cast<std::size_t>(num_groups), 2);
    } else if (static_cast<int>(terminals.size()) == 1 && num_groups > 1) {
        terminals.assign(static_cast<std::size_t>(num_groups), terminals[0]);
    } else if (static_cast<int>(terminals.size()) != num_groups) {
        std::cerr << "error: --terminals length must be 1 or equal --num_groups\n";
        return 1;
    }

    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "error: cannot create --out: " << ec.message() << "\n";
        return 1;
    }

    int ok_count = 0;
    int fail_count = 0;

    for (int i = 0; i < num; ++i) {
        const std::uint32_t seed = seed0 + static_cast<std::uint32_t>(i);
        pipe_design::benchmark::Instance inst;
        YAML::Node gen_params;
        bool ok = false;

        if (generator == "axis_ray") {
            pipe_design::benchmark::axis_ray::GenParams p;
            p.size_x = size_x;
            p.size_y = size_y;
            p.size_z = size_z;
            p.radius = radius;
            p.shell_margin = shell_margin;
            p.terminal_gap = terminal_gap;
            p.num_groups = num_groups;
            p.terminals_per_group = terminals;
            p.seed = seed;
            p.max_iterations = max_iterations;
            p.subdomain_n = subdomain_n;
            p.init_waypoints_num = init_waypoints_num;
            p.attach_waypoints_num = attach_waypoints_num;
            p.min_path_size = min_path_size;
            p.max_path_size = max_path_size;
            p.path_gap = path_gap;
            p.branch_gap = branch_gap;
            p.max_attempts = max_attempts;

            gen_params["init_waypoints_num"] = p.init_waypoints_num;
            gen_params["attach_waypoints_num"] = p.attach_waypoints_num;
            gen_params["min_path_size"] = p.min_path_size;
            gen_params["max_path_size"] = p.max_path_size;
            gen_params["path_gap"] = p.path_gap;
            gen_params["branch_gap"] = p.branch_gap;
            gen_params["max_attempts"] = p.max_attempts;

            pipe_design::benchmark::axis_ray::AxisRayGenerator gen(p);
            ok = gen.generate(inst);
        } else if (generator == "plane_slice") {
            pipe_design::benchmark::plane_slice::GenParams p;
            p.size_x = size_x;
            p.size_y = size_y;
            p.size_z = size_z;
            p.radius = radius;
            p.shell_margin = shell_margin;
            p.terminal_gap = terminal_gap;
            p.num_groups = num_groups;
            p.terminals_per_group = terminals;
            p.seed = seed;
            p.max_iterations = max_iterations;
            p.subdomain_n = subdomain_n;
            p.plane_gap = plane_gap;

            gen_params["plane_gap"] = p.plane_gap;

            pipe_design::benchmark::plane_slice::PlaneSlicedGenerator gen(p);
            ok = gen.generate(inst);
        } else {
            std::cerr << "error: unknown generator '" << generator << "'\n";
            return 1;
        }

        if (!ok || !inst.success) {
            ++fail_count;
            std::cerr << "[" << i << "] FAIL seed=" << seed << " "
                      << (inst.reason.empty() ? "(no reason)" : inst.reason) << "\n";
            if (!keep_failed) {
                continue;
            }
        } else {
            ++ok_count;
        }

        const std::string path =
            (fs::path(out_dir) / ("instance_" + std::to_string(i) + ".yaml")).string();
        if (!pipe_design::benchmark::write_instance_yaml(path, inst, generator, gen_params)) {
            std::cerr << "[" << i << "] error writing " << path << "\n";
            return 1;
        }
        std::cout << "[" << i << "] wrote " << path
                  << (ok && inst.success ? " (ok)" : " (failed instance)") << "\n";
    }

    std::cout << "done: ok=" << ok_count << " fail=" << fail_count << "\n";
    return fail_count > 0 ? 2 : 0;
}
