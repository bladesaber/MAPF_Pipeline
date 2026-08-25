#include "problem/Problem.h"

#include <algorithm>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace pipe_design::problem {
namespace {

using pipe_design::geom::AabbObstacle;
using pipe_design::geom::CapsuleObstacle;
using pipe_design::geom::SphereObstacle;
using pipe_design::geom::Vec3;

Vec3 read_vec3(const YAML::Node& n) {
    if (!n || !n.IsSequence() || n.size() < 3) {
        throw std::runtime_error("Problem::from_instance_file: expected [x,y,z]");
    }
    return Vec3{n[0].as<float>(), n[1].as<float>(), n[2].as<float>()};
}

}  // namespace

Problem Problem::from_instance_file(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);
    if (!root["grid"]) {
        throw std::runtime_error("Problem::from_instance_file: missing grid in " + path);
    }
    if (!root["groups"]) {
        throw std::runtime_error("Problem::from_instance_file: missing groups in " + path);
    }

    const YAML::Node& grid = root["grid"];
    if (!grid["resolution"]) {
        throw std::runtime_error("Problem::from_instance_file: missing grid.resolution in " + path);
    }

    std::vector<SphereObstacle> spheres;
    std::vector<CapsuleObstacle> capsules;
    std::vector<AabbObstacle> aabbs;
    if (root["obstacles"]) {
        const YAML::Node& obs = root["obstacles"];
        if (obs["spheres"]) {
            for (const YAML::Node& sn : obs["spheres"]) {
                spheres.emplace_back(read_vec3(sn["center"]), sn["radius"].as<float>());
            }
        }
        if (obs["capsules"]) {
            for (const YAML::Node& cn : obs["capsules"]) {
                capsules.emplace_back(
                    read_vec3(cn["a"]), read_vec3(cn["b"]), cn["radius"].as<float>());
            }
        }
        if (obs["aabbs"]) {
            for (const YAML::Node& bn : obs["aabbs"]) {
                aabbs.emplace_back(read_vec3(bn["mn"]), read_vec3(bn["mx"]));
            }
        }
    }

    std::vector<PipeGroup> groups;
    for (const YAML::Node& gn : root["groups"]) {
        PipeGroup g(gn["id"].as<std::string>());
        if (!gn["terminals"]) {
            throw std::runtime_error("Problem::from_instance_file: group missing terminals");
        }
        for (const YAML::Node& tn : gn["terminals"]) {
            g.terminals.emplace_back(
                tn["id"].as<std::string>(),
                read_vec3(tn["pos"]),
                tn["radius"].as<float>());
        }
        groups.push_back(std::move(g));
    }

    const float max_pipe_r = compute_max_pipe_radius(groups);
    if (max_pipe_r <= 0.f) {
        throw std::runtime_error(
            "Problem::from_instance_file: need at least one terminal with positive radius");
    }

    // clearance_cap >= max_obstacle_radius + max_pipe_radius (grid README).
    // AABB is plane-based (no radius) — only spheres / capsules count.
    float max_obs_r = 0.f;
    for (const SphereObstacle& s : spheres) {
        max_obs_r = std::max(max_obs_r, s.radius);
    }
    for (const CapsuleObstacle& c : capsules) {
        max_obs_r = std::max(max_obs_r, c.radius);
    }

    // Floor: (max_obs_r + max_pipe_r) / max_pipe_r; 1.2× pipe for cell / AdaptiveStep slack.
    float clearance_relax = (max_obs_r + 1.2f * max_pipe_r) / max_pipe_r;
    if (root["meta"] && root["meta"]["clearance_relax"]) {
        clearance_relax = root["meta"]["clearance_relax"].as<float>();
    }
    if (clearance_relax < 1.f) {
        throw std::runtime_error("Problem::from_instance_file: clearance_relax must be >= 1");
    }

    GridParams gp(
        grid["size_x"].as<int>(),
        grid["size_y"].as<int>(),
        grid["size_z"].as<int>(),
        grid["resolution"].as<float>(),
        clearance_relax);

    return Problem(std::move(gp), std::move(spheres), std::move(capsules),
                   std::move(groups), std::move(aabbs));
}

}  // namespace pipe_design::problem
