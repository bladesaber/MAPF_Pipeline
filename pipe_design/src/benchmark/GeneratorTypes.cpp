#include "benchmark/GeneratorTypes.h"

#include <fstream>
#include <stdexcept>

namespace pipe_design::benchmark {
namespace {

YAML::Node vec3_node(const geom::Vec3& v) {
    YAML::Node n;
    n.push_back(v.x);
    n.push_back(v.y);
    n.push_back(v.z);
    return n;
}

YAML::Node cell_node(const grid::CellCoord& c) {
    YAML::Node n;
    n.push_back(c.x);
    n.push_back(c.y);
    n.push_back(c.z);
    return n;
}

geom::Vec3 read_vec3(const YAML::Node& n) {
    if (!n || !n.IsSequence() || n.size() < 3) {
        throw std::runtime_error("Instance::from_file: expected [x,y,z]");
    }
    return geom::Vec3{n[0].as<float>(), n[1].as<float>(), n[2].as<float>()};
}

grid::CellCoord read_cell(const YAML::Node& n) {
    if (!n || !n.IsSequence() || n.size() < 3) {
        throw std::runtime_error("Instance::from_file: expected cell [x,y,z]");
    }
    return grid::CellCoord{n[0].as<int>(), n[1].as<int>(), n[2].as<int>()};
}

}  // namespace

bool Instance::to_file(const std::string& path,
                       const std::string& generator,
                       const YAML::Node& generator_params) const {
    const GenParams& p = params;

    YAML::Node root;

    YAML::Node meta;
    meta["generator"] = generator;
    meta["success"] = success;
    if (!reason.empty()) {
        meta["reason"] = reason;
    }
    meta["seed"] = p.seed;
    meta["size_x"] = p.size_x;
    meta["size_y"] = p.size_y;
    meta["size_z"] = p.size_z;
    meta["resolution"] = GenParams::resolution;
    meta["radius"] = p.radius;
    meta["shell_margin"] = p.shell_margin;
    meta["shell_cells"] = p.shell_cells();
    meta["terminal_gap"] = p.terminal_gap;
    meta["num_groups"] = p.num_groups;
    meta["terminals_per_group"] = p.terminals_per_group;
    meta["max_iterations"] = p.max_iterations;
    meta["subdomain_n"] = p.subdomain_n;
    {
        YAML::Node obst;
        obst["density_target"] = obstacle_params.density_target;
        obst["corridor_margin"] = obstacle_params.corridor_margin;
        obst["seed"] = obstacle_params.seed;
        obst["max_rounds"] = obstacle_params.max_rounds;
        meta["obstacle"] = obst;
    }
    for (auto it = generator_params.begin(); it != generator_params.end(); ++it) {
        meta[it->first] = it->second;
    }
    root["meta"] = meta;

    YAML::Node grid;
    grid["size_x"] = p.size_x;
    grid["size_y"] = p.size_y;
    grid["size_z"] = p.size_z;
    grid["resolution"] = GenParams::resolution;
    root["grid"] = grid;

    YAML::Node obstacles;
    YAML::Node spheres_n;
    for (const geom::SphereObstacle& s : spheres) {
        YAML::Node sn;
        sn["center"] = vec3_node(s.center);
        sn["radius"] = s.radius;
        spheres_n.push_back(sn);
    }
    obstacles["spheres"] = spheres_n;
    YAML::Node capsules_n;
    for (const geom::CapsuleObstacle& c : capsules) {
        YAML::Node cn;
        cn["a"] = vec3_node(c.a);
        cn["b"] = vec3_node(c.b);
        cn["radius"] = c.radius;
        capsules_n.push_back(cn);
    }
    obstacles["capsules"] = capsules_n;
    YAML::Node aabbs_n;
    for (const geom::AabbObstacle& box : aabbs) {
        YAML::Node bn;
        bn["mn"] = vec3_node(box.mn);
        bn["mx"] = vec3_node(box.mx);
        aabbs_n.push_back(bn);
    }
    obstacles["aabbs"] = aabbs_n;
    root["obstacles"] = obstacles;

    YAML::Node groups_n;
    for (const GroupCorridors& g : groups) {
        YAML::Node gn;
        gn["id"] = g.id;
        YAML::Node terms;
        for (const auto& t : g.terminals) {
            YAML::Node tn;
            tn["id"] = t.id;
            tn["pos"] = vec3_node(t.pos);
            tn["radius"] = t.radius;
            terms.push_back(tn);
        }
        gn["terminals"] = terms;
        groups_n.push_back(gn);
    }
    root["groups"] = groups_n;

    YAML::Node solution;
    for (const GroupCorridors& g : groups) {
        YAML::Node gn;
        gn["id"] = g.id;
        YAML::Node paths;
        for (const auto& path : g.paths) {
            YAML::Node pn;
            pn["cost"] = path.cost;
            YAML::Node cells;
            for (const auto& c : path.cells) {
                cells.push_back(cell_node(c));
            }
            pn["cells"] = cells;
            paths.push_back(pn);
        }
        gn["paths"] = paths;
        solution.push_back(gn);
    }
    root["solution"] = solution;

    std::ofstream ofs(path);
    if (!ofs) {
        return false;
    }
    ofs << root;
    return static_cast<bool>(ofs);
}

Instance Instance::from_file(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);
    Instance out;

    const YAML::Node& grid = root["grid"];
    if (!grid) {
        throw std::runtime_error("Instance::from_file: missing grid in " + path);
    }
    out.params.size_x = grid["size_x"].as<int>();
    out.params.size_y = grid["size_y"].as<int>();
    out.params.size_z = grid["size_z"].as<int>();

    if (root["meta"]) {
        const YAML::Node& meta = root["meta"];
        out.success = meta["success"].as<bool>(true);
        out.reason = meta["reason"].as<std::string>("");
        out.params.seed = meta["seed"].as<std::uint32_t>(0);
        out.params.radius = meta["radius"].as<float>(0.5f);
        out.params.shell_margin = meta["shell_margin"].as<float>(0.f);
        out.params.terminal_gap = meta["terminal_gap"].as<int>(2);
        out.params.num_groups = meta["num_groups"].as<int>(0);
        out.params.max_iterations = meta["max_iterations"].as<int>(256);
        out.params.subdomain_n = meta["subdomain_n"].as<int>(3);
        if (meta["terminals_per_group"]) {
            out.params.terminals_per_group =
                meta["terminals_per_group"].as<std::vector<int>>();
        }
        if (meta["obstacle"]) {
            const YAML::Node& obst = meta["obstacle"];
            out.obstacle_params.density_target = obst["density_target"].as<float>(0.4f);
            out.obstacle_params.corridor_margin = obst["corridor_margin"].as<int>(-1);
            out.obstacle_params.seed = obst["seed"].as<std::uint32_t>(0);
            out.obstacle_params.max_rounds = obst["max_rounds"].as<int>(256);
        }
    }

    if (root["obstacles"]) {
        const YAML::Node& obs = root["obstacles"];
        if (obs["spheres"]) {
            for (const YAML::Node& sn : obs["spheres"]) {
                out.spheres.emplace_back(read_vec3(sn["center"]), sn["radius"].as<float>());
            }
        }
        if (obs["capsules"]) {
            for (const YAML::Node& cn : obs["capsules"]) {
                out.capsules.emplace_back(
                    read_vec3(cn["a"]), read_vec3(cn["b"]), cn["radius"].as<float>());
            }
        }
        if (obs["aabbs"]) {
            for (const YAML::Node& bn : obs["aabbs"]) {
                out.aabbs.emplace_back(read_vec3(bn["mn"]), read_vec3(bn["mx"]));
            }
        }
    }

    if (!root["groups"]) {
        throw std::runtime_error("Instance::from_file: missing groups in " + path);
    }
    for (const YAML::Node& gn : root["groups"]) {
        GroupCorridors g;
        g.id = gn["id"].as<std::string>();
        if (gn["terminals"]) {
            for (const YAML::Node& tn : gn["terminals"]) {
                g.terminals.emplace_back(
                    tn["id"].as<std::string>(),
                    read_vec3(tn["pos"]),
                    tn["radius"].as<float>());
            }
        }
        out.groups.push_back(std::move(g));
    }

    if (root["solution"]) {
        for (const YAML::Node& sn : root["solution"]) {
            const std::string id = sn["id"].as<std::string>();
            GroupCorridors* target = nullptr;
            for (GroupCorridors& g : out.groups) {
                if (g.id == id) {
                    target = &g;
                    break;
                }
            }
            if (target == nullptr) {
                out.groups.push_back(GroupCorridors{});
                out.groups.back().id = id;
                target = &out.groups.back();
            }
            if (sn["paths"]) {
                for (const YAML::Node& pn : sn["paths"]) {
                    planner::Path path;
                    path.cost = pn["cost"].as<double>(0.0);
                    if (pn["cells"]) {
                        for (const YAML::Node& cn : pn["cells"]) {
                            path.cells.push_back(read_cell(cn));
                        }
                    }
                    target->paths.push_back(std::move(path));
                }
            }
        }
    }

    return out;
}

}  // namespace pipe_design::benchmark
