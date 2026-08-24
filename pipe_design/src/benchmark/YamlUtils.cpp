#include "benchmark/YamlUtils.h"

#include <fstream>

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

}  // namespace

bool write_instance_yaml(const std::string& path,
                         const Instance& instance,
                         const std::string& generator,
                         const YAML::Node& generator_params) {
    const GenParams& p = instance.params;

    YAML::Node root;

    YAML::Node meta;
    meta["generator"] = generator;
    meta["success"] = instance.success;
    if (!instance.reason.empty()) {
        meta["reason"] = instance.reason;
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
        const ObstacleGenParams& op = instance.obstacle_params;
        YAML::Node obst;
        obst["density_target"] = op.density_target;
        obst["corridor_margin"] = op.corridor_margin;
        obst["seed"] = op.seed;
        obst["max_rounds"] = op.max_rounds;
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
    YAML::Node spheres;
    for (const geom::SphereObstacle& s : instance.spheres) {
        YAML::Node sn;
        sn["center"] = vec3_node(s.center);
        sn["radius"] = s.radius;
        spheres.push_back(sn);
    }
    obstacles["spheres"] = spheres;
    YAML::Node capsules;
    for (const geom::CapsuleObstacle& c : instance.capsules) {
        YAML::Node cn;
        cn["a"] = vec3_node(c.a);
        cn["b"] = vec3_node(c.b);
        cn["radius"] = c.radius;
        capsules.push_back(cn);
    }
    obstacles["capsules"] = capsules;
    YAML::Node aabbs;
    for (const geom::AabbObstacle& box : instance.aabbs) {
        YAML::Node bn;
        bn["mn"] = vec3_node(box.mn);
        bn["mx"] = vec3_node(box.mx);
        aabbs.push_back(bn);
    }
    obstacles["aabbs"] = aabbs;
    root["obstacles"] = obstacles;

    YAML::Node groups;
    for (const GroupCorridors& g : instance.groups) {
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
        groups.push_back(gn);
    }
    root["groups"] = groups;

    YAML::Node solution;
    for (const GroupCorridors& g : instance.groups) {
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

}  // namespace pipe_design::benchmark
