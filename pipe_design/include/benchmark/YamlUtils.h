#pragma once

#include "benchmark/GeneratorTypes.h"

#include <string>

#include <yaml-cpp/yaml.h>

namespace pipe_design::benchmark {

/// Write a constructive benchmark instance to YAML (ROADMAP §3).
/// `generator` is e.g. "axis_ray" / "plane_slice".
/// `generator_params` is merged into `meta` (method-specific knobs).
/// Writes `instance` obstacles under `obstacles.{spheres,capsules,aabbs}`.
bool write_instance_yaml(const std::string& path,
                         const Instance& instance,
                         const std::string& generator,
                         const YAML::Node& generator_params = {});

}  // namespace pipe_design::benchmark
