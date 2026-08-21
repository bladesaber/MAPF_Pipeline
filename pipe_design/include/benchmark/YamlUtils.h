#pragma once

#include "benchmark/GeneratorTypes.h"

#include <string>

#include <yaml-cpp/yaml.h>

namespace pipe_design::benchmark {

/// Write a constructive benchmark instance to YAML (ROADMAP §3).
/// `generator` is e.g. "axis_ray" / "plane_slice".
/// `generator_params` is merged into `meta` (method-specific knobs).
/// Obstacles are empty until obstacle paint (§2) exists.
bool write_instance_yaml(const std::string& path,
                         const Instance& instance,
                         const std::string& generator,
                         const YAML::Node& generator_params = {});

}  // namespace pipe_design::benchmark
