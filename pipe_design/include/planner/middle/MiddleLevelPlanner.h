#pragma once

#include "planner/low/LowLevelPlanner.h"

namespace pipe_design::planner {

/// Speed / quality wrappers around a low-level planner.
/// Same `step` / `path` / `search` / `status` / `expansions` contract as `LowLevelPlanner`.
class MiddleLevelPlanner : public LowLevelPlanner {
public:
    ~MiddleLevelPlanner() override = default;
};

}  // namespace pipe_design::planner
