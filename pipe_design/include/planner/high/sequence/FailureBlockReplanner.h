#pragma once

#include "planner/high/sequence/SequenceReplanner.h"

namespace pipe_design::planner {

/// Default PBS policy: move the failed Task's group block before the last
/// preceding foreign Task (preserves each group's relative order).
class FailureBlockReplanner : public SequenceReplanner {
public:
    FailureBlockReplanner() = default;
    ~FailureBlockReplanner() override = default;

    std::vector<std::string> replan(
        const std::vector<std::string>& order,
        const std::vector<topology::Task>& tasks) override;
};

}  // namespace pipe_design::planner
