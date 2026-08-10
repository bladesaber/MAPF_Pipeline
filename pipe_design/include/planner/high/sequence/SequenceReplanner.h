#pragma once

#include "topology/Types.h"

#include <string>
#include <vector>

namespace pipe_design::planner {

/// Failure-driven Task-order repair for PBS.
///
/// Preserves each group's relative Task order (interleaving OK; intra-group shuffle not).
/// Default: first failed Task's group moves as a block to just before the last
/// preceding foreign Task.
class SequenceReplanner {
public:
    SequenceReplanner() = default;
    virtual ~SequenceReplanner() = default;

    /// Return a new task_id sequence. Same as `order` means no progress (PBS → Failure).
    virtual std::vector<std::string> replan(
        const std::vector<std::string>& order,
        const std::vector<topology::Task>& tasks) const;
};

}  // namespace pipe_design::planner
