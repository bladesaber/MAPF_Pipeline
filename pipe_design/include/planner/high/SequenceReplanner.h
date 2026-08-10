#pragma once

#include "conflict/Types.h"

#include <string>
#include <vector>

namespace pipe_design::planner {

/// Conflict-driven Task-order repair for PBS.
///
/// Default: swap the two task_ids from the max-deficit conflict that both appear
/// in `order`. Override `replan` for other policies.
class SequenceReplanner {
public:
    SequenceReplanner() = default;
    virtual ~SequenceReplanner() = default;

    /// Return a new task_id sequence. Same as `order` means no progress (PBS → Failure).
    virtual std::vector<std::string> replan(
        const std::vector<std::string>& order,
        const std::vector<conflict::Conflict>& conflicts) const;
};

}  // namespace pipe_design::planner
