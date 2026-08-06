#pragma once

#include "planner/low/LowLevelPlanner.h"
#include "planner/low/Path.h"

#include <cstddef>
#include <limits>

namespace pipe_design::planner {

/// Parallel to `LowLevelPlanner`: steppable multi-task session.
/// Session results live on internal Tasks; take a copy via derived `tasks()`.
class MultiQueryPlanner {
public:
    static constexpr std::size_t kUnlimited =
        std::numeric_limits<std::size_t>::max();

    MultiQueryPlanner() = default;
    virtual ~MultiQueryPlanner() = default;

    virtual SearchStatus step() = 0;

    /// Path of the last finished Task (empty if none).
    virtual Path last_path() const = 0;

    virtual SearchStatus search(std::size_t max_expansions = kUnlimited) {
        while (status_ == SearchStatus::Running && expansions_ < max_expansions) {
            step();
        }
        return status_;
    }

    SearchStatus status() const { return status_; }
    std::size_t expansions() const { return expansions_; }

protected:
    SearchStatus status_ = SearchStatus::Ready;
    std::size_t expansions_ = 0;
};

}  // namespace pipe_design::planner
