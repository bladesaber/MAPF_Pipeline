#pragma once

#include "planner/low/Path.h"

#include <cstddef>
#include <limits>

namespace pipe_design::planner {

enum class SearchStatus {
    Ready,     // reset not called, or cleared
    Running,   // search in progress
    Success,   // goal reached
    Failure,   // exhausted open set / invalid setup
};

/// Thin base for low-level single-query planners (A*, LPA*, …).
///
/// Owns search status counters only. Does not own grid / heuristic / open / actions.
class LowLevelPlanner {
public:
    static constexpr std::size_t kUnlimited =
        std::numeric_limits<std::size_t>::max();

    LowLevelPlanner() = default;
    virtual ~LowLevelPlanner() = default;

    virtual SearchStatus step() = 0;
    virtual Path path() const = 0;

    /// Repeated `step()` until not Running or `max_expansions` hit.
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
