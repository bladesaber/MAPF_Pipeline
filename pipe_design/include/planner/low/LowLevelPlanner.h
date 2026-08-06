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

    /// Optional callback around each `step()` inside `search()` (nullptr = off).
    using SearchStepHook = void (*)(LowLevelPlanner& planner, void* user);

    LowLevelPlanner() = default;
    virtual ~LowLevelPlanner() = default;

    virtual SearchStatus step() = 0;
    virtual Path path() const = 0;

    void set_before_step_hook(SearchStepHook hook, void* user = nullptr) {
        before_step_hook_ = hook;
        before_step_hook_user_ = user;
    }

    void set_after_step_hook(SearchStepHook hook, void* user = nullptr) {
        after_step_hook_ = hook;
        after_step_hook_user_ = user;
    }

    SearchStepHook before_step_hook() const { return before_step_hook_; }
    SearchStepHook after_step_hook() const { return after_step_hook_; }

    /// Repeated `step()` until not Running or `max_expansions` hit.
    /// Optional before/after hooks wrap each `step()` (timing / debug).
    virtual SearchStatus search(std::size_t max_expansions = kUnlimited) {
        while (status_ == SearchStatus::Running && expansions_ < max_expansions) {
            if (before_step_hook_ != nullptr) {
                before_step_hook_(*this, before_step_hook_user_);
            }
            step();
            if (after_step_hook_ != nullptr) {
                after_step_hook_(*this, after_step_hook_user_);
            }
        }
        return status_;
    }

    SearchStatus status() const { return status_; }
    std::size_t expansions() const { return expansions_; }

protected:
    SearchStatus status_ = SearchStatus::Ready;
    std::size_t expansions_ = 0;

private:
    SearchStepHook before_step_hook_ = nullptr;
    void* before_step_hook_user_ = nullptr;
    SearchStepHook after_step_hook_ = nullptr;
    void* after_step_hook_user_ = nullptr;
};

}  // namespace pipe_design::planner
