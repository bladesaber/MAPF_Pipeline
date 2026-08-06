#pragma once

#include "planner/low/LowLevelPlanner.h"
#include "topology/Types.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace pipe_design::planner {

/// Thin base for high-level group-conflict resolvers (CBS, PBS, LNS, …).
///
/// Parallel control surface to `LowLevelPlanner` / `MultiQueryPlanner`:
///   - `step()`     — one high-level iteration (conflict-tree expand / priority fix / …)
///   - `search()`   — loop `step()` while Running
///   - `solution()` — incumbent Task results (by value; reuse-safe after another reset)
///
/// Does not own TopologyPlan / grid / multi drivers. Derived `reset(...)` binds them.
/// Do **not** inherit `LowLevelPlanner` or `MultiQueryPlanner`: result arity and
/// `step` semantics differ (conflict fix vs cell expand vs Task finish).
class HighLevelPlanner {
public:
    static constexpr std::size_t kUnlimited =
        std::numeric_limits<std::size_t>::max();

    /// Optional callback around each `step()` inside `search()` (nullptr = off).
    using SearchStepHook = void (*)(HighLevelPlanner& planner, void* user);

    HighLevelPlanner() = default;
    virtual ~HighLevelPlanner() = default;

    /// One conflict-resolution iteration. Derived defines the unit of work.
    virtual SearchStatus step() = 0;

    /// Incumbent multi-group Task results (copy; safe to keep after another `reset`).
    virtual std::vector<topology::Task> solution() const = 0;

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
