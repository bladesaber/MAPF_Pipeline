#pragma once

#include "planner/low/LowLevelPlanner.h"
#include "topology/Types.h"

#include <string>
#include <vector>

namespace pipe_design::planner {

/// Outcome of `SequenceReplanner::replan` (not low-level / Task `SearchStatus`).
enum class ReplanStatus {
    Ready,     // cleared / not yet called
    Running,   // replan in progress
    Success,   // proposed a new unused order
    Failure,   // no progress (unchanged or already in history)
};

/// One attempted Task schedule: order + per-Task status (aligned with `order`).
struct PlanRecord {
    std::vector<std::string> order;
    std::vector<SearchStatus> statuses;
};

/// Base for failure-driven Task-order repair (PP).
///
/// Owns current `order_`, replan `status_`, and `history_`. Policy lives in subclasses;
/// `replan` is pure virtual. Intra-group relative order should stay fixed in all policies.
class SequenceReplanner {
public:
    SequenceReplanner() = default;
    virtual ~SequenceReplanner() = default;

    /// Clear `order_`, `history_`; `status_` → Ready.
    virtual void clear();

    /// Record / propose a new order. Subclass sets `status_` and updates `order_` / `history_`.
    /// On success: `status_` = Success and return the new order.
    /// On no progress: `status_` = Failure and return an empty order.
    virtual std::vector<std::string> replan(
        const std::vector<std::string>& order,
        const std::vector<topology::Task>& tasks) = 0;

    const std::vector<std::string>& order() const { return order_; }
    ReplanStatus status() const { return status_; }
    const std::vector<PlanRecord>& history() const { return history_; }

    /// True if `order` already appears in `history_`.
    bool order_used(const std::vector<std::string>& order) const;

    /// Whether `PrioritySequencer` should stop at the first Task failure.
    /// Default false: finish all Tasks so `replan` can see partial results.
    bool stop_on_failure() const { return stop_on_failure_; }
    void set_stop_on_failure(bool stop) { stop_on_failure_ = stop; }

protected:
    /// Build a history entry: `order` + statuses aligned to that order (from `tasks`).
    static PlanRecord make_plan_record(
        const std::vector<std::string>& order,
        const std::vector<topology::Task>& tasks);

    std::vector<std::string> order_;
    ReplanStatus status_ = ReplanStatus::Ready;
    std::vector<PlanRecord> history_;
    bool stop_on_failure_ = false;
};

}  // namespace pipe_design::planner
