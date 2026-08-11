#pragma once

#include "problem/Problem.h"
#include "topology/Types.h"

namespace pipe_design::topology {

/// Base for `[T]` generators (PureMST, RadiusSpine, …).
///
/// Owns the `GlobalTopologyPlan` consumed by multi-group high-level
/// (CBS / PP / …). Derived `build(problem)` fills `plan_` from frozen input;
/// `replan()` is the LNS hook (may reuse `problem_`).
class TopologyBuilder {
public:
    TopologyBuilder() = default;
    virtual ~TopologyBuilder() = default;

    /// Fill `plan_` from `problem.groups()` (terminals / pipe ends).
    virtual void build(const problem::Problem& problem) = 0;

    /// Retopologize / LNS. Default: no-op.
    virtual void replan() {}

    const GlobalTopologyPlan& plan() const { return plan_; }
    GlobalTopologyPlan& plan() { return plan_; }

    const problem::Problem* problem() const { return problem_; }

protected:
    GlobalTopologyPlan plan_;
    const problem::Problem* problem_ = nullptr;  // set by derived `build`
};

}  // namespace pipe_design::topology
