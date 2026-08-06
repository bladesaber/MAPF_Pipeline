#pragma once

#include "planner/low/AStar.h"

#include <unordered_set>

namespace pipe_design::planner {

/// Paper: ARA*: Anytime A* with Provable Bounds on Sub-Optimality
/// Anytime Repairing A* (Likhachev et al.) — paper pseudocode mapped here.
///
/// fvalue(s):
///   f(s) = g(s) + ε · h(s)     →  `combine_key` / node->f
///
/// ImprovePath():                         →  repeated `step()` until done
///   while f(goal) > min_{s ∈ OPEN} f(s):
///     pop s with smallest f from OPEN; CLOSED ← CLOSED ∪ {s}
///     for each successor s' of s:
///       if s' unseen: g(s') ← ∞
///       if g(s') > g(s) + c(s,s'):
///         g(s') ← g(s) + c(s,s');  update parent
///         if s' ∉ CLOSED:  insert/update s' in OPEN with f(s')
///         else:            insert s' into INCONS
///   (stop when OPEN empty or open_->top()->f ≥ g(goal); goal g via goal_node_)
///
/// Main():                                →  `reset` + `search` / outer rounds in `step`
///   g(goal)←∞; g(start)←0; OPEN←{start}; CLOSED←INCONS←∅
///   ImprovePath();  publish solution;  ε' ← suboptimality_bound()
///   while ε' > 1:                        (we use eps_ > eps_target_, default target=1)
///     decrease ε
///     move INCONS into OPEN              →  incons_
///     update priorities of all s ∈ OPEN with new ε
///     CLOSED ← ∅
///     ImprovePath();  publish;  recompute ε'
///
/// Call `set_eps_schedule` before `reset`. Ownership same as A*.
/// While still repairing, `status` may stay `Running` but `path()` returns the
/// incumbent once `goal_node_` is set (`publish` ≈ path() / suboptimality_bound()).
class ARAStar : public AStar<SearchNode> {
public:
    /// ε schedule: start → target, subtracting `step` each repair round.
    void set_eps_schedule(double start, double target = 1.0, double step = 0.5);
    double eps() const { return eps_; }
    double eps_target() const { return eps_target_; }

    /// Paper ε': min(ε, g(goal) / min_{OPEN∪INCONS}(g+h)); ∞ if no solution yet.
    double suboptimality_bound() const;

    /// One ImprovePath expansion, or one Main repair transition (↓ε, rekey OPEN).
    SearchStatus step() override;
    /// Published incumbent path (OK while Running once a goal g is known).
    Path path() const override;

protected:
    double combine_key(double g, double h) const override;
    void expand(SearchNode* current) override;

    double eps_ = 3.0;
    double eps_target_ = 1.0;
    double eps_step_ = 0.5;
    std::unordered_set<SearchNode*> incons_;
};

}  // namespace pipe_design::planner
