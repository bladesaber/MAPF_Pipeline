#pragma once

#include "planner/low/AStar.h"

namespace pipe_design::planner {

/// Anytime Nonparametric A*: no ε schedule. Open key maximizes
/// e(s) = (G − g(s)) / h(s) (stored as f = −e for the min-heap).
/// Improves incumbent G until open is exhausted (optimal) or budget ends.
///
/// Call `reset` as with A*. `path()` returns the incumbent while still Running.
class ANAStar : public AStar<SearchNode> {
public:
    SearchStatus step() override;
    SearchStatus search(std::size_t max_expansions = kUnlimited) override;
    Path path() const override;

    /// Incumbent solution cost G (∞ before first goal).
    double incumbent_cost() const;

protected:
    double combine_key(double g, double h) const override;
    void expand(SearchNode* current) override;
};

}  // namespace pipe_design::planner
