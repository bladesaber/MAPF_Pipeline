#pragma once

#include "planner/low/ThetaStar.h"

namespace pipe_design::planner {

/// Lazy Theta*: assume parent visibility on expand; validate in SetVertex.
///
/// Same any-angle parent shortcuts as Theta*, but defers LOS checks until a
/// node is selected for expansion (`before_expand`). Usually fewer LOS tests.
class LazyThetaStar : public ThetaStar {
protected:
    void before_expand(SearchNode* current) override;
    void expand(SearchNode* current) override;

    /// Fix parent/g if assumed LOS(parent, current) fails (closed neighbors).
    void set_vertex(SearchNode* current);
};

}  // namespace pipe_design::planner
