#pragma once

#include "geom/Heading.h"
#include "grid/Types.h"

#include <vector>

namespace pipe_design::planner::motion {

/// One legal transition from a cell.
///
/// `heading` is the travel direction of this edge (from → cell), always a
/// unit Heading (or none only if from == cell, which models should not emit).
/// `cost` is length plus optional turn penalty vs the caller-supplied in-heading.
struct Successor {
    grid::CellCoord cell;
    geom::Heading heading{};
    double cost = 1.0;
};

/// Generates successors with per-edge heading and cost for grid search.
///
/// Turn penalties (default 0): when `in_heading` is set and `turn_weight` > 0,
/// `cost` includes extra cost proportional to heading misalignment.
class ActionModel {
public:
    virtual ~ActionModel() = default;

    void set_turn_weight(double weight) { turn_weight_ = weight; }
    double turn_weight() const { return turn_weight_; }

    /// Append successors of `from` given the heading used to enter `from`.
    /// Pass a none Heading at starts or when ignoring turns.
    virtual void successors(
        const grid::CellCoord& from,
        const geom::Heading& in_heading,
        std::vector<Successor>& out) const = 0;

    /// Convenience: no incoming heading (no turn penalty applied).
    void successors(const grid::CellCoord& from, std::vector<Successor>& out) const {
        successors(from, geom::Heading{}, out);
    }

protected:
    /// Combine travel length with soft turn penalty from heading change.
    double combine_edge_cost(
        const geom::Heading& in_heading,
        const geom::Heading& out_heading,
        double length) const {
        if (turn_weight_ == 0.0 || in_heading.is_none() || out_heading.is_none()) {
            return length;
        }
        if (in_heading.nearly_equal(out_heading)) {
            return length;
        }
        // Soft turn: misalignment in [0, 1] from alignment in [-1, 1].
        const double misalign =
            0.5 * (1.0 - static_cast<double>(in_heading.alignment(out_heading)));
        return length + turn_weight_ * misalign;
    }

    double turn_weight_ = 0.0;
};

}  // namespace pipe_design::planner::motion
