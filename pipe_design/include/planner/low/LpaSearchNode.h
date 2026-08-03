#pragma once

#include "planner/low/SearchNode.h"

#include <limits>

namespace pipe_design::planner {

/// Grid node for LPA* / D* Lite (adds `rhs`). Ordering is defined by the open-set Compare.
///
/// Convention (paper Initialize): unset nodes have g = rhs = +∞; start has rhs = 0.
struct LpaSearchNode : SearchNode {
    using Key = CellStateKey;

    static constexpr double kInf = std::numeric_limits<double>::infinity();

    double rhs = kInf;

    LpaSearchNode() {
        g = kInf;
        h = 0.0;
        f = 0.0;  // unused by LPA open-set ordering
    }

    explicit LpaSearchNode(grid::CellCoord coord_) : SearchNode(coord_) {
        g = kInf;
        rhs = kInf;
        h = 0.0;
        f = 0.0;
    }

    Key key() const { return Key{coord}; }

    static Key make_key(const grid::CellCoord& cell) { return Key{cell}; }

    static LpaSearchNode from_key(const Key& key) { return LpaSearchNode{key.coord}; }

    bool consistent() const { return g == rhs; }
    bool overconsistent() const { return g > rhs; }   // g > rhs → set g = rhs
    bool underconsistent() const { return g < rhs; }  // g < rhs → set g = ∞

    const LpaSearchNode* lpa_parent() const {
        return static_cast<const LpaSearchNode*>(parent);
    }
    LpaSearchNode* lpa_parent() { return static_cast<LpaSearchNode*>(parent); }
};

}  // namespace pipe_design::planner
