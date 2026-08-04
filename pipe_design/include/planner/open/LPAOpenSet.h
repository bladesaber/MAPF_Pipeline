#pragma once

#include "planner/low/LpaSearchNode.h"
#include "planner/open/PairingHeap.h"

#include <algorithm>
#include <limits>

namespace pipe_design::planner {

/// Lexicographic LPA* / D* Lite priority: [k1; k2] = [min(g,rhs)+h; min(g,rhs)].
struct LPAKey {
    double k1 = 0.0;
    double k2 = 0.0;

    static LPAKey inf() {
        constexpr double kInf = std::numeric_limits<double>::infinity();
        return {kInf, kInf};
    }

    bool operator<(const LPAKey& o) const {
        if (k1 != o.k1) {
            return k1 < o.k1;
        }
        return k2 < o.k2;
    }

    bool operator==(const LPAKey& o) const { return k1 == o.k1 && k2 == o.k2; }
    bool operator!=(const LPAKey& o) const { return !(*this == o); }
};

inline LPAKey lpa_key_of(const LpaSearchNode* n) {
    const double m = std::min(n->g, n->rhs);
    return LPAKey{m + n->h, m};
}

/// OpenSetBase worse-than: true iff `a` has strictly worse LPA key than `b`.
struct LpaKeyCompare {
    bool operator()(const LpaSearchNode* a, const LpaSearchNode* b) const {
        return lpa_key_of(b) < lpa_key_of(a);
    }
};

/// LPA* queue U: PairingHeapBase + Remove / TopKey (paper UpdateVertex).
template <typename Node = LpaSearchNode, typename Compare = LpaKeyCompare>
class LPAOpenSet : public PairingHeapBase<Node, Compare> {
public:
    using Base = PairingHeapBase<Node, Compare>;

    using Base::Base;
    using Base::clear;
    using Base::empty;
    using Base::size;
    using Base::push;
    using Base::pop;
    using Base::top;
    using Base::contains;
    using Base::remove;
    using Base::decrease_key;

    /// Key of current top; {inf,inf} if empty (paper TopKey).
    LPAKey top_key() const {
        if (this->empty()) {
            return LPAKey::inf();
        }
        return lpa_key_of(static_cast<const LpaSearchNode*>(this->top()));
    }
};

}  // namespace pipe_design::planner
