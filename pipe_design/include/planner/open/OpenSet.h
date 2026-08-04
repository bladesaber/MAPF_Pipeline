#pragma once

#include "planner/low/SearchNode.h"

#include <cstddef>

namespace pipe_design::planner {

/// Default A* / best-first ordering: min f; on tie prefer larger g.
/// Returns true iff `a` is strictly worse than `b` (min-heap).
struct MinFgCompare {
    bool operator()(const SearchNode* a, const SearchNode* b) const {
        if (a->f != b->f) {
            return a->f > b->f;
        }
        return a->g < b->g;
    }
};

/// Abstract open set parameterized by node pointer type and strict worse-than compare.
///
/// Ownership: does not own Node*; the planner owns node storage.
/// Runtime-swappable backends use the default alias `OpenSet` below.
template <typename Node, typename Compare>
class OpenSetBase {
public:
    using NodeType = Node;
    using CompareType = Compare;

    virtual ~OpenSetBase() = default;

    virtual void clear() = 0;
    virtual bool empty() const = 0;
    virtual std::size_t size() const = 0;

    virtual void push(Node* node) = 0;
    virtual Node* pop() = 0;
    virtual Node* top() const = 0;
    virtual bool contains(const Node* node) const = 0;

    /// Remove `node` if present. Returns true iff it was in the open set.
    /// Does not own/delete the node. After a successful remove, the planner
    /// typically marks the node Closed (same as after `pop`).
    virtual bool remove(Node* node) = 0;

    /// After the node's key improved while already on open, restore heap order.
    /// Returns true if updated in place; false → caller must push(node) again.
    virtual bool decrease_key(Node* node) {
        (void)node;
        return false;
    }

protected:
    Compare comp_{};

    bool worse_than(const Node* a, const Node* b) const { return comp_(a, b); }
};

/// Grid A* open set (SearchNode* + MinFgCompare). DirectedSearchNode* is-a SearchNode*.
using OpenSet = OpenSetBase<SearchNode, MinFgCompare>;

}  // namespace pipe_design::planner
