#pragma once

#include "planner/low/SearchNode.h"

#include <cstddef>

namespace pipe_design::planner {

/// Abstract open set for A* / best-first search.
///
/// Ownership: does not own SearchNode*; the planner owns node storage.
/// Key for ordering is SearchNode::f (min-heap: smaller f pops first).
/// Tie-break: larger g preferred (more progress) when f is equal.
class OpenSet {
public:
    virtual ~OpenSet() = default;

    virtual void clear() = 0;
    virtual bool empty() const = 0;
    virtual std::size_t size() const = 0;

    /// Insert node. Node should have status Open and a valid f.
    virtual void push(SearchNode* node) = 0;

    /// Remove and return the best node. Undefined if empty().
    virtual SearchNode* pop() = 0;

    /// Best node without removing. Undefined if empty().
    virtual SearchNode* top() const = 0;

    /// True if this open set currently tracks the node.
    virtual bool contains(const SearchNode* node) const = 0;

    /// After node->f decreased while already on open, restore heap order.
    /// Returns true if the heap was updated in place.
    /// Returns false if not applied — caller must push(node) again (lazy open sets).
    virtual bool decrease_key(SearchNode* node) {
        (void)node;
        return false;
    }

protected:
    /// Min-heap compare on f; on tie prefer larger g (smaller h).
    static bool worse_than(const SearchNode* a, const SearchNode* b) {
        if (a->f != b->f) {
            return a->f > b->f;
        }
        return a->g < b->g;
    }
};

}  // namespace pipe_design::planner
