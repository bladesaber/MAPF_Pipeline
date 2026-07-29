#pragma once

#include "planner/open/OpenSet.h"

#include <queue>
#include <stdexcept>
#include <vector>

namespace pipe_design::planner {

/// Lazy binary heap: decrease_key = push a fresh entry; stale pops are skipped.
/// Often fastest for grid A* when reopenings are rare.
///
/// Contract: keep node->status == Open while the node is logically on open.
/// When expanding, set Closed so stale duplicates are discarded on pop/empty.
/// After improving f/g, call push(node) again (decrease_key is a no-op hint).
class LazyBinaryHeap final : public OpenSet {
public:
    void clear() override;
    bool empty() const override;
    std::size_t size() const override { return heap_.size(); }

    void push(SearchNode* node) override;
    SearchNode* pop() override;
    bool contains(const SearchNode* node) const override;
    bool decrease_key(SearchNode* node) override;

private:
    struct Entry {
        double f = 0.0;
        double g = 0.0;
        SearchNode* node = nullptr;
    };

    struct Greater {
        bool operator()(const Entry& a, const Entry& b) const {
            if (a.f != b.f) {
                return a.f > b.f;
            }
            return a.g < b.g;
        }
    };

    void discard_stale() const;

    mutable std::priority_queue<Entry, std::vector<Entry>, Greater> heap_;
};

}  // namespace pipe_design::planner
