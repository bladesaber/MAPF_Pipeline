#include "planner/open/LazyBinaryHeap.h"

#include <stdexcept>

namespace pipe_design::planner {

void LazyBinaryHeap::clear() {
    heap_ = {};
}

void LazyBinaryHeap::discard_stale() const {
    while (!heap_.empty()) {
        const Entry& top = heap_.top();
        if (top.node != nullptr && top.node->status == NodeStatus::Open && top.f == top.node->f &&
            top.g == top.node->g) {
            break;
        }
        heap_.pop();
    }
}

bool LazyBinaryHeap::empty() const {
    discard_stale();
    return heap_.empty();
}

void LazyBinaryHeap::push(SearchNode* node) {
    heap_.push(Entry{node->f, node->g, node});
}

SearchNode* LazyBinaryHeap::pop() {
    discard_stale();
    if (heap_.empty()) {
        throw std::logic_error("LazyBinaryHeap::pop: empty");
    }
    SearchNode* node = heap_.top().node;
    heap_.pop();
    return node;
}

SearchNode* LazyBinaryHeap::top() const {
    discard_stale();
    if (heap_.empty()) {
        throw std::logic_error("LazyBinaryHeap::top: empty");
    }
    return heap_.top().node;
}

bool LazyBinaryHeap::contains(const SearchNode* node) const {
    return node != nullptr && node->status == NodeStatus::Open;
}

bool LazyBinaryHeap::remove(SearchNode* node) {
    // Cannot unlink from std::priority_queue; stale entries are skipped once
    // the planner marks the node Closed (same contract as expand-after-pop).
    return contains(node);
}

bool LazyBinaryHeap::decrease_key(SearchNode* node) {
    (void)node;
    // No in-place fix: caller must push(node) again with the new f/g.
    return false;
}

}  // namespace pipe_design::planner
