#include "planner/open/BinaryHeap.h"

#include <stdexcept>
#include <utility>

namespace pipe_design::planner {

void BinaryHeap::clear() {
    heap_.clear();
    index_.clear();
}

bool BinaryHeap::contains(const SearchNode* node) const {
    return index_.find(node) != index_.end();
}

void BinaryHeap::swap_indices(std::size_t i, std::size_t j) {
    std::swap(heap_[i], heap_[j]);
    index_[heap_[i]] = i;
    index_[heap_[j]] = j;
}

void BinaryHeap::sift_up(std::size_t index) {
    while (index > 0) {
        const std::size_t parent = (index - 1) / 2;
        if (!worse_than(heap_[parent], heap_[index])) {
            break;
        }
        swap_indices(parent, index);
        index = parent;
    }
}

void BinaryHeap::sift_down(std::size_t index) {
    const std::size_t n = heap_.size();
    while (true) {
        std::size_t best = index;
        const std::size_t left = 2 * index + 1;
        const std::size_t right = 2 * index + 2;
        if (left < n && worse_than(heap_[best], heap_[left])) {
            best = left;
        }
        if (right < n && worse_than(heap_[best], heap_[right])) {
            best = right;
        }
        if (best == index) {
            break;
        }
        swap_indices(index, best);
        index = best;
    }
}

void BinaryHeap::push(SearchNode* node) {
    if (node == nullptr) {
        throw std::invalid_argument("BinaryHeap::push: null node");
    }
    if (contains(node)) {
        throw std::logic_error("BinaryHeap::push: node already in heap (use decrease_key)");
    }
    index_[node] = heap_.size();
    heap_.push_back(node);
    sift_up(heap_.size() - 1);
}

SearchNode* BinaryHeap::pop() {
    if (heap_.empty()) {
        throw std::logic_error("BinaryHeap::pop: empty");
    }
    SearchNode* top = heap_.front();
    index_.erase(top);
    if (heap_.size() == 1) {
        heap_.pop_back();
        return top;
    }
    heap_.front() = heap_.back();
    heap_.pop_back();
    index_[heap_.front()] = 0;
    sift_down(0);
    return top;
}

bool BinaryHeap::decrease_key(SearchNode* node) {
    const auto it = index_.find(node);
    if (it == index_.end()) {
        return false;
    }
    sift_up(it->second);
    return true;
}

}  // namespace pipe_design::planner
