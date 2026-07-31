#include "planner/open/BucketHeap.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace pipe_design::planner {

BucketHeap::BucketHeap(double delta, double max_cost) : delta_(delta), max_cost_(max_cost) {
    if (!(delta_ > 0.0) || !std::isfinite(delta_)) {
        throw std::invalid_argument("BucketHeap: delta must be finite and > 0");
    }
    if (std::isnan(max_cost_) || max_cost_ < 0.0) {
        throw std::invalid_argument("BucketHeap: max_cost must be >= 0 (or +inf)");
    }
}

void BucketHeap::clear() {
    buckets_.clear();
    bin_of_.clear();
    size_ = 0;
}

bool BucketHeap::contains(const SearchNode* node) const {
    return bin_of_.find(node) != bin_of_.end();
}

int BucketHeap::bin_index(double f) const {
    if (std::isnan(f) || f < 0.0) {
        throw std::out_of_range("BucketHeap: f must be finite and >= 0");
    }
    if (!std::isfinite(f)) {
        throw std::out_of_range("BucketHeap: f must be finite");
    }
    if (f > max_cost_) {
        throw std::out_of_range("BucketHeap: f exceeds max_cost");
    }
    return static_cast<int>(std::floor(f / delta_));
}

void BucketHeap::erase_from_bin(SearchNode* node, int bin) {
    auto it = buckets_.find(bin);
    if (it == buckets_.end()) {
        throw std::logic_error("BucketHeap: missing bin during erase");
    }
    auto& vec = it->second;
    auto pos = std::find(vec.begin(), vec.end(), node);
    if (pos == vec.end()) {
        throw std::logic_error("BucketHeap: node not found in bin");
    }
    *pos = vec.back();
    vec.pop_back();
    if (vec.empty()) {
        buckets_.erase(it);
    }
}

void BucketHeap::insert_into_bin(SearchNode* node, int bin) {
    buckets_[bin].push_back(node);
    bin_of_[node] = bin;
}

void BucketHeap::push(SearchNode* node) {
    if (node == nullptr) {
        throw std::invalid_argument("BucketHeap::push: null node");
    }
    if (contains(node)) {
        throw std::logic_error("BucketHeap::push: node already in heap (use decrease_key)");
    }
    const int bin = bin_index(node->f);
    insert_into_bin(node, bin);
    ++size_;
}

SearchNode* BucketHeap::pop() {
    if (size_ == 0 || buckets_.empty()) {
        throw std::logic_error("BucketHeap::pop: empty");
    }

    auto bit = buckets_.begin();  // lowest bin key
    auto& vec = bit->second;
    std::size_t best_i = 0;
    for (std::size_t i = 1; i < vec.size(); ++i) {
        if (worse_than(vec[best_i], vec[i])) {
            best_i = i;
        }
    }

    SearchNode* best = vec[best_i];
    vec[best_i] = vec.back();
    vec.pop_back();
    bin_of_.erase(best);
    --size_;

    if (vec.empty()) {
        buckets_.erase(bit);
    }
    return best;
}

SearchNode* BucketHeap::top() const {
    if (size_ == 0 || buckets_.empty()) {
        throw std::logic_error("BucketHeap::top: empty");
    }
    const auto& vec = buckets_.begin()->second;
    std::size_t best_i = 0;
    for (std::size_t i = 1; i < vec.size(); ++i) {
        if (worse_than(vec[best_i], vec[i])) {
            best_i = i;
        }
    }
    return vec[best_i];
}

bool BucketHeap::decrease_key(SearchNode* node) {
    const auto it = bin_of_.find(node);
    if (it == bin_of_.end()) {
        return false;
    }
    const int old_bin = it->second;
    const int new_bin = bin_index(node->f);
    if (old_bin == new_bin) {
        // Still in same quantized bucket; exact order is fixed on pop scan.
        return true;
    }
    erase_from_bin(node, old_bin);
    insert_into_bin(node, new_bin);
    return true;
}

}  // namespace pipe_design::planner
