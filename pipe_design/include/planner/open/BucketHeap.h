#pragma once

#include "planner/open/OpenSet.h"

#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

namespace pipe_design::planner {

/// Sparse bucket open set (Dial-style) backed by std::map<bin, vector>.
///
/// bin = floor(f / delta). Only non-empty bins exist in the map, so small delta
/// does not allocate a huge array. If f > max_cost (or f < 0), push/decrease_key throw.
///
/// Within a bin, nodes are kept in an unsorted vector; pop scans the lowest bin
/// for the best node by `worse_than` / `MinFgCompare` (exact f/g, not only quantized key).
///
/// decrease_key moves the node from its old bin to the new bin.
class BucketHeap final : public OpenSet {
public:
    /// @param delta      Cost quantization step; must be > 0.
    /// @param max_cost   Inclusive upper bound on node->f; exceeding throws.
    explicit BucketHeap(
        double delta,
        double max_cost = std::numeric_limits<double>::infinity());

    double delta() const { return delta_; }
    double max_cost() const { return max_cost_; }

    void clear() override;
    bool empty() const override { return size_ == 0; }
    std::size_t size() const override { return size_; }

    void push(SearchNode* node) override;
    SearchNode* pop() override;
    SearchNode* top() const override;
    bool contains(const SearchNode* node) const override;
    bool remove(SearchNode* node) override;
    bool decrease_key(SearchNode* node) override;

private:
    int bin_index(double f) const;
    void erase_from_bin(SearchNode* node, int bin);
    void insert_into_bin(SearchNode* node, int bin);

    double delta_ = 1.0;
    double max_cost_ = std::numeric_limits<double>::infinity();
    std::size_t size_ = 0;

    std::map<int, std::vector<SearchNode*>> buckets_;
    std::unordered_map<const SearchNode*, int> bin_of_;
};

}  // namespace pipe_design::planner
