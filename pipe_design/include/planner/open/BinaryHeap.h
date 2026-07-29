#pragma once

#include "planner/open/OpenSet.h"

#include <unordered_map>
#include <vector>

namespace pipe_design::planner {

/// Binary min-heap with true decrease_key.
/// Handles live in this class (node → index map), not on SearchNode.
class BinaryHeap final : public OpenSet {
public:
    void clear() override;
    bool empty() const override { return heap_.empty(); }
    std::size_t size() const override { return heap_.size(); }

    void push(SearchNode* node) override;
    SearchNode* pop() override;
    bool contains(const SearchNode* node) const override;
    bool decrease_key(SearchNode* node) override;

private:
    void sift_up(std::size_t index);
    void sift_down(std::size_t index);
    void swap_indices(std::size_t i, std::size_t j);

    std::vector<SearchNode*> heap_;
    std::unordered_map<const SearchNode*, std::size_t> index_;
};

}  // namespace pipe_design::planner
