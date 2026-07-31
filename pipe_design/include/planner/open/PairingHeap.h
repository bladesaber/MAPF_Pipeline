#pragma once

#include "planner/open/OpenSet.h"

#include <unordered_map>

namespace pipe_design::planner {

/// Pairing min-heap with decrease_key. Handles stored in a side map (not on SearchNode).
class PairingHeap final : public OpenSet {
public:
    PairingHeap() = default;
    ~PairingHeap() override { clear(); }

    PairingHeap(const PairingHeap&) = delete;             // 禁止拷贝构造
    PairingHeap& operator=(const PairingHeap&) = delete;  // 禁止赋值操作

    void clear() override;
    bool empty() const override { return root_ == nullptr; }
    std::size_t size() const override { return size_; }

    void push(SearchNode* node) override;
    SearchNode* pop() override;
    SearchNode* top() const override;
    bool contains(const SearchNode* node) const override;
    bool decrease_key(SearchNode* node) override;

private:
    struct HeapNode {
        SearchNode* item = nullptr;
        HeapNode* child = nullptr;
        HeapNode* next = nullptr;
        HeapNode* prev = nullptr;  // sibling prev, or parent if leftmost child
    };

    static HeapNode* meld(HeapNode* a, HeapNode* b);
    static HeapNode* merge_pairs(HeapNode* first);
    void detach(HeapNode* node);

    HeapNode* root_ = nullptr;
    std::size_t size_ = 0;
    std::unordered_map<const SearchNode*, HeapNode*> handle_;
};

}  // namespace pipe_design::planner
