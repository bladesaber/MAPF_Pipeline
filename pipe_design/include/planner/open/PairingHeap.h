#pragma once

#include "planner/open/OpenSet.h"

#include <unordered_map>

namespace pipe_design::planner {

/// Pairing min-heap with decrease_key. Handles stored in a side map (not on Node).
template <typename Node = SearchNode, typename Compare = MinFgCompare>
class PairingHeapBase: public OpenSetBase<Node, Compare> {
public:
    PairingHeapBase() = default;
    ~PairingHeapBase() override { clear(); }

    PairingHeapBase(const PairingHeapBase&) = delete;             // 禁止拷贝构造
    PairingHeapBase& operator=(const PairingHeapBase&) = delete;  // 禁止赋值操作

    void clear() override;
    bool empty() const override { return root_ == nullptr; }
    std::size_t size() const override { return size_; }

    void push(Node* node) override;
    Node* pop() override;
    Node* top() const override;
    bool contains(const Node* node) const override;
    bool remove(Node* node) override;
    bool decrease_key(Node* node) override;

protected:
    struct HeapNode {
        Node* item = nullptr;
        HeapNode* child = nullptr;
        HeapNode* next = nullptr;
        HeapNode* prev = nullptr;  // sibling prev, or parent if leftmost child
    };

    HeapNode* meld(HeapNode* a, HeapNode* b);
    HeapNode* merge_pairs(HeapNode* first);
    void detach(HeapNode* node);

    HeapNode* root_ = nullptr;
    std::size_t size_ = 0;
    std::unordered_map<const Node*, HeapNode*> handle_;
};

using PairingHeap = PairingHeapBase<SearchNode, MinFgCompare>;

extern template class PairingHeapBase<SearchNode, MinFgCompare>;

}  // namespace pipe_design::planner
