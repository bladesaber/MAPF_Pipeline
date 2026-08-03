#include "planner/open/PairingHeap.h"

#include "planner/open/LPAOpenSet.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace pipe_design::planner {

template <typename Node, typename Compare>
void PairingHeapBase<Node, Compare>::clear() {
    std::vector<HeapNode*> stack;
    if (root_ != nullptr) {
        stack.push_back(root_);
    }
    while (!stack.empty()) {
        HeapNode* node = stack.back();
        stack.pop_back();
        for (HeapNode* c = node->child; c != nullptr; ) {
            HeapNode* next = c->next;
            stack.push_back(c);
            c = next;
        }
        delete node;
    }
    root_ = nullptr;
    size_ = 0;
    handle_.clear();
}

template <typename Node, typename Compare>
bool PairingHeapBase<Node, Compare>::contains(const Node* node) const {
    return handle_.find(node) != handle_.end();
}

template <typename Node, typename Compare>
typename PairingHeapBase<Node, Compare>::HeapNode* PairingHeapBase<Node, Compare>::meld(
    HeapNode* a,
    HeapNode* b) {
    if (a == nullptr) {
        return b;
    }
    if (b == nullptr) {
        return a;
    }
    if (this->worse_than(a->item, b->item)) {
        std::swap(a, b);
    }
    // b becomes leftmost child of a.
    b->next = a->child;
    b->prev = a;
    if (a->child != nullptr) {
        a->child->prev = b;
    }
    a->child = b;
    return a;
}

template <typename Node, typename Compare>
typename PairingHeapBase<Node, Compare>::HeapNode* PairingHeapBase<Node, Compare>::merge_pairs(
    HeapNode* first) {
    if (first == nullptr) {
        return nullptr;
    }
    if (first->next == nullptr) {
        first->prev = nullptr;
        return first;
    }
    HeapNode* a = first;
    HeapNode* b = first->next;
    HeapNode* rest = b->next;
    a->next = nullptr;
    a->prev = nullptr;
    b->next = nullptr;
    b->prev = nullptr;
    return meld(meld(a, b), merge_pairs(rest));
}

template <typename Node, typename Compare>
void PairingHeapBase<Node, Compare>::detach(HeapNode* node) {
    if (node == root_) {
        return;
    }
    HeapNode* p = node->prev;
    if (p->child == node) {
        p->child = node->next;
    } else {
        p->next = node->next;
    }
    if (node->next != nullptr) {
        node->next->prev = p;
    }
    node->next = nullptr;
    node->prev = nullptr;
}

template <typename Node, typename Compare>
void PairingHeapBase<Node, Compare>::push(Node* node) {
    if (node == nullptr) {
        throw std::invalid_argument("PairingHeap::push: null node");
    }
    if (contains(node)) {
        throw std::logic_error("PairingHeap::push: node already in heap (use decrease_key)");
    }
    auto* hn = new HeapNode{};
    hn->item = node;
    handle_[node] = hn;
    root_ = meld(root_, hn);
    ++size_;
}

template <typename Node, typename Compare>
Node* PairingHeapBase<Node, Compare>::pop() {
    if (root_ == nullptr) {
        throw std::logic_error("PairingHeap::pop: empty");
    }
    HeapNode* old = root_;
    Node* item = old->item;
    handle_.erase(item);

    HeapNode* children = old->child;
    delete old;
    --size_;

    for (HeapNode* c = children; c != nullptr; c = c->next) {
        c->prev = nullptr;
    }
    root_ = merge_pairs(children);
    if (root_ != nullptr) {
        root_->prev = nullptr;
        root_->next = nullptr;
    }
    return item;
}

template <typename Node, typename Compare>
Node* PairingHeapBase<Node, Compare>::top() const {
    if (root_ == nullptr) {
        throw std::logic_error("PairingHeap::top: empty");
    }
    return root_->item;
}

template <typename Node, typename Compare>
bool PairingHeapBase<Node, Compare>::decrease_key(Node* node) {
    const auto it = handle_.find(node);
    if (it == handle_.end()) {
        return false;
    }
    HeapNode* hn = it->second;
    if (hn == root_) {
        return true;
    }
    detach(hn);
    root_ = meld(root_, hn);
    return true;
}

template class PairingHeapBase<SearchNode, MinFgCompare>;
template class PairingHeapBase<LpaSearchNode, LpaKeyCompare>;

}  // namespace pipe_design::planner
