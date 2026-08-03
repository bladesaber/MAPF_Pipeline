#include "planner/open/LPAOpenSet.h"

namespace pipe_design::planner {

template <typename Node, typename Compare>
bool LPAOpenSet<Node, Compare>::remove(Node* node) {
    const auto it = this->handle_.find(node);
    if (it == this->handle_.end()) {
        return false;
    }
    auto* hn = it->second;
    this->handle_.erase(it);
    --this->size_;

    if (hn == this->root_) {
        auto* children = hn->child;
        delete hn;
        for (auto* c = children; c != nullptr; c = c->next) {
            c->prev = nullptr;
        }
        this->root_ = this->merge_pairs(children);
        if (this->root_ != nullptr) {
            this->root_->prev = nullptr;
            this->root_->next = nullptr;
        }
        return true;
    }

    this->detach(hn);
    auto* children = hn->child;
    delete hn;
    for (auto* c = children; c != nullptr; c = c->next) {
        c->prev = nullptr;
    }
    this->root_ = this->meld(this->root_, this->merge_pairs(children));
    return true;
}

template class LPAOpenSet<LpaSearchNode, LpaKeyCompare>;

}  // namespace pipe_design::planner
