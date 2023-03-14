//
// Created by quan on 23-3-13.
//

#include "constraintTable.h"

void ConstraintTable::copy(const ConstraintTable& other){
    this->ct = other.ct;
}

void ConstraintTable::insert2CT(int loc){
    this->ct[loc] = 1;
}

void ConstraintTable::buildCT(const CBSNode& node, int agent){
    auto curr = &node;
    int a, loc, t;
	constraint_type type;

    while (curr->parent != nullptr)
    {
        std::map<int, std::vector<Constraint>> constrains = *curr->constraints;
        std::vector<Constraint> agent_constrains = constrains[agent];
        for (const Constraint constrain : agent_constrains)
        {
            std::tie(a, loc, t, type) = constrain;
            insert2CT(loc);
        }
    }
}

// build the conflict avoidance table
void ConstraintTable::insert2CAT(int loc){
    this->cat.emplace_back(loc);
}

void ConstraintTable::buildCAT(const CBSNode& node, int agent){
    auto curr = &node;
    std::map<int, Path> paths = *node.paths;
    for (auto it = paths.begin(); it != paths.end(); it++) {
        if (it->first != agent)
        {
            for (PathEntry loc : it->second)
            {
                insert2CAT(loc.location);
            }
        }
    }

}

bool ConstraintTable::isConstrained(size_t loc) const{
    const auto& it = this->ct.find(loc);
    if (it == ct.end())
    {
        return false;
    }
    return true;
    
}
