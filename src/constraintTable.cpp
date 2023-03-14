//
// Created by quan on 23-3-13.
//

#include "constraintTable.h"

void ConstraintTable::copy(const ConstraintTable& other){
    this->ct = other.ct;
}

void ConstraintTable::insert2CT(size_t loc, int t_min, int t_max){
    this->ct[loc].emplace_back(t_min, t_max);
}

// build the conflict avoidance table
void ConstraintTable::buildCAT(int agent, const std::vector<Path*>& paths, size_t cat_size){

}
