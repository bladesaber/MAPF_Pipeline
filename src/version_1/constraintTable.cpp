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

void ConstraintTable::insertConstrains2CT(std::vector<Constraint>& constrains){
    int a, loc, t;
	constraint_type type;
    
    for (const Constraint constrain : constrains)
    {
        std::tie(a, loc, t, type) = constrain;
        insert2CT(loc);
    }
}

// build the conflict avoidance table
void ConstraintTable::insert2CAT(int loc){
    if (this->cat.find(loc) != this->cat.end())
    {
        this->cat[loc] += 1;
    }else{
        this->cat[loc] = 1;
    }
}

void ConstraintTable::insertPath2CAT(Path& path){
    for (size_t loc : path)
    {
        insert2CAT(loc);
    }
}

bool ConstraintTable::isConstrained(size_t loc) const{
    const auto it = this->ct.find(loc);
    if (it == ct.end())
    {
        return false;
    }
    return true;
    
}

int ConstraintTable::getNumOfConflictsForStep(int curr_loc, int next_loc) const{
    const auto it = this->cat.find(next_loc);
    if (it == cat.end())
    {
        return 0;
    }
    return it->second;
}