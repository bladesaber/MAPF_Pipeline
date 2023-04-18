#ifndef MAPF_PIPELINE_CONSTRAINTTABLE_H
#define MAPF_PIPELINE_CONSTRAINTTABLE_H

#include "common.h"
#include "utils.h"
#include "instance.h"

class ConstraintTable{
public:
    ConstraintTable(){};
    ~ConstraintTable(){
        ct.clear();
    };

    void insert2CT(int loc, double radius);

    bool isConstrained(int loc) const;

    bool islineOnSight(Instance& instance, int parent_loc, int child_loc, double bound) const;

    std::map<int, double>& getCT(){
        return ct;
    }

private:
    std::map<int, double> ct;

};

#endif