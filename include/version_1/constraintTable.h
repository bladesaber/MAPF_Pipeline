//
// Created by quan on 23-3-13.
//

#ifndef MAPF_PIPELINE_CONSTRAINTTABLE_H
#define MAPF_PIPELINE_CONSTRAINTTABLE_H

#include "common.h"
#include "conflict.h"

class ConstraintTable{
public:
    ConstraintTable(){};
    ~ConstraintTable(){
        ct.clear();
        cat.clear();
    };

    ConstraintTable(const ConstraintTable& other) {
        copy(other);
    }
    void copy(const ConstraintTable& other);
    // void insert2CT(size_t loc, int t_min, int t_max);
    void insert2CT(int loc);
    void insert2CAT(int loc);

    void insertConstrains2CT(std::vector<Constraint>& constrains);
    void insertPath2CAT(Path& path);

    // bool isConstrained(size_t loc, int t) const;
    bool isConstrained(size_t loc) const;
    int getNumOfConflictsForStep(int curr_loc, int next_loc) const;

    std::map<int, int>& getCT(){return ct;}
    std::map<int, int>& getCAT(){return cat;}

private:
    // location -> time range, or edge -> time range
    // boost::unordered_map<int, std::list<std::pair<int, int>>> ct;

    // Using Map is just because faster to find
    std::map<int, int> ct;
    std::map<int, int> cat;

};

#endif //MAPF_PIPELINE_CONSTRAINTTABLE_H
