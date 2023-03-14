//
// Created by quan on 23-3-13.
//

#ifndef MAPF_PIPELINE_CONSTRAINTTABLE_H
#define MAPF_PIPELINE_CONSTRAINTTABLE_H

#include "common.h"
#include "cbsNode.h"

class ConstraintTable{
public:
    ConstraintTable(){};
    ~ConstraintTable(){};

    ConstraintTable(const ConstraintTable& other) {
        copy(other);
    }
    void copy(const ConstraintTable& other);
    // void insert2CT(size_t loc, int t_min, int t_max);
    void insert2CT(int loc);
    void insert2CAT(int loc);

    void buildCT(const CBSNode& node, int agent);
    void buildCAT(const CBSNode& node, int agent);

    // bool isConstrained(size_t loc, int t) const;
    bool isConstrained(size_t loc) const;

private:
    // location -> time range, or edge -> time range
    // boost::unordered_map<int, std::list<std::pair<int, int>>> ct;

    std::map<int, int> ct;
    std::vector<int> cat;

};

#endif //MAPF_PIPELINE_CONSTRAINTTABLE_H
