//
// Created by quan on 23-3-13.
//

#ifndef MAPF_PIPELINE_CONSTRAINTTABLE_H
#define MAPF_PIPELINE_CONSTRAINTTABLE_H

#include "common.h"

class ConstraintTable{
public:
    ConstraintTable(){};
    ~ConstraintTable(){this->ct = nullptr};

    ConstraintTable(const ConstraintTable& other) {
        copy(other);
    }
    void copy(const ConstraintTable& other);
    void insert2CT(size_t loc, int t_min, int t_max);

    void build(const CBSNode& node, int agent);
    void buildCAT(int agent, const std::vector<Path*>& paths, size_t cat_size);

private:
    boost::unordered_map<int, std::list<std::pair<int, int>>> ct; // location -> time range, or edge -> time range

};

#endif //MAPF_PIPELINE_CONSTRAINTTABLE_H
