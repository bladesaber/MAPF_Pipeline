//
// Created by quan on 23-3-14.
//

#ifndef MAPF_PIPELINE_CBSNODE_H
#define MAPF_PIPELINE_CBSNODE_H

#include "conflict.h"

class CBSNode{
public:
    CBSNode(){};

    std::list<Constraint> constraints;
};

#endif //MAPF_PIPELINE_CBSNODE_H
