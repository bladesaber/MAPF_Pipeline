//
// Created by quan on 23-3-14.
//

#ifndef MAPF_PIPELINE_CONFLICT_H
#define MAPF_PIPELINE_CONFLICT_H

#include "common.h"

enum constraint_type {
    LEQLENGTH, GLENGTH, RANGE, BARRIER, VERTEX, EDGE,
    POSITIVE_VERTEX, POSITIVE_EDGE, POSITIVE_BARRIER, POSITIVE_RANGE
    };

// agent, loc, timestep
typedef std::tuple<int, int, int, constraint_type> Constraint;

class Conflict{
public:
    Constraint constraint1;
    Constraint constraint2;

    Conflict(size_t a1, size_t a2, size_t a1_timeStep, size_t a2_timeStep, size_t loc):
        a1(a1), a2(a2), a1_timeStep(a1_timeStep), a2_timeStep(a2_timeStep), loc(loc)
    {};

    void vertexConflict(){
        this->constraint1 = std::make_tuple(a1, loc, 0, constraint_type::VERTEX);
        this->constraint2 = std::make_tuple(a2, loc, 0, constraint_type::VERTEX);
    };

    size_t a1;
    size_t a2;
    size_t a1_timeStep;
    size_t a2_timeStep;
    size_t loc;
};

#endif //MAPF_PIPELINE_CONFLICT_H
