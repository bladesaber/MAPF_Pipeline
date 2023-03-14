//
// Created by quan on 23-3-14.
//

#ifndef MAPF_PIPELINE_CONFLICT_H
#define MAPF_PIPELINE_CONFLICT_H

#include "common.h"

enum constraint_type { LEQLENGTH, GLENGTH, RANGE, BARRIER, VERTEX, EDGE,
    POSITIVE_VERTEX, POSITIVE_EDGE, POSITIVE_BARRIER, POSITIVE_RANGE };

typedef std::tuple<int, int, int, int, constraint_type> Constraint;

#endif //MAPF_PIPELINE_CONFLICT_H
