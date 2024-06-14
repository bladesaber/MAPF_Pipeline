//
// Created by admin123456 on 2024/6/3.
//

#ifndef MAPF_PIPELINE_CONFLICT_UTILS_H
#define MAPF_PIPELINE_CONFLICT_UTILS_H

#include "common.h"

using namespace std;

class ConflictCell {
public:
    size_t idx0, idx1;
    double x0, y0, z0, x1, y1, z1;
    double radius0, radius1;

    ConflictCell(
            size_t idx0, double x0, double y0, double z0, double radius0,
            size_t idx1, double x1, double y1, double z1, double radius1
    ) : idx0(idx0), idx1(idx1), x0(x0), y0(y0), z0(z0), x1(x1), y1(y1), z1(z1), radius0(radius0), radius1(radius1) {};

    ~ConflictCell() {};

};

#endif //MAPF_PIPELINE_CONFLICT_UTILS_H
