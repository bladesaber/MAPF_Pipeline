//
// Created by admin123456 on 2024/6/3.
//

#ifndef MAPF_PIPELINE_CONFLICT_UTILS_H
#define MAPF_PIPELINE_CONFLICT_UTILS_H

#include "common.h"

using namespace std;

class ConflictCell {
public:
    ConflictCell(
            size_t idx0, double conflict_x0, double conflict_y0, double conflict_z0, double conflict_radius0,
            size_t idx1, double conflict_x1, double conflict_y1, double conflict_z1, double conflict_radius1
    ) : idx0(idx0), idx1(idx1),
        conflict_x0(conflict_x0), conflict_y0(conflict_y0), conflict_z0(conflict_z0),
        conflict_x1(conflict_x1), conflict_y1(conflict_y1), conflict_z1(conflict_z1),
        conflict_radius0(conflict_radius0), conflict_radius1(conflict_radius1) {};

    ~ConflictCell() {};

private:
    size_t idx0, idx1;
    double conflict_x0, conflict_y0, conflict_z0, conflict_x1, conflict_y1, conflict_z1;
    double conflict_radius0, conflict_radius1;

};

#endif //MAPF_PIPELINE_CONFLICT_UTILS_H
