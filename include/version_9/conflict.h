#ifndef MAPF_PIPELINE_CONFLICT_H
#define MAPF_PIPELINE_CONFLICT_H

#include "common.h"

class Conflict{
public:
    ConstrainType constrain1;
    ConstrainType constrain2;

    size_t groupIdx1;
    double conflict1_x, conflict1_y, conflict1_z, conflict1_radius;

    size_t groupIdx2;
    double conflict2_x, conflict2_y, conflict2_z, conflict2_radius;

    Conflict(){}
    Conflict(
        size_t groupIdx1, double conflict1_x, double conflict1_y, double conflict1_z, double conflict1_radius,
        size_t groupIdx2, double conflict2_x, double conflict2_y, double conflict2_z, double conflict2_radius
    ):groupIdx1(groupIdx1), conflict1_x(conflict1_x), conflict1_y(conflict1_y), conflict1_z(conflict1_z), conflict1_radius(conflict1_radius),
      groupIdx2(groupIdx2), conflict2_x(conflict2_x), conflict2_y(conflict2_y), conflict2_z(conflict2_z), conflict2_radius(conflict2_radius)
      {};
    ~Conflict(){}

    void conflictExtend(){
        this->constrain1 = std::make_tuple(conflict1_x, conflict1_y, conflict1_z, conflict1_radius);
        this->constrain2 = std::make_tuple(conflict2_x, conflict2_y, conflict2_z, conflict2_radius);
    };

    void info(){
        std::cout << "Conflict:" << std::endl;
        std::cout << "   GroupIdx:" << groupIdx1 << " x:" << conflict1_x << " y:" << conflict1_y << " z:" << conflict1_z << " radius:" << conflict1_radius << std::endl;
        std::cout << "   GroupIdx:" << groupIdx2 << " x:" << conflict2_x << " y:" << conflict2_y << " z:" << conflict2_z << " radius:" << conflict2_radius << std::endl;
    }

};

#endif