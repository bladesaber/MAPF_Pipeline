#ifndef MAPF_PIPELINE_CONFLICT_H
#define MAPF_PIPELINE_CONFLICT_H

#include "common.h"

class Conflict{
public:
    ConstrainType constrain1;
    ConstrainType constrain2;

    size_ut agent1;
    double conflict1_x, conflict1_y, conflict1_z, conflict1_radius;
    double conflict1_length = DBL_MAX;

    size_ut agent2;
    double conflict2_x, conflict2_y, conflict2_z, conflict2_radius;
    double conflict2_length = DBL_MAX;

    Conflict(){};
    Conflict(
        size_ut agent1, 
        double conflict1_length,
        double conflict1_x, double conflict1_y, double conflict1_z, double conflict1_radius,
        
        size_ut agent2,
        double conflict2_length,
        double conflict2_x, double conflict2_y, double conflict2_z, double conflict2_radius
    ):agent1(agent1), 
      conflict1_x(conflict1_x), conflict1_y(conflict1_y), conflict1_z(conflict1_z), 
      conflict1_radius(conflict1_radius), conflict1_length(conflict1_length),
      agent2(agent2),
      conflict2_x(conflict2_x), conflict2_y(conflict2_y), conflict2_z(conflict2_z),
      conflict2_radius(conflict2_radius), conflict2_length(conflict2_length)
      {};
    
    ~Conflict(){}

    void conflictExtend(){
        this->constrain1 = std::make_tuple(conflict1_x, conflict1_y, conflict1_z, conflict1_radius);
        this->constrain2 = std::make_tuple(conflict2_x, conflict2_y, conflict2_z, conflict2_radius);
    };

    double getMinLength(){
        return std::min(conflict1_length, conflict2_length);
    }

    void info(){
        std::cout << "Conflict:" << std::endl;
        std::cout << "   AgentIdx:" << agent1 << " x:" << conflict1_x << " y:" << conflict1_y << " z:" << conflict1_z << " radius:" << conflict1_radius << " length:" << conflict1_length << std::endl;
        std::cout << "   AgentIdx:" << agent2 << " x:" << conflict2_x << " y:" << conflict2_y << " z:" << conflict2_z << " radius:" << conflict2_radius << " length:" << conflict2_length << std::endl;
    }

};

#endif