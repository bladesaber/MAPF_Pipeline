//
// Created by quan on 23-3-13.
//

#ifndef MAPF_PIPELINE_SPACETIMEASTAR_H
#define MAPF_PIPELINE_SPACETIMEASTAR_H

#include "iostream"
#include "common.h"
#include "instance.h"

class SpaceTimeAStar{
public:
    SpaceTimeAStar(Instance& instance);
    ~SpaceTimeAStar(){};

    Path findPath(const std::pair<int, int> start_state, std::pair<int, int> goal_state);
    int getHeuristic(int loc1, int loc2);
    int getHeuristic(const std::pair<int, int>& loc1, const std::pair<int, int>& loc2);

    bool validMove(int curr, int next) const;

private:
    Instance* instance;

};

#endif //MAPF_PIPELINE_SPACETIMEASTAR_H
