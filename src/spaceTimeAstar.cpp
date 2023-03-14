//
// Created by quan on 23-3-13.
//

#include "spaceTimeAstar.h"

SpaceTimeAStar::SpaceTimeAStar(Instance &instance) {
    this->instance = &instance;
}

bool SpaceTimeAStar::validMove(int curr, int next) const {
    if (next < 0 || next >= this->instance->map_size){
        return false;
    }
    return true;
}
