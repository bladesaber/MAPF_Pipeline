//
// Created by quan on 23-3-13.
//

#include "instance.h"

std::list<int> Instance::getNeighbors(int curr) const {
    std::list<int> neighbors;
    int candidates[4] = {
            curr + 1,
            curr - 1,
            curr + num_of_cols,
            curr - num_of_cols
    };
    return neighbors;
}
