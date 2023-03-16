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
    for (int next : candidates){
        if (next >= 0 && next < map_size){
            neighbors.emplace_back(next);
        }
    }
    return neighbors;
}

std::list<int> Instance3D::getNeighbors(int curr) const {
    std::list<int> neighbors;

    int x = getColCoordinate(curr);
    int y = getRowCoordinate(curr);
    int z = getZCoordinate(curr);
    std::list<std::tuple<int, int, int>> candidates{
        std::tuple<int, int, int>(y,   x+1, z),
        std::tuple<int, int, int>(y,   x-1, z),
        std::tuple<int, int, int>(y+1, x,   z),
        std::tuple<int, int, int>(y-1, x,   z),
        std::tuple<int, int, int>(y,   x,   z+1),
        std::tuple<int, int, int>(y,   x,   z-1)
    };

    for (auto next : candidates){
        x = std::get<0>(next);
        y = std::get<1>(next);
        z = std::get<2>(next);
        if (
            (x >= 0 && x < num_of_cols) && 
            (y >= 0 && y < num_of_rows) &&
            (z >= 0 && z < num_of_z)
        )
        {
            neighbors.emplace_back(linearizeCoordinate(y, x, z));
        }
    }
    return neighbors;
}