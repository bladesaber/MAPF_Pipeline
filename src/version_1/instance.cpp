//
// Created by quan on 23-3-13.
//

#include "instance.h"

std::list<int> Instance::getNeighbors(int curr) const {
    std::list<int> neighbors;

    int x = getColCoordinate(curr);
    int y = getRowCoordinate(curr);
    std::list<std::pair<int, int>> candidates{
        std::pair<int, int>(y,   x+1),
        std::pair<int, int>(y,   x-1),
        std::pair<int, int>(y+1, x  ),
        std::pair<int, int>(y-1, x  )
    };

    for (auto next : candidates){
        y = next.first;
        x = next.second;
        if (
            (x >= 0 && x < num_of_cols) && 
            (y >= 0 && y < num_of_rows)
        )
        {
            neighbors.emplace_back(linearizeCoordinate(y, x));
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
        y = std::get<0>(next);
        x = std::get<1>(next);
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