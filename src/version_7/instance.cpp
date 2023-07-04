#include "instance.h"

std::list<int> Instance::getNeighbors(int curr) const {
    std::list<int> neighbors;

    int x = getXCoordinate(curr);
    int y = getYCoordinate(curr);
    int z = getZCoordinate(curr);
    std::list<std::tuple<int, int, int>> candidates{
        std::tuple<int, int, int>(x+1, y,   z),
        std::tuple<int, int, int>(x-1, y,   z),
        std::tuple<int, int, int>(x,   y+1, z),
        std::tuple<int, int, int>(x,   y-1, z),
        std::tuple<int, int, int>(x,   y,   z+1),
        std::tuple<int, int, int>(x,   y,   z-1)
    };

    for (auto next : candidates){
        x = std::get<0>(next);
        y = std::get<1>(next);
        z = std::get<2>(next);
        if (
            (x >= 0 && x < num_of_x) && 
            (y >= 0 && y < num_of_y) &&
            (z >= 0 && z < num_of_z)
        )
        {
            neighbors.emplace_back(linearizeCoordinate(x=x, y=y, z=z));
        }
    }
    return neighbors;
}

bool Instance::isValidPos(double x, double y, double z){
    if (x < 0 || x > num_of_x - 1)
    {
        return false;
    }

    if (y < 0 || y > num_of_y - 1)
    {
        return false;
    }
    
    if (z < 0 || z > num_of_z - 1)
    {
        return false;
    }

    return true;
}