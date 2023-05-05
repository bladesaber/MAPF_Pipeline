#ifndef MAPF_PIPELINE_INSTANCE_H
#define MAPF_PIPELINE_INSTANCE_H

#include "common.h"

class Instance{
public:
    int num_of_x = 0;
    int num_of_y = 0;
    int num_of_z = 0;

    int XY_area = 0;
    int map_size = 0;

    Instance(int num_of_x = 0, int num_of_y = 0, int num_of_z = 0){
        this->num_of_y = num_of_y;
        this->num_of_x = num_of_x;
        this->num_of_z = num_of_z;

        this->XY_area = num_of_y * num_of_x;
        this->map_size = num_of_y * num_of_x * num_of_z;
    }
    ~Instance(){}

    inline int linearizeCoordinate(int x, int y, int z) const {
        return this->XY_area * z + this->num_of_x * y + x;
    }
    inline int linearizeCoordinate(const std::tuple<int, int, int>& curr) const {
        return linearizeCoordinate(std::get<0>(curr), std::get<1>(curr), std::get<2>(curr));
    }

    inline int getZCoordinate(int curr) const {
        return curr / this->XY_area;
    }
    inline int getYCoordinate(int curr) const {
        return (curr - getZCoordinate(curr) * this->XY_area) / this->num_of_x;
    }
    inline int getXCoordinate(int curr) const {
        return (curr - getZCoordinate(curr) * this->XY_area) % this->num_of_x;
    }
    inline std::tuple<int, int, int> getCoordinate(int curr) const {
        std::tuple<int, int, int> res = std::make_tuple(
            getXCoordinate(curr), getYCoordinate(curr), getZCoordinate(curr)
        );
        return res;
    }

    inline double getManhattanDistance(int loc1, int loc2) const
    {
        int loc1_x = getXCoordinate(loc1);
        int loc1_y = getYCoordinate(loc1);
        int loc1_z = getZCoordinate(loc1);

        int loc2_x = getXCoordinate(loc2);
        int loc2_y = getYCoordinate(loc2);
        int loc2_z = getZCoordinate(loc2);

        return (double)abs(loc1_x - loc2_x) + 
               (double)abs(loc1_y - loc2_y) + 
               (double)abs(loc1_z - loc2_z);
    }

    inline double getEulerDistance(int loc1, int loc2){
        int loc1_x = getXCoordinate(loc1);
        int loc1_y = getYCoordinate(loc1);
        int loc1_z = getZCoordinate(loc1);

        int loc2_x = getXCoordinate(loc2);
        int loc2_y = getYCoordinate(loc2);
        int loc2_z = getZCoordinate(loc2);

        double distance = sqrt(
            pow(loc1_x - loc2_x, 2) + 
            pow(loc1_y - loc2_y, 2) + 
            pow(loc1_z - loc2_z, 2)
        );

        return distance;
    }

    std::list<int> getNeighbors(int curr) const;

    bool isValidPos(double x, double y, double z);

    void info(){
        std::cout << "X:" << num_of_x << " Y:" << num_of_y << " Z:" << num_of_z << " Map Size:" << map_size << std::endl;
    }
};

#endif