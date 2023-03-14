//
// Created by quan on 23-3-13.
//

#ifndef MAPF_PIPELINE_INSTANCE_H
#define MAPF_PIPELINE_INSTANCE_H

#include "common.h"

class Instance{
public:
    int num_of_cols = 0;
    int num_of_rows = 0;
    int map_size = 0;

    Instance(int num_of_rows = 0, int num_of_cols = 0){
        this->num_of_rows = num_of_rows;
        this->num_of_cols = num_of_cols;
        this->map_size = num_of_rows * num_of_cols;
    };
    ~Instance(){}

    inline int linearizeCoordinate(int row, int col) const {
        return ( this->num_of_cols * row + col);
    }
    inline int linearizeCoordinate(const std::pair<int, int>& curr) const {
        return linearizeCoordinate(curr.first, curr.second);
    }
    inline int getRowCoordinate(int curr) const {
        return curr / this->num_of_cols;
    }
    inline int getColCoordinate(int curr) const {
        return curr % this->num_of_cols;
    }
    inline std::pair<int, int> getCoordinate(int curr) const {
        return std::make_pair(curr / this->num_of_cols, curr % this->num_of_cols);
    }

    inline int getManhattanDistance(int loc1, int loc2) const
    {
        int loc1_x = getRowCoordinate(loc1);
        int loc1_y = getColCoordinate(loc1);
        int loc2_x = getRowCoordinate(loc2);
        int loc2_y = getColCoordinate(loc2);
        return abs(loc1_x - loc2_x) + abs(loc1_y - loc2_y);
    }
    inline int getManhattanDistance(const std::pair<int, int>& loc1, const std::pair<int, int>& loc2)
    {
        return abs(loc1.first - loc2.first) + abs(loc1.second - loc2.second);
    }

    std::list<int> getNeighbors(int curr) const;

};

#endif //MAPF_PIPELINE_INSTANCE_H
