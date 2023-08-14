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
    inline int getManhattanDistance(const std::pair<int, int>& loc1, const std::pair<int, int>& loc2) const
    {
        return abs(loc1.first - loc2.first) + abs(loc1.second - loc2.second);
    }

    std::list<int> getNeighbors(int curr) const;

    void print(){
        std::cout << "Row: " << num_of_rows << " Cols: " << num_of_cols << " Map Size: " << map_size << std::endl;
    }

    inline void printCoordinate(int curr){
        std::pair<int, int> coodr = getCoordinate(curr);
        std::cout << curr << ":(" << coodr.first << " ," << coodr.second << ")";
    }

};

class Instance3D: public Instance{
public:
    int num_of_z = 0;
    int row_col_area = 0;

    Instance3D(int num_of_rows = 0, int num_of_cols = 0, int num_of_z = 0){
        this->num_of_rows = num_of_rows;
        this->num_of_cols = num_of_cols;
        this->row_col_area = num_of_rows * num_of_cols;
        this->num_of_z = num_of_z;
        this->map_size = num_of_rows * num_of_cols * num_of_z;
    }
    inline int linearizeCoordinate(int row, int col, int z) const {
        return this->row_col_area * z + this->num_of_cols * row + col;
    }
    inline int linearizeCoordinate(const std::tuple<int, int, int>& curr) const {
        return linearizeCoordinate(std::get<0>(curr), std::get<1>(curr), std::get<2>(curr));
    }

    inline int getZCoordinate(int curr) const {
        return curr / this->row_col_area;
    }
    inline int getRowCoordinate(int curr) const {
        return (curr - getZCoordinate(curr) * this->row_col_area) / this->num_of_cols;
    }
    inline int getColCoordinate(int curr) const {
        return (curr - getZCoordinate(curr) * this->row_col_area) % this->num_of_cols;
    }
    inline std::tuple<int, int, int> getCoordinate(int curr) const {
        std::tuple<int, int, int> res = std::make_tuple(
            getRowCoordinate(curr), getColCoordinate(curr), getZCoordinate(curr)
        );
        return res;
    }

    inline void printCoordinate(int curr){
        std::tuple<int, int, int> coodr = getCoordinate(curr);
        std::cout << curr << ":(" << std::get<0>(coodr) << " ," << std::get<1>(coodr) << " ," << std::get<2>(coodr) << ")";
    }

    inline int getManhattanDistance(int loc1, int loc2) const
    {
        int loc1_x = getColCoordinate(loc1);
        int loc1_y = getRowCoordinate(loc1);
        int loc1_z = getZCoordinate(loc1);

        int loc2_x = getColCoordinate(loc2);
        int loc2_y = getRowCoordinate(loc2);
        int loc2_z = getZCoordinate(loc2);

        return abs(loc1_x - loc2_x) + abs(loc1_y - loc2_y) + abs(loc1_z - loc2_z);
    }
    inline int getManhattanDistance(const std::tuple<int, int, int>& loc1, const std::tuple<int, int, int>& loc2) const
    {
        return abs(std::get<0>(loc1) - std::get<0>(loc2)) + 
                abs(std::get<1>(loc1) - std::get<1>(loc2)) +
                abs(std::get<2>(loc1) - std::get<2>(loc2));
    }

    std::list<int> getNeighbors(int curr) const;

    void print(){
        std::cout << "Row: " << num_of_rows << " Cols: " << num_of_cols << " Z: " << num_of_z << " Map Size: " << map_size << std::endl;
    }

};

#endif //MAPF_PIPELINE_INSTANCE_H
