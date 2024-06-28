//
// Created by admin123456 on 2024/5/31.
//
/*
 * 搜索半径包含厚度
 * */

#ifndef MAPF_PIPELINE_COLLISION_UTILS_H
#define MAPF_PIPELINE_COLLISION_UTILS_H

#include "common.h"
#include "math_utils.h"
#include "pcl_utils.h"

using namespace std;

class CollisionDetector {
public:
    CollisionDetector() {}

    ~CollisionDetector() {}

    void update_data(vector<CellXYZR> &data);

    void create_tree();

    void insert_data_point(double x, double y, double z, double radius);

    bool is_valid(double x, double y, double z, double point_radius);

    bool is_valid(double x0, double y0, double z0, double x1, double y1, double z1, double point_radius);

    bool is_line_on_sight(double x0, double y0, double z0, double x1, double y1, double z1, double point_radius);

    size_t get_size() { return tree.get_pcd_size(); }

    void print_info();

private:
    PclUtils::XYZRTree tree;
};

#endif //MAPF_PIPELINE_COLLISION_UTILS_H