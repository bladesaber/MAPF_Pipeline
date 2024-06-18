//
// Created by admin123456 on 2024/5/30.
//

#ifndef MAPF_PIPELINE_COMMON_H
#define MAPF_PIPELINE_COMMON_H

#include "iostream"
#include <math.h>
#include "numeric"
#include <tuple>
#include <map>
#include "string"
#include <ctime>
#include "set"
#include "list"
#include "vector"
#include "algorithm"
#include <climits>
#include <cfloat>
#include "assert.h"
#include "memory"

#include <boost/heap/pairing_heap.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#include "Eigen/Core"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "pcl/point_cloud.h"

using namespace std;

/*
 * Grid(step_length特征) + obstacle_conflict_detector + dynamic_conflict_detector + A_star_orig/target_manager = 单次(不是单group)A_star_search
 * */

typedef std::tuple<double, double, double> CellXYZ;                    // path_x, path_y, path_z
typedef std::tuple<double, double, double, double> CellXYZR;           // path_x, path_y, path_z, path_radius
typedef std::tuple<double, double, double, double, double> CellXYZRL;  // path_x, path_y, path_z, path_radius, path_step_length
typedef std::tuple<double, double, double, double> ObstacleType;       // obstacle_x, obstacle_y, obstacle_z, obstacle_radius
typedef std::tuple<size_t, int, int, int> Cell_Flag_Orient;            // location_flag, direction_x, direction_y, direction_z

namespace Grid {
    inline vector<tuple<int, int, int>> candidate_1D{
            tuple<int, int, int>(1, 0, 0),
            tuple<int, int, int>(-1, 0, 0),
            tuple<int, int, int>(0, 1, 0),
            tuple<int, int, int>(0, -1, 0),
            tuple<int, int, int>(0, 0, 1),
            tuple<int, int, int>(0, 0, -1),
    };

    inline vector<tuple<int, int, int>> candidate_2D{
            tuple<int, int, int>(1, 1, 0),
            tuple<int, int, int>(1, -1, 0),
            tuple<int, int, int>(-1, 1, 0),
            tuple<int, int, int>(-1, -1, 0),
            tuple<int, int, int>(0, 1, 1),
            tuple<int, int, int>(0, 1, -1),
            tuple<int, int, int>(0, -1, 1),
            tuple<int, int, int>(0, -1, -1),
            tuple<int, int, int>(1, 0, 1),
            tuple<int, int, int>(1, 0, -1),
            tuple<int, int, int>(-1, 0, 1),
            tuple<int, int, int>(-1, 0, -1),
    };

    inline vector<tuple<int, int, int>> candidate_3D{
            tuple<int, int, int>(1, 1, 1),
            tuple<int, int, int>(1, -1, 1),
            tuple<int, int, int>(1, 1, -1),
            tuple<int, int, int>(1, -1, -1),
            tuple<int, int, int>(-1, 1, 1),
            tuple<int, int, int>(-1, -1, 1),
            tuple<int, int, int>(-1, 1, -1),
            tuple<int, int, int>(-1, -1, -1),
    };
}

#endif //MAPF_PIPELINE_COMMON_H
