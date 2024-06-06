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

#include <boost/heap/pairing_heap.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "pcl/point_cloud.h"

/*
 * Grid(step_length特征) + obstacle_conflict_detector + dynamic_conflict_detector + A_star_orig/target_manager = 单次(不是单group)A_star_search
 * */

typedef std::tuple<double, double, double> CellXYZ;                    // path_x, path_y, path_z
typedef std::tuple<double, double, double, double> CellXYZR;           // path_x, path_y, path_z, path_radius
typedef std::tuple<double, double, double, double, double> CellXYZRL;  // path_x, path_y, path_z, path_radius, path_step_length
typedef std::tuple<double, double, double, double> ObstacleType;       // obstacle_x, obstacle_y, obstacle_z, obstacle_radius

#endif //MAPF_PIPELINE_COMMON_H
