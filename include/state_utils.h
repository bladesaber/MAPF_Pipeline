//
// Created by admin123456 on 2024/5/31.
//

#ifndef MAPF_PIPELINE_STATE_UTILS_H
#define MAPF_PIPELINE_STATE_UTILS_H

#include "common.h"
#include "pcl_utils.h"
#include "pcl/point_cloud.h"
#include "math_utils.h"
#include "grid_utils.h"

using namespace std;

namespace Grid {
    class DiscreteStateDetector {
    public:
        DiscreteStateDetector(Grid::DiscreteGridEnv &grid) : grid(grid) {}

        ~DiscreteStateDetector() {}

        bool is_target(size_t loc_flag);

        void insert_start_flags(size_t loc_flag, double direction_x, double direction_y, double direction_z, bool force);

        void insert_target_flags(size_t loc_flag, double direction_x, double direction_y, double direction_z, bool force);

        tuple<double, double, double> get_start_info(size_t loc_flag);

        tuple<double, double, double> get_target_info(size_t loc_flag);

        vector<size_t> get_start_pos_flags();

        vector<size_t> get_target_pos_flags();

        void clear();

        DiscreteGridEnv *get_grid() const { return &grid; }

    protected:
        map<size_t, tuple<double, double, double>> start_pos_map;   // position_flag, direction_x, direction_y, direction_z
        map<size_t, tuple<double, double, double>> target_pos_map;
        DiscreteGridEnv &grid;
    };

    class TreeStateDetector : public DiscreteStateDetector {
    public:
        TreeStateDetector(DiscreteGridEnv &grid, double radius) : DiscreteStateDetector(grid), radius(radius) {}

        void insert_target_flags(size_t loc_flag, double direction_x, double direction_y, double direction_z);

        void create_tree();

        bool is_target(size_t loc0_flag, size_t loc1_flag);

    private:
        double radius;
        PclUtils::KDTree tree;
    };

    class DynamicStepStateDetector : public DiscreteStateDetector {
    public:
        DynamicStepStateDetector(DiscreteGridEnv &grid) : DiscreteStateDetector(grid) {}

        int adjust_scale(size_t loc_flag, int current_scale);

        void update_dynamic_info(double shrink_distance, int scale = 1) {
            shrink_dist = shrink_distance;
            shrink_scale = scale;
        }

    private:
        double shrink_dist;
        int shrink_scale;
    };

}

#endif //MAPF_PIPELINE_STATE_UTILS_H
