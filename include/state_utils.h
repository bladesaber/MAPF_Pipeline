//
// Created by admin123456 on 2024/5/31.
//

#ifndef MAPF_PIPELINE_ALGO_UTILS_H
#define MAPF_PIPELINE_ALGO_UTILS_H

#include "common.h"
#include "pcl_utils.h"
#include "pcl/point_cloud.h"
#include "math_utils.h"
#include "grid_utils.h"

using namespace std;

namespace Grid {

    class ContinueStateDetector {
        /*
         * is_target的判断需要在适当精度的路径信息下进行
         * */
    public:
        ContinueStateDetector() {}

        ~ContinueStateDetector() {}

        void start_coords2loc_flag(ContinueGridEnv &grid, vector<vector<double>> &coords, vector<double> radius) {
            throw invalid_argument("NonImplementation Error");
        }

        void update_targets(vector<vector<double>> &coords, vector<double> radius) {
            target_tree.update_data(coords);
            target_radius.clear();
            target_radius.insert(target_radius.end(), radius.begin(), radius.end());
        }

        void insert_targets(vector<vector<double>> &coords, vector<double> radius) {
            target_tree.insert_data(coords);
            target_radius.insert(target_radius.end(), radius.begin(), radius.end());
        }

        void init_target() {
            target_tree.create_tree();
        }

        bool is_target(double x, double y, double z, double point_radius, vector<double> &target_point) {
            /*
             * 用于判断点是否与目标点相接触
             * */
            target_tree.nearestKSearch(x, y, z, 1);
            double search_radius = max(target_radius[target_tree.result_idxs_1D[0]], point_radius);
            if (target_tree.result_distance_1D[0] < search_radius) {
                target_point[0] = target_tree.get_point_from_data(target_tree.result_idxs_1D[0]).x;
                target_point[1] = target_tree.get_point_from_data(target_tree.result_idxs_1D[0]).y;
                target_point[2] = target_tree.get_point_from_data(target_tree.result_idxs_1D[0]).z;
                return true;
            }
            return false;
        }

        bool is_target(
                double x0, double y0, double z0, double x1, double y1, double z1, double point_radius,
                vector<vector<double>> &connect_points) {
            /*
             * 用于判断路径线是否与目标点相接触
             * */
            double point_x, point_y, point_z, dist, compare_radius;
            double connect_x, connect_y, connect_z;
            vector<vector<double>> xyz_list{vector<double>{x0, y0, z0}, vector<double>{x1, y1, z1}};

            for (vector<double> xyz: xyz_list) {
                target_tree.nearestKSearch(xyz[0], xyz[1], xyz[2], 1);
                point_x = target_tree.get_point_from_data(target_tree.result_idxs_1D[0]).x;
                point_y = target_tree.get_point_from_data(target_tree.result_idxs_1D[0]).y;
                point_z = target_tree.get_point_from_data(target_tree.result_idxs_1D[0]).z;
                compare_radius = max(target_radius[target_tree.result_idxs_1D[0]], point_radius);
                dist = point2line_dist(point_x, point_y, point_z, x0, y0, z0, x1, y1, z1);
                if (dist < compare_radius) {
                    if (is_point2line_projection_inner(point_x, point_y, point_z, x0, y0, z0, x1, y1, z1)) {
                        tie(connect_x, connect_y, connect_z) = point2line_projection_xyz(
                                point_x, point_y, point_z, x0, y0, z0, x1, y1, z1
                        );
                        connect_points.emplace_back(vector<double>{connect_x, connect_y, connect_z});
                    }
                    connect_points.emplace_back(vector<double>{point_x, point_y, point_z});
                    return true;
                }
            }
            return false;
        }

    private:
        vector<vector<double>> start_coords;
        PclUtils::KDTree target_tree;
        vector<double> target_radius;
    };

    class StandardStateDetector {
    public:
        StandardStateDetector() {}

        ~StandardStateDetector() {}

        bool is_target(size_t loc_flag) {
            // todo 不充足，没考虑不同步长的跨越错配问题

            auto it = target_pos_map.find(loc_flag);
            if(it != target_pos_map.end()){
                return true;
            }
            return false;
        }

        void remove_start_flags(vector<size_t> flags) {
            for (size_t loc: flags) {
                auto it = start_pos_map.find(loc);
                if(it != start_pos_map.end()){
                    start_pos_map.erase(it);
                }
            }
        }

        void remove_target_flags(vector<size_t> flags) {
            for (size_t loc: flags) {
                auto it = target_pos_map.find(loc);
                if(it != target_pos_map.end()){
                    target_pos_map.erase(it);
                }
            }
        }

        void insert_target_flags(size_t loc_flag, int vec_x, int vec_y, int vec_z) {
            target_pos_map[loc_flag] = make_tuple(vec_x, vec_y, vec_z);
        }

        void insert_start_flags(size_t loc_flag, int vec_x, int vec_y, int vec_z) {
            start_pos_map[loc_flag] = make_tuple(vec_x, vec_y, vec_z);
        }

        tuple<int, int, int> get_start_info(size_t loc_flag) {
            auto it = start_pos_map.find(loc_flag);
            if(it == start_pos_map.end()){
                assert("[ERROR]: location isn't in state detector.");
            }
            return it->second;
        }

        tuple<int, int, int> get_target_info(size_t loc_flag) {
            auto it = target_pos_map.find(loc_flag);
            if(it == target_pos_map.end()){
                assert("[ERROR]: location isn't in state detector.");
            }
            return it->second;
        }

        vector<size_t> get_start_pos_flags(){
            vector<size_t> flags;
            for (auto it = start_pos_map.begin(); it != start_pos_map.end(); ++it){
                flags.emplace_back(it->first);
            }
            return flags;
        }

        vector<size_t> get_target_pos_flags(){
            vector<size_t> flags;
            for (auto it = target_pos_map.begin(); it != target_pos_map.end(); ++it){
                flags.emplace_back(it->first);
            }
            return flags;
        }

        void clear_start_pos_map(){
            start_pos_map.clear();
        }

        void clear_target_pos_map(){
            target_pos_map.clear();
        }

    private:
        map<size_t, tuple<int, int, int>> start_pos_map;
        map<size_t, tuple<int, int, int>> target_pos_map;
    };

}
#endif //MAPF_PIPELINE_ALGO_UTILS_H
