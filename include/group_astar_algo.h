//
// Created by admin123456 on 2024/6/3.
//

#ifndef MAPF_PIPELINE_GROUP_ASTAR_ALGO_H
#define MAPF_PIPELINE_GROUP_ASTAR_ALGO_H

#include "common.h"
#include "conflict_utils.h"
#include "astar_algo.h"

using namespace std;

class TaskLeaf {
public:
    TaskLeaf() {};

    TaskLeaf(size_t loc, int vec_x, int vec_y, int vec_z) {
        locs_map[loc] = make_tuple(vec_x, vec_y, vec_z);
    };

    ~TaskLeaf() {};

    void merge_leafs(TaskLeaf *rhs0, TaskLeaf *rhs1) {
        for (size_t loc: rhs0->members) {
            members.insert(loc);
        }
        for (auto i = rhs0->locs_map.begin(); i != rhs0->locs_map.end(); ++i) {
            locs_map[i->first] = i->second;
        }

        for (size_t loc: rhs1->members) {
            members.insert(loc);
        }
        for (auto i = rhs1->locs_map.begin(); i != rhs1->locs_map.end(); ++i) {
            locs_map[i->first] = i->second;
        }
    }

    void merge_path(PathResult &path) {
        vector<size_t> path_list = path.get_path_flags();
        for (int i = 0; i < path_list.size(); ++i) {
            update_loc_info(path_list[i], 0, 0, 0);
        }
    }

    void update_loc_info(size_t loc, int vec_x, int vec_y, int vec_z) {
        locs_map[loc] = make_tuple(vec_x, vec_y, vec_z);
    }

    map<size_t, tuple<int, int, int>> &get_locs_map() {
        return locs_map;
    }

    set<size_t> &get_members() {
        return members;
    }

private:
    set<size_t> members;
    map<size_t, tuple<int, int, int>> locs_map;
};

class TaskInfo {
public:
    string name;
    size_t begin_loc, final_loc;
    int vec_x0, vec_y0, vec_z0, vec_x1, vec_y1, vec_z1;
    double search_radius;
    int x_step_scale, y_step_scale, z_step_scale;
    list<string> expand_methods;
    bool with_theta_star, with_curvature_cost;

    TaskInfo(
            string name,
            size_t begin_loc, int vec_x0, int vec_y0, int vec_z0,
            size_t final_loc, int vec_x1, int vec_y1, int vec_z1,
            double search_radius, double thickness,
            int x_step_scale, int y_step_scale, int z_step_scale,
            list<string> expand_methods, bool with_theta_star, bool with_curvature_cost
    ) : name(name),
        begin_loc(begin_loc), vec_x0(vec_x0), vec_y0(vec_y0), vec_z0(vec_z0),
        final_loc(final_loc), vec_x1(vec_x1), vec_y1(vec_y1), vec_z1(vec_z1),
        search_radius(search_radius),
        x_step_scale(x_step_scale), y_step_scale(y_step_scale), z_step_scale(z_step_scale),
        expand_methods(expand_methods), with_theta_star(with_theta_star), with_curvature_cost(with_curvature_cost) {};

    ~TaskInfo() {};
};

class GroupAstar {
public:
    GroupAstar() {}

    GroupAstar(Grid::StandardGridEnv *grid, CollisionDetector *obstacle_detector, vector<TaskInfo> &task_tree) :
            grid(grid), obstacle_detector(obstacle_detector), task_tree(task_tree) {};

    ~GroupAstar() { reset(); };

    bool find_path(vector<ObstacleType> dynamic_obstacles, size_t max_iter) {
        map<size_t, TaskLeaf *> leafs_map;
        for (TaskInfo it: task_tree) {
            if (leafs_map.find(it.begin_loc) == leafs_map.end()) {
                leafs_map[it.begin_loc] = new TaskLeaf(it.begin_loc, it.vec_x0, it.vec_y0, it.vec_z0);
            }
            if (leafs_map.find(it.final_loc) == leafs_map.end()) {
                leafs_map[it.final_loc] = new TaskLeaf(it.final_loc, it.vec_x1, it.vec_y1, it.vec_z1);
            }
        }

        Grid::StandardStateDetector state_detector = Grid::StandardStateDetector();
        CollisionDetector dynamic_detector = CollisionDetector();
        for (int i = 0; i < dynamic_obstacles.size(); ++i) {
            double obs_x, obs_y, obs_z, obs_radius;
            tie(obs_x, obs_y, obs_z, obs_radius) = dynamic_obstacles[i];
            dynamic_detector.insert_data_point(obs_x, obs_y, obs_z, obs_radius);
        }

        StandardAStarSolver solver = StandardAStarSolver(*grid, *obstacle_detector, dynamic_detector, state_detector);

        bool group_search_success = false;
        for (TaskInfo iter: task_tree) {
            TaskLeaf *leaf_0 = leafs_map[iter.begin_loc];
            TaskLeaf *leaf_1 = leafs_map[iter.final_loc];

            state_detector.clear_start_pos_map();
            state_detector.clear_target_pos_map();
            for (auto info: leaf_0->get_locs_map()) {
                state_detector.insert_start_flags(
                        info.first, get<0>(info.second), get<1>(info.second), get<2>(info.second)
                );
            }
            for (auto info: leaf_1->get_locs_map()) {
                state_detector.insert_target_flags(
                        info.first, get<0>(info.second), get<1>(info.second), get<2>(info.second)
                );
            }

            PathResult sub_path;
            solver.update_configuration(
                    iter.search_radius, iter.x_step_scale, iter.y_step_scale, iter.z_step_scale,
                    iter.expand_methods, iter.with_theta_star, iter.with_curvature_cost
            );
            bool is_success = solver.find_path(sub_path, max_iter);
            group_search_success = group_search_success && is_success;
            if (!is_success) {
                break;
            }

            res_list[iter.name] = sub_path;

            TaskLeaf *new_leaf = new TaskLeaf();
            new_leaf->merge_leafs(leaf_0, leaf_1);
            new_leaf->merge_path(sub_path);
            delete leafs_map[iter.begin_loc], leafs_map[iter.final_loc];
            for (size_t loc: new_leaf->get_members()) {
                leafs_map[loc] = new_leaf;
            }
        }

        for (auto leaf_iter: leafs_map) {
            if (leaf_iter.second != nullptr) {
                delete leaf_iter.second; // 防止二次释放
            }
        }
        leafs_map.clear();

        if (!group_search_success) {
            reset();
        }

        return group_search_success;
    }

    void reset() {
        res_list.clear();
    }

    Grid::StandardGridEnv *get_grid() { return grid; }

    CollisionDetector *get_obstacle_detector() { return obstacle_detector; }

    vector<TaskInfo> &get_task_tree() { return task_tree; };

    map<string, PathResult> &get_res() { return res_list; }

    double get_path_length() {
        double length = 0.0;
        for (auto iter: res_list) { length += iter.second.get_path_length(); }
        return length;
    }

private:
    Grid::StandardGridEnv *grid;
    CollisionDetector *obstacle_detector;
    vector<TaskInfo> task_tree;

    map<string, PathResult> res_list;
};

#endif //MAPF_PIPELINE_GROUP_ASTAR_ALGO_H
