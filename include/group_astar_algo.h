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
    TaskLeaf() {
        member_tags = new set<string>();
        locs_map = new map<size_t, tuple<int, int, int>>();
    };

    ~TaskLeaf() {
        locs_map->clear();
        delete locs_map;
        delete member_tags;
    };

    void merge_leafs(TaskLeaf *rhs0, TaskLeaf *rhs1);

    void merge_path(PathResult &path);

    void insert(string tag, size_t loc_flag, int vec_x, int vec_y, int vec_z, bool force);

    void insert(string tag, const Cell_Flag_Orient &mark, bool force);

    void insert(size_t loc_flag, int vec_x, int vec_y, int vec_z, bool force);

    void insert(const Cell_Flag_Orient &mark, bool force);

    map<size_t, tuple<int, int, int>> &get_locs_map() const {return *locs_map;}

    set<string> &get_members() const {return *member_tags;}

private:
    set<string> *member_tags;
    map<size_t, tuple<int, int, int>> *locs_map;
};

class TaskInfo {
public:
    string task_name;
    string begin_tag, final_tag;
    vector<Cell_Flag_Orient> begin_marks, final_marks;
    double search_radius, shrink_distance;
    int step_scale, shrink_scale;
    vector<tuple<int, int, int>> expand_grid_cell;
    bool with_theta_star, with_curvature_cost;
    double curvature_cost_weight;

    TaskInfo(
            string task_name, string begin_tag, string final_tag,
            vector<Cell_Flag_Orient> begin_marks, vector<Cell_Flag_Orient> final_marks,
            double search_radius, int step_scale, double shrink_distance, int shrink_scale,
            vector<tuple<int, int, int>> expand_grid_cell, bool with_theta_star, bool with_curvature_cost,
            double curvature_cost_weight
    ) : task_name(task_name), begin_tag(begin_tag), final_tag(final_tag),
        begin_marks(begin_marks), final_marks(final_marks),
        search_radius(search_radius), step_scale(step_scale),
        shrink_distance(shrink_distance), shrink_scale(shrink_scale), expand_grid_cell(expand_grid_cell),
        with_theta_star(with_theta_star), with_curvature_cost(with_curvature_cost),
        curvature_cost_weight(curvature_cost_weight) {};

    ~TaskInfo() {};
};

class GroupAstar {
public:
    GroupAstar() {}

    GroupAstar(Grid::DiscreteGridEnv *grid, CollisionDetector *obstacle_detector) :
            grid(grid), obstacle_detector(obstacle_detector) {};

    ~GroupAstar() { reset(); };

    bool find_path(const vector<ObstacleType> &dynamic_obstacles, size_t max_iter);

    void update_task_tree(vector<TaskInfo> task_list) { task_tree = task_list; }

    void reset() { res_list.clear(); }

    Grid::DiscreteGridEnv *get_grid() const { return grid; }

    CollisionDetector *get_obstacle_detector() const { return obstacle_detector; }

    vector<TaskInfo> get_task_tree() const { return task_tree; };

    map<string, PathResult> get_res() const { return res_list; }

    double get_path_length();

private:
    Grid::DiscreteGridEnv *grid;
    CollisionDetector *obstacle_detector;
    vector<TaskInfo> task_tree;
    map<string, PathResult> res_list;
};


#endif //MAPF_PIPELINE_GROUP_ASTAR_ALGO_H
