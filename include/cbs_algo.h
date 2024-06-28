//
// Created by admin123456 on 2024/6/3.
//
/*
 * 方法主要集中在: 1.CBS+Astar 2.先形成拓扑结构再膨胀体本身(部分情况下无解) 3.disjoint_CBS+Astar
 * */

#ifndef MAPF_PIPELINE_CBS_ALGO_H
#define MAPF_PIPELINE_CBS_ALGO_H

#include "common.h"
#include "constraint_avoid_table.h"
#include "conflict_utils.h"
#include "group_astar_algo.h"

using namespace std;

class CbsNode {
public:
    CbsNode() {};

    CbsNode(size_t node_id) : node_id(node_id) {}

    ~CbsNode() { reset(); };

    map<size_t, shared_ptr<GroupAstar>> group_res;
    map<size_t, shared_ptr<vector<ObstacleType>>> constrains_map;

    size_t node_id;
    double g_val = 0.0;
    double h_val = 0.0;

    size_t num_expanded = 0;
    size_t num_generated = 0;
    double search_time_cost = -1.0;

    bool update_group_path(size_t group_idx, size_t max_iter = 1000);

    void update_constrains_map(size_t group_idx, const vector<ObstacleType> &group_dynamic_obstacles);

    void update_group_cell(
            size_t group_idx, vector<TaskInfo> group_task_tree,
            Grid::DiscreteGridEnv *group_grid, CollisionDetector *obstacle_detector,
            const vector<ObstacleType> &group_dynamic_obstacles
    );

    void copy_from_node(CbsNode *rhs);

    bool find_inner_conflict_point2point();

    bool find_inner_conflict_segment2segment();

    bool is_conflict_free() { return conflict_list.size() == 0; }

    int get_conflict_size() { return conflict_list.size(); }

    double compute_g_val();

    double compute_h_val();

    void init_conflict_avoid_table(size_t main_group_idx);

    inline double get_f_val() const { return g_val + h_val; }

    PathResult get_group_path(size_t group_idx, string name) { return (*group_res[group_idx]).get_res()[name]; }

    double get_conflict_length(size_t group_idx) { return group_conflict_length_map[group_idx]; }

    vector<ConflictCell> get_conflict_cells() const { return conflict_list; }

    vector<ObstacleType> get_constrain(size_t group_idx) { return *constrains_map[group_idx]; }

    void reset() {
        group_conflict_length_map.clear();
        conflict_list.clear();

        for (auto iter: group_res) {
            iter.second = nullptr;
        }
        group_res.clear();

        for (auto iter: constrains_map) {
            iter.second = nullptr;
        }
        constrains_map.clear();
    }

    struct compare_node {
        bool operator()(const CbsNode *n1, const CbsNode *n2) const {
            if (n1->g_val + n1->h_val == n2->g_val + n2->h_val) {
                if (n1->h_val == n2->h_val)
                    return rand() % 2;
                return n1->h_val >= n2->h_val;
            }
            return n1->g_val + n1->h_val > n2->g_val + n2->h_val;
        }
    };

private:
    map<size_t, double> group_conflict_length_map;
    vector<ConflictCell> conflict_list;
    ConflictAvoidTable avoid_table;
};

class CbsSolver {
public:
    CbsSolver() {}

    ~CbsSolver() { reset(); }

    void push_node(CbsNode *node) {
        open_list.push(node);
    }

    CbsNode *pop_node() {
        CbsNode *node = open_list.top();
        open_list.pop();
        return node;
    }

    bool is_openList_empty() {
        return open_list.empty();
    }

    void reset() {
        // for (auto node: open_list) {
        //     delete node;
        // }
        open_list.clear(); // 释放交由python管理
    }

private:
    boost::heap::pairing_heap<CbsNode *, boost::heap::compare<CbsNode::compare_node>> open_list;
};


#endif //MAPF_PIPELINE_CBS_ALGO_H
