//
// Created by admin123456 on 2024/6/3.
//
/*
 * 方法主要集中在: 1.CBS+Astar 2.先形成拓扑结构再膨胀体本身(部分情况下无解) 3.disjoint_CBS+Astar
 * */

#ifndef MAPF_PIPELINE_CBS_ALGO_H
#define MAPF_PIPELINE_CBS_ALGO_H

#include "common.h"
#include "conflict_utils.h"
#include "group_astar_algo.h"

using namespace std;

class CbsNode {
public:
    CbsNode() {};

    ~CbsNode() { reset(); };

    map<size_t, shared_ptr<GroupAstar>> groupRes;
    map<size_t, shared_ptr<vector<ObstacleType>>> constrainsMap;

    double g_val = 0.0;
    double h_val = 0.0;

    inline double get_f_val() const {
        return g_val + h_val;
    }

    bool update_group_path(size_t group_idx, size_t max_iter = 1000) {
        shared_ptr<GroupAstar> agent = std::make_shared<GroupAstar>(
                groupRes[group_idx]->get_grid(), groupRes[group_idx]->get_obstacle_detector(),
                groupRes[group_idx]->get_task_tree()
        );
        groupRes[group_idx] = nullptr;

        bool is_success = agent->find_path(*constrainsMap[group_idx], max_iter);
        if (!is_success) {
            agent = nullptr;
            return false;
        }

        groupRes[group_idx] = agent;
        return true;
    }

    void update_constrainsMap(size_t group_idx, vector<ObstacleType> group_dynamic_obstacles) {
        constrainsMap[group_idx] = nullptr;
        constrainsMap[group_idx] = std::make_shared<vector<ObstacleType>>(group_dynamic_obstacles);
    }

    void copy_from_node(CbsNode *rhs) {
        for (auto iter: rhs->groupRes) {
            groupRes[iter.first] = shared_ptr<GroupAstar>(iter.second);
        }
        for (auto iter: rhs->constrainsMap) {
            constrainsMap[iter.first] = shared_ptr<vector<ObstacleType>>(iter.second);
        }
    }

    void update_group_cell(
            size_t group_idx, vector<TaskInfo> *group_task_tree,
            Grid::StandardGridEnv *group_grid, CollisionDetector *obstacle_detector,
            vector<ObstacleType> group_dynamic_obstacles
    ) {
        groupRes[group_idx] = nullptr;
        groupRes[group_idx] = std::make_shared<GroupAstar>(group_grid, obstacle_detector, *group_task_tree);

        constrainsMap[group_idx] = nullptr;
        constrainsMap[group_idx] = std::make_shared<vector<ObstacleType>>(group_dynamic_obstacles);
    }

    bool find_inner_conflict();

    bool is_conflict_free() { return conflict_list.size() > 0; }

    double compute_g_val() {
        g_val = 0.0;
        for (auto iter: groupRes) { g_val += iter.second->get_path_length(); }
        return g_val;
    }

    double compute_h_val() {
        h_val = 0.0;
        for (auto iter: group_conflict_length_map) {
            h_val += iter.second;
        }
    }

    void reset() {
        group_conflict_length_map.clear();
        conflict_list.clear();

        for (auto iter: groupRes) {
            iter.second = nullptr;
        }
        groupRes.clear();

        for (auto iter: constrainsMap) {
            iter.second = nullptr;
        }
        constrainsMap.clear();
    }

    struct compare_node {
        bool operator()(const CbsNode *n1, const CbsNode *n2) const {
            if (n1->get_f_val() == n2->get_f_val()) {
                if (n1->h_val == n2->h_val)
                    return rand() % 2;
                return n1->h_val >= n2->h_val;
            }
            return n1->get_f_val() > n2->get_f_val();
        }
    };

private:
    map<size_t, double> group_conflict_length_map;
    vector<ConflictCell> conflict_list;
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
        for (auto node: open_list) {
            delete node;
        }
        open_list.clear();
    }

private:
    boost::heap::pairing_heap<CbsNode *, boost::heap::compare<CbsNode::compare_node>> open_list;
};

#endif //MAPF_PIPELINE_CBS_ALGO_H
