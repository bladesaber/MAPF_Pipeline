//
// Created by admin123456 on 2024/6/4.
//

#include "cbs_algo.h"

bool CbsNode::update_group_path(size_t group_idx, size_t max_iter) {
    shared_ptr<GroupAstar> agent = std::make_shared<GroupAstar>(
            group_res[group_idx]->get_grid(), group_res[group_idx]->get_obstacle_detector()
    );
    agent->update_task_tree(group_res[group_idx]->get_task_tree());
    group_res[group_idx] = nullptr;

    bool is_success = agent->find_path(*constrains_map[group_idx], max_iter);
    if (!is_success) {
        agent = nullptr;
        return false;
    }

    group_res[group_idx] = agent;
    return true;
}

void CbsNode::update_constrains_map(size_t group_idx, const vector<ObstacleType>& group_dynamic_obstacles) {
    constrains_map[group_idx] = nullptr;
    constrains_map[group_idx] = std::make_shared<vector<ObstacleType>>(group_dynamic_obstacles);
}

void CbsNode::update_group_cell(
        size_t group_idx, vector<TaskInfo> group_task_tree, Grid::DiscreteGridEnv *group_grid,
        CollisionDetector *obstacle_detector, const vector<ObstacleType>& group_dynamic_obstacles
) {
    group_res[group_idx] = nullptr;
    group_res[group_idx] = std::make_shared<GroupAstar>(group_grid, obstacle_detector);
    group_res[group_idx]->update_task_tree(group_task_tree);

    constrains_map[group_idx] = nullptr;
    constrains_map[group_idx] = std::make_shared<vector<ObstacleType>>(group_dynamic_obstacles);
}

void CbsNode::copy_from_node(CbsNode *rhs) {
    for (const auto& iter: rhs->group_res) {
        group_res[iter.first] = shared_ptr<GroupAstar>(iter.second);
    }
    for (const auto& iter: rhs->constrains_map) {
        constrains_map[iter.first] = shared_ptr<vector<ObstacleType>>(iter.second);
    }
}

double CbsNode::compute_g_val() {
    g_val = 0.0;
    for (const auto& iter: group_res) {
        g_val += iter.second->get_path_length();
    }
    return g_val;
}

double CbsNode::compute_h_val() {
    h_val = 0.0;
    for (auto iter: group_conflict_length_map) {
        h_val += iter.second;
    }
    return h_val;
}

bool CbsNode::find_inner_conflict() {
    // todo 改用线段与线段的最段距离是不是更好呢？但这就难评估障碍点到各出口距离
    conflict_list.clear();

    vector<size_t> group_idxs;
    for (const auto& iter: group_res) {
        group_idxs.emplace_back(iter.first);
    }
    unsigned int num_of_agent = group_idxs.size();

    size_t group_idx;
    GroupAstar agent;
    map<size_t, PclUtils::XYZRTree *> group_trees;
    for (int i = 0; i < num_of_agent; i++) {
        group_idx = group_idxs[i];
        agent = *(group_res[group_idx]);

        vector<CellXYZR> path_set;
        for (const auto& res: agent.get_res()) {
            res.second.get_path_xyzr(path_set);
        }
        auto *tree = new PclUtils::XYZRTree();
        tree->update_data(path_set);
        tree->create_tree();
        group_trees[group_idx] = tree;

        group_conflict_length_map[group_idx] = 0.0; // conflict length init to 0
    }

    PclUtils::XYZRTree *tree0, *tree1;
    size_t group_idx0, group_idx1;
    double radius0, radius1;
    pcl::PointXYZ point0, point1;

    for (int i = 0; i < num_of_agent; i++) {
        group_idx0 = group_idxs[i];
        tree0 = group_trees[group_idx0];

        for (int j = i + 1; j < num_of_agent; j++) {
            group_idx1 = group_idxs[j];
            tree1 = group_trees[group_idx1];

            for (int k = 0; k < tree0->get_pcd_size(); k++) {
                point0 = tree0->get_point_from_data(k);
                radius0 = tree0->get_radius(k);

                tree1->nearestKSearch(point0.x, point0.y, point0.z, 1);
                radius1 = tree1->get_radius(tree1->result_idxs_1D[0]);

                if (tree1->result_distance_1D[0] < radius0 + radius1) {
                    point1 = tree1->get_point_from_data(tree1->result_idxs_1D[0]);
                    ConflictCell conflict_cell = ConflictCell(
                            group_idx0, point1.x, point1.y, point1.z, radius1,
                            group_idx1, point0.x, point0.y, point0.z, radius0
                    );
                    conflict_list.emplace_back(conflict_cell);

                    // ------ method 1: 使用conflict的总长作为启发式
                    // group_conflict_length_map[group_idx0] += tree0->get_step_length(k);
                    // group_conflict_length_map[group_idx1] += tree1->get_step_length(tree1->result_idxs_1D[0]);

                    // ------ method 2: 我认为使用跨越conflict所需总长度作为启发式可能更好
                    group_conflict_length_map[group_idx0] += (radius0 + radius1 - tree1->result_distance_1D[0]);
                    group_conflict_length_map[group_idx1] += (radius0 + radius1 - tree1->result_distance_1D[0]);
                }
            }
        }
    }

    for (auto iter: group_trees) {
        delete iter.second;
    }
    group_trees.clear();

    return is_conflict_free();
}
