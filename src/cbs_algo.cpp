//
// Created by admin123456 on 2024/6/4.
//

#include "cbs_algo.h"

bool CbsNode::find_inner_conflict() {
    conflict_list.clear();

    vector<size_t> group_idxs;
    for (auto iter: groupRes) { group_idxs.emplace_back(iter.first); }
    unsigned int num_of_agent = group_idxs.size();

    size_t group_idx;
    GroupAstar agent;
    map<size_t, PclUtils::XYZRLTree *> group_trees;
    for (int i = 0; i < num_of_agent; i++) {
        group_idx = group_idxs[i];
        agent = *groupRes[group_idx];
        vector<CellXYZRL> path_set;

        for (auto res: agent.get_res()) {
            res.second.get_path_xyzrl(*agent.get_grid(), path_set);
        }
        PclUtils::XYZRLTree *tree = new PclUtils::XYZRLTree();
        tree->update_data(path_set);
        group_trees[group_idx] = tree;
    }

    for (int i = 0; i < num_of_agent; i++){
        group_conflict_length_map[group_idxs[i]] = 0.0;
    }

    PclUtils::XYZRLTree *tree0, *tree1;
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

    return is_conflict_free();
}
