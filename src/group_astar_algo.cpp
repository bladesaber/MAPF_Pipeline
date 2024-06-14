//
// Created by admin123456 on 2024/6/12.
//

#include "group_astar_algo.h"

void TaskLeaf::insert(size_t loc_flag, int vec_x, int vec_y, int vec_z, bool force) {
    if (force) {
        (*locs_map)[loc_flag] = make_tuple(vec_x, vec_y, vec_z);
    } else {
        if (locs_map->find(loc_flag) == locs_map->end()) {
            (*locs_map)[loc_flag] = make_tuple(vec_x, vec_y, vec_z);
        }
    }
}

void TaskLeaf::insert(const Cell_Flag_Orient &mark, bool force) {
    size_t loc_flag;
    int vec_x, vec_y, vec_z;
    tie(loc_flag, vec_x, vec_y, vec_z) = mark;
    insert(loc_flag, vec_x, vec_y, vec_z, force);
}

void TaskLeaf::insert(string tag, size_t loc_flag, int vec_x, int vec_y, int vec_z, bool force) {
    member_tags->insert(tag);
    insert(loc_flag, vec_x, vec_y, vec_z, force);
}

void TaskLeaf::insert(string tag, const Cell_Flag_Orient &mark, bool force) {
    member_tags->insert(tag);
    insert(mark, force);
}

void TaskLeaf::merge_leafs(TaskLeaf *rhs0, TaskLeaf *rhs1) {
    for (string tag: *rhs0->member_tags) {
        member_tags->insert(tag);
    }
    for (auto i: *rhs0->locs_map) {
        insert(i.first, get<0>(i.second), get<1>(i.second), get<2>(i.second), false);
    }

    for (string tag: *rhs1->member_tags) {
        member_tags->insert(tag);
    }
    for (auto i: *rhs1->locs_map) {
        insert(i.first, get<0>(i.second), get<1>(i.second), get<2>(i.second), false);
    }
}

void TaskLeaf::merge_path(PathResult &path) {
    for (unsigned long i: path.get_path_flags()) {
        insert(i, 0, 0, 0, false);
    }
}

double GroupAstar::get_path_length() {
    double length = 0.0;
    for (const auto &iter: res_list) {
        length += iter.second.get_length();
    }
    return length;
}

bool GroupAstar::find_path(const vector<ObstacleType> &dynamic_obstacles, size_t max_iter) {
    map<string, TaskLeaf *> leafs_map;
    for (const auto &it: task_tree) {
        if (leafs_map.find(it.begin_tag) == leafs_map.end()) {
            leafs_map[it.begin_tag] = new TaskLeaf();
            for (const auto &mark: it.begin_marks) {
                leafs_map[it.begin_tag]->insert(it.begin_tag, mark, true);
            }
        }
        if (leafs_map.find(it.final_tag) == leafs_map.end()) {
            leafs_map[it.final_tag] = new TaskLeaf();
            for (const auto &mark: it.final_marks) {
                leafs_map[it.final_tag]->insert(it.final_tag, mark, true);
            }
        }
    }

    CollisionDetector dynamic_detector = CollisionDetector();
    for (const auto dynamic_obstacle: dynamic_obstacles) {
        double obs_x, obs_y, obs_z, obs_radius;
        tie(obs_x, obs_y, obs_z, obs_radius) = dynamic_obstacle;
        dynamic_detector.insert_data_point(obs_x, obs_y, obs_z, obs_radius);
    }
    dynamic_detector.create_tree();

    Grid::DynamicStepStateDetector state_detector = Grid::DynamicStepStateDetector(*grid);
    StandardAStarSolver solver = StandardAStarSolver(*grid, *obstacle_detector, dynamic_detector, state_detector);

    bool group_search_success = true;
    for (TaskInfo &iter: task_tree) {
        TaskLeaf *leaf_0 = leafs_map[iter.begin_tag];
        TaskLeaf *leaf_1 = leafs_map[iter.final_tag];

        // ------ update start positions and target positions
        state_detector.clear();
        for (auto &info: leaf_0->get_locs_map()) {
            state_detector.insert_start_flags(
                    info.first, get<0>(info.second), get<1>(info.second), get<2>(info.second), true
            );
        }
        for (auto &info: leaf_1->get_locs_map()) {
            state_detector.insert_target_flags(
                    info.first, get<0>(info.second), get<1>(info.second), get<2>(info.second), true
            );
        }

        // ------ update shrink info
        state_detector.update_dynamic_info(iter.shrink_distance, iter.shrink_scale);

        // ------ update search info
        solver.update_configuration(
                iter.search_radius, iter.step_scale,
                iter.expand_grid_cell, iter.with_theta_star, iter.with_curvature_cost,
                iter.curvature_cost_weight
        );

        // ------ start search
        PathResult sub_path = PathResult(grid);
        bool is_success = solver.find_path(sub_path, max_iter);
        group_search_success = group_search_success && is_success;
        if (!is_success) {
            break;
        }
        res_list[iter.task_name] = sub_path;

        auto *new_leaf = new TaskLeaf();
        new_leaf->merge_leafs(leaf_0, leaf_1);
        new_leaf->merge_path(sub_path);
        delete leafs_map[iter.begin_tag], leafs_map[iter.final_tag];
        for (const string tag: new_leaf->get_members()) {
            leafs_map[tag] = new_leaf;
        }
    }

    for (auto leaf_iter: leafs_map) {
        if (leaf_iter.second != nullptr) {
            for (const string &tag: leaf_iter.second->get_members()) {
                auto member_iter = leafs_map.find(tag);
                if (member_iter->first != leaf_iter.first) {
                    member_iter->second = nullptr; // 防止二次释放
                }
            }
            delete leaf_iter.second;
            leaf_iter.second = nullptr;
        }
    }
    leafs_map.clear();

    if (!group_search_success) {
        reset();
    }

    return group_search_success;
}