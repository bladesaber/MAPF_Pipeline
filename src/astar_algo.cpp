//
// Created by admin123456 on 2024/5/31.
//

#include "astar_algo.h"

void PathResult::update_result(const AStarNode *goal_node, double search_radius) {
    radius = search_radius;
    path_flags.clear();
    path_step_lengths.clear();
    path_length = 0.0;

    int step = 0;
    double last_x, last_y, last_z, x, y, z, step_length;
    auto curr = goal_node;
    while (curr != nullptr) {
        if (step == 0) {
            tie(last_x, last_y, last_z) = grid->flag2xyz(curr->loc_flag);

        } else {
            tie(x, y, z) = grid->flag2xyz(curr->loc_flag);
            step_length = norm2_dist(last_x, last_y, last_z, x, y, z);
            path_length += step_length;
            last_x = x, last_y = y, last_z = z;
            path_step_lengths.emplace_back(step_length);
        }
        step += 1;

        path_flags.emplace_back(curr->loc_flag);
        curr = curr->parent;
    }

    reverse(path_flags.begin(), path_flags.end());
    reverse(path_step_lengths.begin(), path_step_lengths.end());
    // path_step_lengths.insert(path_step_lengths.begin(), mean(path_step_lengths)); // 由于第1个节点无步长，因此估算1个
}

double StandardAStarSolver::compute_h_cost(size_t loc_flag) {
    auto min_val = DBL_MAX;
    for (size_t loc: state_detector.get_target_pos_flags()) {
        min_val = min(min_val, grid.get_manhattan_cost(loc, loc_flag));
    }
    return min_val;
}

double StandardAStarSolver::compute_move_orient(AStarNode *parent, size_t next_loc_flag) {
    int parent_x, parent_y, parent_z, node_x, node_y, node_z;
    tie(parent_x, parent_y, parent_z) = grid.flag2grid(parent->loc_flag);
    tie(node_x, node_y, node_z) = grid.flag2grid(next_loc_flag);
    int vec_x = node_x - parent_x;
    int vec_y = node_y - parent_y;
    int vec_z = node_z - parent_z;
    return line2line_cos(vec_x, vec_y, vec_z, parent->straight_x, parent->straight_y, parent->straight_z);
}

tuple<int, int, int, double> StandardAStarSolver::compute_move_info(AStarNode *parent_node, size_t loc_flag) {
    int parent_x, parent_y, parent_z, node_x, node_y, node_z;
    tie(parent_x, parent_y, parent_z) = grid.flag2grid(parent_node->loc_flag);
    tie(node_x, node_y, node_z) = grid.flag2grid(loc_flag);

    double cost = grid.get_euler_cost(parent_node->loc_flag, loc_flag);

    int straight_x, straight_y, straight_z;
    int orig_x = parent_x - parent_node->straight_x;
    int orig_y = parent_y - parent_node->straight_y;
    int orig_z = parent_z - parent_node->straight_z;

    // ------ 先计算路径折角
    int vec_x = node_x - parent_x;
    int vec_y = node_y - parent_y;
    int vec_z = node_z - parent_z;

    double cos_val = line2line_cos(
            parent_node->straight_x, parent_node->straight_y, parent_node->straight_z,
            vec_x, vec_y, vec_z
    );
    if (cos_val > 1.0 - 1e-2) {
        straight_x = parent_node->straight_x + vec_x;
        straight_y = parent_node->straight_y + vec_y;
        straight_z = parent_node->straight_z + vec_z;
    } else {
        straight_x = vec_x;
        straight_y = vec_y;
        straight_z = vec_z;

        if (with_curvature_cost) {
            cost += grid.get_curvature_cost(
                    orig_x, orig_y, orig_z,
                    parent_x, parent_y, parent_z,
                    node_x, node_y, node_z,
                    radius * 3.0, grid.unit_length * current_scale * curvature_weight, true
            );
        }
    }

    // ------ 如果是终点，则计算终端折角与目标折角的cost
    if (state_detector.is_target(loc_flag)) {
        if (with_curvature_cost) {
            if (cos_val > 1.0 - 1e-2) {
                cost += grid.get_curvature_cost(
                        orig_x, orig_y, orig_z,
                        node_x, node_y, node_z,
                        node_x + straight_x, node_y + straight_y, node_z + straight_z,
                        radius * 3.0, grid.unit_length * current_scale * curvature_weight, false
                );
            } else {
                cost += grid.get_curvature_cost(
                        parent_x, parent_y, parent_z,
                        node_x, node_y, node_z,
                        node_x + straight_x, node_y + straight_y, node_z + straight_z,
                        radius * 3.0, grid.unit_length * current_scale * curvature_weight, false
                );
            }
        }
    }

    return make_tuple(straight_x, straight_y, straight_z, cost);
}

AStarNode *StandardAStarSolver::get_next_node(size_t neighbour_loc_flag, AStarNode *parent_node) {
    int straight_x, straight_y, straight_z;
    double moving_cost;
    tie(straight_x, straight_y, straight_z, moving_cost) = compute_move_info(parent_node, neighbour_loc_flag);

    double h_val = compute_h_cost(neighbour_loc_flag);
    double g_val = parent_node->g_val + moving_cost;
    auto *next_node = new AStarNode(
            neighbour_loc_flag,
            g_val,
            h_val,
            parent_node,
            parent_node->timestep + 1,
            0,
            false,
            straight_x, straight_y, straight_z
    );

    if (with_theta_star) {
        AStarNode *elder_node = parent_node->parent;
        if (elder_node != nullptr) {
            double pre_x, pre_y, pre_z, next_x, next_y, next_z;
            tie(pre_x, pre_y, pre_z) = grid.flag2xyz(elder_node->loc_flag);
            tie(next_x, next_y, next_z) = grid.flag2xyz(neighbour_loc_flag);

            bool obstacle_free = obstacle_detector.is_line_on_sight(
                    pre_x, pre_y, pre_z, next_x, next_y, next_z, radius
            );
            bool dynamic_free = dynamic_detector.is_line_on_sight(
                    pre_x, pre_y, pre_z, next_x, next_y, next_z, radius
            );

            if (obstacle_free && dynamic_free) {
                tie(straight_x, straight_y, straight_z, moving_cost) = compute_move_info(
                        elder_node, neighbour_loc_flag
                );
                double new_g_val = elder_node->g_val + moving_cost;
                if (new_g_val <= next_node->g_val) {
                    next_node->g_val = new_g_val;
                    next_node->parent = elder_node;
                    next_node->straight_x = straight_x;
                    next_node->straight_y = straight_y;
                    next_node->straight_z = straight_z;
                }
            }
        }
    }

    return next_node;
}

bool StandardAStarSolver::find_path(PathResult &res_path, size_t max_iter) {
    num_generated = 0;
    num_expanded = 0;
    size_t run_times = 0;

    for (size_t loc_flag: state_detector.get_start_pos_flags()) {
        int orient_x, orient_y, orient_z;
        tie(orient_x, orient_y, orient_z) = state_detector.get_start_info(loc_flag);

        auto *start_node = new AStarNode(
                loc_flag, 0, compute_h_cost(loc_flag),
                orient_x, orient_y, orient_z
        );
        pushNode(start_node);
        all_nodes_table.insert(start_node);
    }

    double cur_x, cur_y, cur_z;
    double neg_x, neg_y, neg_z;

    bool is_success = false;
    auto start_time = clock();

    while (!open_list.empty()) {
        AStarNode *current_node = popNode();
        tie(cur_x, cur_y, cur_z) = grid.flag2xyz(current_node->loc_flag);

        if (state_detector.is_target(current_node->loc_flag)) {
            res_path.update_result(current_node, radius);
            is_success = true;
            break;
        }

        num_expanded++;
        current_scale = state_detector.adjust_scale(current_node->loc_flag, step_scale);
        for (size_t neighbour_loc_flag: grid.get_valid_neighbors(
                current_node->loc_flag, current_scale, expand_candidates
        )) {
            if (compute_move_orient(current_node, neighbour_loc_flag) < -1e-2) {
                continue; // skip if moving angle larger than 90 degrees
            }
            tie(neg_x, neg_y, neg_z) = grid.flag2xyz(neighbour_loc_flag);
            if (!obstacle_detector.is_valid(cur_x, cur_y, cur_z, neg_x, neg_y, neg_z, radius)) {
                continue;
            }
            if (!dynamic_detector.is_valid(cur_x, cur_y, cur_z, neg_x, neg_y, neg_z, radius)) {
                continue;
            }

            AStarNode *next_node = get_next_node(neighbour_loc_flag, current_node);

            auto it = all_nodes_table.find(next_node);
            if (it == all_nodes_table.end()) {
                pushNode(next_node);
                all_nodes_table.insert(next_node);
                num_generated++;
                continue;
            }

            AStarNode *existing_next = *it;
            if (next_node->get_f_val() < existing_next->get_f_val()) {
                existing_next->copy(*next_node);
                if (!existing_next->in_openlist) {
                    pushNode(existing_next); // if it's in the closed list (reopen)
                } else {
                    open_list.increase(existing_next->open_handle);
                }
            }

            delete next_node;
        }

        run_times += 1;
        if (run_times >= max_iter) {
            break;
        }
    }

    search_time_cost = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    reset();

    return is_success;
}