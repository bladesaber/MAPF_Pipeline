//
// Created by admin123456 on 2024/6/11.
//

#include "state_utils.h"

bool Grid::DiscreteStateDetector::is_target(size_t loc_flag) {
    auto it = target_pos_map.find(loc_flag);
    if (it != target_pos_map.end()) { return true; }
    return false;
}

void Grid::DiscreteStateDetector::insert_start_flags(
        size_t loc_flag, double direction_x, double direction_y, double direction_z, bool force
) {
    if (force) {
        start_pos_map[loc_flag] = make_tuple(direction_x, direction_y, direction_z);
    } else {
        if (start_pos_map.find(loc_flag) == start_pos_map.end()) {
            start_pos_map[loc_flag] = make_tuple(direction_x, direction_y, direction_z);
        }
    }
}

void Grid::DiscreteStateDetector::insert_target_flags(
        size_t loc_flag, double direction_x, double direction_y, double direction_z, bool force
) {
    if (force) {
        target_pos_map[loc_flag] = make_tuple(direction_x, direction_y, direction_z);
    } else {
        if (target_pos_map.find(loc_flag) == target_pos_map.end()) {
            target_pos_map[loc_flag] = make_tuple(direction_x, direction_y, direction_z);
        }
    }
}

tuple<double, double, double> Grid::DiscreteStateDetector::get_start_info(size_t loc_flag) {
    auto it = start_pos_map.find(loc_flag);
    if (it == start_pos_map.end()) {
        assert("[ERROR]: location isn't in state detector.");
    }
    return it->second;
}

tuple<double, double, double> Grid::DiscreteStateDetector::get_target_info(size_t loc_flag) {
    auto it = target_pos_map.find(loc_flag);
    if (it == target_pos_map.end()) {
        assert("[ERROR]: location isn't in state detector.");
    }
    return it->second;
}

vector<size_t> Grid::DiscreteStateDetector::get_start_pos_flags() {
    vector<size_t> flags;
    for (auto &it: start_pos_map) { flags.emplace_back(it.first); }
    return flags;
}

vector<size_t> Grid::DiscreteStateDetector::get_target_pos_flags() {
    vector<size_t> flags;
    for (auto &it: target_pos_map) { flags.emplace_back(it.first); }
    return flags;
}

void Grid::DiscreteStateDetector::clear() {
    start_pos_map.clear();
    target_pos_map.clear();
}

void Grid::TreeStateDetector::insert_target_flags(
        size_t loc_flag, double direction_x, double direction_y, double direction_z
) {
    DiscreteStateDetector::insert_target_flags(loc_flag, direction_x, direction_y, direction_z, true);

    double x, y, z;
    tie(x, y, z) = grid.flag2xyz(loc_flag);
    tree.insert_data(x, y, z);
}

void Grid::TreeStateDetector::create_tree() { tree.create_tree(); }

bool Grid::TreeStateDetector::is_target(size_t loc0_flag, size_t loc1_flag) {
    double x0, y0, z0, x1, y1, z1;
    tie(x0, y0, z0) = grid.flag2xyz(loc0_flag);
    tie(x1, y1, z1) = grid.flag2xyz(loc1_flag);

    double dist = norm2_dist(x0, y0, z0, x1, y1, z1);
    double search_dist = sqrt(pow(dist, 2.0) + pow(radius, 2.0));
    double point2line_val;

    tree.radiusSearch(x1, y1, z1, search_dist);
    for (int idx: tree.result_idxs_1D) {
        PointXYZ point = tree.get_point_from_data(idx);
        point2line_val = point2line_dist(point.x, point.y, point.z, x0, y0, z0, x1, y1, z1);
        if (point2line_val < radius) {
            return true;
        }
    }

    return false;
}

int Grid::DynamicStepStateDetector::adjust_scale(size_t loc_flag, int current_scale) {
    bool in_shrink_range = false;
    double x0, y0, z0, x1, y1, z1, distance;
    tie(x0, y0, z0) = grid.flag2xyz(loc_flag);

    for (auto it: target_pos_map) {
        tie(x1, y1, z1) = grid.flag2xyz(it.first);
        distance = norm2_dist(x0, y0, z0, x1, y1, z1);
        if (distance <= shrink_dist) {
            in_shrink_range = true;
            break;
        }
    }

    if (in_shrink_range) {
        return shrink_scale;
    }
    return current_scale;
}