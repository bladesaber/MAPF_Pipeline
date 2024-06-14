//
// Created by admin123456 on 2024/6/11.
//

#include "collision_utils.h"

void CollisionDetector::update_data(vector<CellXYZR> &data) {
    tree.update_data(data);
}

void CollisionDetector::create_tree() {
    if (tree.get_pcd_size() > 0) {
        tree.create_tree();
    }
}

void CollisionDetector::insert_data_point(double x, double y, double z, double radius) {
    tree.insert_data(x, y, z, radius);
}

bool CollisionDetector::is_valid(double x, double y, double z, double point_radius) {
    if (tree.get_pcd_size() == 0) { return true; }

    double max_radius = tree.get_max_radius();
    tree.radiusSearch(x, y, z, max_radius + point_radius);
    if (tree.result_idxs_1D.empty()) { return true; }

    for (int i = 0; i < tree.result_idxs_1D.size(); i++) {
        if (tree.result_distance_1D[i] < tree.get_radius(tree.result_idxs_1D[i]) + point_radius) {
            return false;
        }
    }

    return true;
}

bool CollisionDetector::is_valid(
        double x0, double y0, double z0, double x1, double y1, double z1, double point_radius
) {
    if (tree.get_pcd_size() == 0) { return true; }

    double max_radius = tree.get_max_radius();
    double line_dist = norm2_dist(x0, y0, z0, x1, y1, z1);
    double request_dist = max_radius + point_radius;
    double search_dist = sqrt(pow(line_dist * 0.5, 2.0) + pow(request_dist, 2.0));
    double point2line_val;

    tree.radiusSearch(x0, y0, z0, search_dist);
    for (int idx: tree.result_idxs_1D) {
        PointXYZ point = tree.get_point_from_data(idx);
        point2line_val = point2line_dist(point.x, point.y, point.z, x0, y0, z0, x1, y1, z1);
        if (point2line_val < tree.get_radius(idx) + point_radius) {
            return false;
        }
    }
    tree.radiusSearch(x1, y1, z1, search_dist);
    for (int idx: tree.result_idxs_1D) {
        PointXYZ point = tree.get_point_from_data(idx);
        point2line_val = point2line_dist(point.x, point.y, point.z, x0, y0, z0, x1, y1, z1);
        if (point2line_val < tree.get_radius(idx) + point_radius) {
            return false;
        }
    }
    return true;
}

bool CollisionDetector::is_line_on_sight(
        double x0, double y0, double z0, double x1, double y1, double z1, double point_radius
) {
    // 检测两点间是否有障碍物
    return is_valid(x0, y0, z0, x1, y1, z1, point_radius);
}