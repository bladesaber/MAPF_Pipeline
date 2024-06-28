//
// Created by admin123456 on 2024/6/11.
//

#include "pcl_utils.h"

void PclUtils::KDTree::create_tree() {
    tree.setInputCloud(pcd); // 更改point cloud后必须条用该指令重构KDtree
}

void PclUtils::KDTree::update_data(vector<CellXYZ> &data) {
    pcd->clear();
    pcd->width = data.size();
    pcd->height = 1;
    pcd->points.resize(pcd->width * pcd->height);
    for (int i = 0; i < data.size(); i++) {
        (*pcd)[i].x = get<0>(data[i]);
        (*pcd)[i].y = get<1>(data[i]);
        (*pcd)[i].z = get<2>(data[i]);
    }
}

void PclUtils::KDTree::insert_data(double x, double y, double z) {
    pcd->emplace_back(PointXYZ(x, y, z));
}

void PclUtils::KDTree::insert_data(vector<CellXYZ> &data) {
    for (CellXYZ xyz: data) {
        insert_data(get<0>(xyz), get<1>(xyz), get<2>(xyz));
    }
}

void PclUtils::KDTree::clear_data() {
    pcd->clear();
}

void PclUtils::KDTree::nearestKSearch(double x, double y, double z, int k) {
    search_point.x = x;
    search_point.y = y;
    search_point.z = z;
    result_idxs_1D.clear();
    result_distance_1D.clear();
    tree.nearestKSearch(search_point, k, result_idxs_1D, result_distance_1D);
    for (float &i: result_distance_1D) {
        i = sqrt(i);
    }
}

void PclUtils::KDTree::radiusSearch(double x, double y, double z, double radius) {
    search_point.x = x;
    search_point.y = y;
    search_point.z = z;
    result_idxs_1D.clear();
    result_distance_1D.clear();
    tree.radiusSearch(search_point, radius, result_idxs_1D, result_distance_1D);
    for (float &i: result_distance_1D) {
        i = sqrt(i);
    }
}

PointXYZ PclUtils::KDTree::get_point_from_data(int idx) const {
    return (*pcd)[idx];
}

size_t PclUtils::KDTree::get_pcd_size() const {
    return pcd->size();
}

void PclUtils::XYZRTree::update_data(vector<CellXYZR> &data) {
    pcd->clear();
    pcd_radius.clear();

    pcd->width = data.size();
    pcd->height = 1;
    pcd->points.resize(pcd->width * pcd->height);
    for (int i = 0; i < data.size(); i++) {
        double x, y, z, radius;
        tie(x, y, z, radius) = data[i];
        (*pcd)[i].x = x;
        (*pcd)[i].y = y;
        (*pcd)[i].z = z;
        pcd_radius.emplace_back(radius);
    }
}

void PclUtils::XYZRTree::insert_data(double x, double y, double z, double radius) {
    pcd->emplace_back(PointXYZ(x, y, z));
    pcd_radius.emplace_back(radius);
}

void PclUtils::XYZRTree::insert_data(vector<CellXYZR> &data) {
    for (CellXYZR xyzr: data) {
        insert_data(get<0>(xyzr), get<1>(xyzr), get<2>(xyzr), get<3>(xyzr));
    }
}

double PclUtils::XYZRTree::get_radius(size_t idx) const {
    return pcd_radius[idx];
}

double PclUtils::XYZRTree::get_max_radius() const {
    return *max_element(pcd_radius.begin(), pcd_radius.end());
}

void PclUtils::XYZRTree::clear_data() {
    KDTree::clear_data();
    pcd_radius.clear();
}

void PclUtils::XYZRLTree::update_data(vector<CellXYZRL> &data) {
    pcd->clear();
    path_radius.clear();
    path_steps_length.clear();

    pcd->width = data.size();
    pcd->height = 1;
    pcd->points.resize(pcd->width * pcd->height);
    for (int i = 0; i < data.size(); i++) {
        double x, y, z, radius, length;
        tie(x, y, z, radius, length) = data[i];
        (*pcd)[i].x = x, (*pcd)[i].y = y, (*pcd)[i].z = z;
        path_radius.emplace_back(radius);
        path_steps_length.emplace_back(length);
    }
}

void PclUtils::XYZRLTree::insert_data(double x, double y, double z, double radius, double length) {
    pcd->emplace_back(PointXYZ(x, y, z));
    path_radius.emplace_back(radius);
    path_steps_length.emplace_back(length);
}

void PclUtils::XYZRLTree::insert_data(vector<CellXYZRL> &data) {
    for (CellXYZRL xyzrl: data) {
        insert_data(get<0>(xyzrl), get<1>(xyzrl), get<2>(xyzrl), get<3>(xyzrl), get<4>(xyzrl));
    }
}

double PclUtils::XYZRLTree::get_radius(size_t idx) const {
    return path_radius[idx];
}

double PclUtils::XYZRLTree::get_step_length(size_t idx) const {
    return path_steps_length[idx];
}

double PclUtils::XYZRLTree::get_max_radius() const {
    return *max_element(path_radius.begin(), path_radius.end());
}

double PclUtils::XYZRLTree::get_max_step_length() const {
    return *max_element(path_steps_length.begin(), path_steps_length.end());
}

void PclUtils::XYZRLTree::clear_data() {
    KDTree::clear_data();
    path_radius.clear();
    path_steps_length.clear();
}