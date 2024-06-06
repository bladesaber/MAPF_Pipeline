//
// Created by admin123456 on 2024/5/30.
//

#ifndef MAPF_PIPELINE_PCL_UTILS_H
#define MAPF_PIPELINE_PCL_UTILS_H

#include "common.h"

#include "pcl/point_cloud.h"
#include "pcl/kdtree/kdtree_flann.h"

using namespace std;
using namespace pcl;

namespace PclUtils {
    class KDTree {
    public:
        vector<int> result_idxs_1D;
        vector<float> result_distance_1D;

        KDTree() { pcd = std::make_shared<PointCloud<PointXYZ>>(); }

        ~KDTree() { pcd = nullptr; }

        void create_tree() {
            tree.setInputCloud(pcd); // 更改point cloud后必须条用该指令重构KDtree
        }

        void update_data(vector<CellXYZ> &data) {
            pcd->clear();
            pcd->width = data.size();
            pcd->height = 1;
            pcd->points.resize(pcd->width * pcd->height);
            for (int i = 0; i < data.size(); ++i) {
                (*pcd)[i].x = get<0>(data[i]);
                (*pcd)[i].y = get<1>(data[i]);
                (*pcd)[i].z = get<2>(data[i]);
            }
        }

        void insert_data(double x, double y, double z) {
            pcd->emplace_back(PointXYZ(x, y, z));
        }

        void insert_data(vector<CellXYZ> &data) {
            for (CellXYZ xyz: data) {
                insert_data(get<0>(xyz), get<1>(xyz), get<2>(xyz));
            }
        }

        void nearestKSearch(double x, double y, double z, int k) {
            search_point.x = x;
            search_point.y = y;
            search_point.z = z;
            result_idxs_1D.clear();
            result_distance_1D.clear();
            tree.nearestKSearch(search_point, k, result_idxs_1D, result_distance_1D);
        }

        void radiusSearch(double x, double y, double z, double radius) {
            search_point.x = x;
            search_point.y = y;
            search_point.z = z;
            result_idxs_1D.clear();
            result_distance_1D.clear();
            tree.radiusSearch(search_point, radius, result_idxs_1D, result_distance_1D);
        }

        PointXYZ get_point_from_data(int idx) {
            return (*pcd)[idx];
        }

        int get_pcd_size() {
            return pcd->size();
        }

    protected:
        PointCloud<PointXYZ>::Ptr pcd;
        KdTreeFLANN <PointXYZ> tree;
        PointXYZ search_point;

    private:
    };

    class XYZRTree : public KDTree {
    public:
        void update_data(vector<CellXYZR> &data) {
            pcd->clear();
            pcd_radius.clear();

            pcd->width = data.size();
            pcd->height = 1;
            pcd->points.resize(pcd->width * pcd->height);
            for (int i = 0; i < data.size(); ++i) {
                double x, y, z, radius;
                tie(x, y, z, radius) = data[i];
                (*pcd)[i].x = x, (*pcd)[i].y = y, (*pcd)[i].z = z;
                pcd_radius.emplace_back(radius);
            }
        }

        void insert_data(double x, double y, double z, double radius) {
            pcd->emplace_back(PointXYZ(x, y, z));
            pcd_radius.emplace_back(radius);
        }

        void insert_data(vector<CellXYZR> &data) {
            for (CellXYZR xyzr: data) {
                insert_data(get<0>(xyzr), get<1>(xyzr), get<2>(xyzr), get<3>(xyzr));
            }
        }

        double get_radius(size_t idx) { return pcd_radius[idx]; }

        double get_max_radius() {
            return *max_element(pcd_radius.begin(), pcd_radius.end());
        }

    private:
        vector<double> pcd_radius;
    };

    class XYZRLTree : public KDTree {
    public:
        void update_data(vector<CellXYZRL> &data) {
            pcd->clear();
            path_radius.clear();
            path_steps_length.clear();

            pcd->width = data.size();
            pcd->height = 1;
            pcd->points.resize(pcd->width * pcd->height);
            for (int i = 0; i < data.size(); ++i) {
                double x, y, z, radius, length;
                tie(x, y, z, radius, length) = data[i];
                (*pcd)[i].x = x, (*pcd)[i].y = y, (*pcd)[i].z = z;
                path_radius.emplace_back(radius);
                path_steps_length.emplace_back(length);
            }
        }

        void insert_data(double x, double y, double z, double radius, double length) {
            pcd->emplace_back(PointXYZ(x, y, z));
            path_radius.emplace_back(radius);
            path_steps_length.emplace_back(length);
        }

        void insert_data(vector<CellXYZRL> &data) {
            for (CellXYZRL xyzrl: data) {
                insert_data(get<0>(xyzrl), get<1>(xyzrl), get<2>(xyzrl), get<3>(xyzrl), get<4>(xyzrl));
            }
        }

        double get_radius(size_t idx) { return path_radius[idx]; }

        double get_step_length(size_t idx) { return path_steps_length[idx]; }

        double get_max_radius() {
            return *max_element(path_radius.begin(), path_radius.end());
        }

        double get_max_step_length() {
            return *max_element(path_steps_length.begin(), path_steps_length.end());
        }

    private:
        vector<double> path_radius;
        vector<double> path_steps_length;
    };


}

#endif //MAPF_PIPELINE_PCL_UTILS_H
