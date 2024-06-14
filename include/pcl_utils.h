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
        vector<float> result_distance_1D; // pcl kdtree的计算距离是欧拉距离的平方

        KDTree() { pcd = std::make_shared<PointCloud<PointXYZ>>(); }

        ~KDTree() { pcd = nullptr; }

        void create_tree();

        void update_data(vector<CellXYZ> &data);

        void insert_data(double x, double y, double z);

        void insert_data(vector<CellXYZ> &data);

        void clear_data();

        void nearestKSearch(double x, double y, double z, int k);

        void radiusSearch(double x, double y, double z, double radius);

        PointXYZ get_point_from_data(int idx) const;

        size_t get_pcd_size() const;

    protected:
        PointCloud<PointXYZ>::Ptr pcd;
        KdTreeFLANN<PointXYZ> tree;
        PointXYZ search_point;
    };

    class XYZRTree : public KDTree {
    public:
        void update_data(vector<CellXYZR> &data);

        void insert_data(double x, double y, double z, double radius);

        void insert_data(vector<CellXYZR> &data);

        double get_radius(size_t idx) const;

        double get_max_radius() const;

        void clear_data();

    private:
        vector<double> pcd_radius;
    };

    class XYZRLTree : public KDTree {
    public:
        void update_data(vector<CellXYZRL> &data);

        void insert_data(double x, double y, double z, double radius, double length);

        void insert_data(vector<CellXYZRL> &data);

        double get_radius(size_t idx) const;

        double get_step_length(size_t idx) const;

        double get_max_radius() const;

        double get_max_step_length() const;

        void clear_data();

    private:
        vector<double> path_radius;
        vector<double> path_steps_length;
    };

}

#endif //MAPF_PIPELINE_PCL_UTILS_H
