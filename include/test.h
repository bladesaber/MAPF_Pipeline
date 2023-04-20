//
// Created by quan on 23-3-13.
//

#ifndef MAPF_PIPELINE_TEST_H
#define MAPF_PIPELINE_TEST_H

#include "common.h"

// #include "instance.h"
// #include "cbsNode.h"

#include<pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

namespace py = pybind11;

int testPring(int i, int j) {
    return i + j;
}

void debugPring_vector(std::vector<int>& a){
    for (int i : a)
    {
        std::cout << i << "->";
    }
    std::cout << std::endl;
}

void debugPring_list(std::list<int>& a){
    for (int i : a)
    {
        std::cout << i << "->";
    }
    std::cout << std::endl;
}

void debugPring_map(std::map<std::string, int>& a){
    for (auto it : a)
    {
        std::cout << it.first << " : " << it.second << std::endl;
    }
}

void debugPring_pair(const std::pair<std::string, int>& a){
    std::cout << a.first << " : " << a.second << std::endl;
}

void debugPring_tuple(const std::tuple<std::string, int>& a){
    std::cout << std::get<0>(a) << " : " << std::get<1>(a) << std::endl;
}

void debugPring(){
    int x = 10;
    int y = 5;
    int z = 1;

    std::list<std::tuple<int, int, int>> candidates{
        std::tuple<int, int, int>(y,   x+1, z),
        std::tuple<int, int, int>(y,   x-1, z),
        std::tuple<int, int, int>(y+1, x,   z),
        std::tuple<int, int, int>(y-1, x,   z),
        std::tuple<int, int, int>(y,   x,   z+1),
        std::tuple<int, int, int>(y,   x,   z-1)
    };
    for (auto next : candidates){
        x = std::get<0>(next);
        y = std::get<1>(next);
        z = std::get<2>(next);
        std::cout << x << " | " << y << " | " << z << std::endl;
    }
}

// 针对自定义对象 pybind 会传导指针
// void debugTransformArg_Ownclass(CBSNode* node){
//     node->g_val = 100;
// }
// void debugTransformArg_Ownclass(CBSNode& node){
//     node.g_val = 100;
// }
// 针对转换对象， pybind会将python类型转为C++类型，转换过程必然使指针变更，因为这是个新的类型
// void debugTransformArg(std::vector<int>* a){
//     a->emplace_back(10);
// }

// void debugNumpy(py::array_t<double> xyzs){
//     py::buffer_info buf1 = xyzs.request();
//     std::cout << "xyzs Shape: " << buf1.shape[0] << " and " << buf1.shape[1] << std::endl;
// }

void debug_kdtree(){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width = 5;
    cloud->height = 1;
    cloud->points.resize (5);

    for (std::size_t i = 0; i < cloud->size (); ++i){
        (*cloud)[i].x = (double)i;
        (*cloud)[i].y = (double)i;
        (*cloud)[i].z = (double)i;
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    pcl::PointXYZ searchPoint;
    searchPoint.x = 2.0;
    searchPoint.y = 2.0;
    searchPoint.z = 2.0;

    std::vector<int> pointIdxKNNSearch(1);
    std::vector<float> pointKNNSquaredDistance(1);

    kdtree.nearestKSearch(searchPoint, 1, pointIdxKNNSearch, pointKNNSquaredDistance);
    for (size_t i = 0; i < pointIdxKNNSearch.size (); ++i){
      std::cout << "x:" << (*cloud)[pointIdxKNNSearch[i]].x 
                << " y:" << (*cloud)[pointIdxKNNSearch[i]].y 
                << " z:" << (*cloud)[pointIdxKNNSearch[i]].z 
                << " squared distance: " << pointKNNSquaredDistance[i] << std::endl;
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree2 = pcl::KdTreeFLANN<pcl::PointXYZ>(kdtree);

    pcl::PointCloud<pcl::PointXYZ> cloud2 = pcl::PointCloud<pcl::PointXYZ>(*cloud);
    std::cout << "c1 pointer: " << cloud << " c2 pointer: " << &cloud2;

}

#endif //MAPF_PIPELINE_TEST_H
