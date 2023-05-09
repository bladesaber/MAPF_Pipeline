#ifndef MAPF_PIPELINE_GROUPPATH_H
#define MAPF_PIPELINE_GROUPPATH_H

#include "assert.h"
#include "common.h"
#include "vector3d.h"
#include "utils.h"
#include "kdtreeWrapper.h"

class GroupPathNode
{
public:
    GroupPathNode(){};
    GroupPathNode(size_ut nodeIdx, size_ut pathIdx, double x, double y, double z, double radius):
        nodeIdx(nodeIdx), x(x), y(y), z(z), radius(radius){
            pathIdx_set.insert(pathIdx);
        };
    ~GroupPathNode(){
        release();
    };
    
    size_ut nodeIdx;
    double x, y, z, radius;
    
    std::set<size_ut> pathIdx_set;
    std::map<size_ut, size_ut> parentIdxsMap;
    std::map<size_ut, size_ut> childIdxsMap;

    Vector3D pos_vec;

    void updateGrad(Vector3D& vec){
        x += vec.getX();
        y += vec.getY();
        z += vec.getZ();
    }

private:
    void release(){
        pathIdx_set.clear();
    }
};

class GroupPath
{
public:
    GroupPath(size_ut groupIdx):groupIdx(groupIdx){};
    ~GroupPath(){
        release();
    };

    size_ut groupIdx;
    std::set<size_ut> pathIdxs_set;
    double max_radius = 0.0;

    std::map<size_ut, size_ut> startPathIdxMap; // pathIdx, nodeIdx
    std::map<size_ut, size_ut> endPathIdxMap;   // pathIdx, nodeIdx
    std::map<size_ut, GroupPathNode*> nodeMap;  // nodeIdx, Node

    KDTreeWrapper* nodeTree;

    void insertPath(size_ut pathIdx, DetailPath& detailPath, double radius);
    std::vector<std::tuple<double, double, double, double>> extractPath(size_ut pathIdx);

private:
    void release(){
        for (auto iter : nodeMap)
        {
            delete iter.second;
        }
        nodeMap.clear();
        startPathIdxMap.clear();
        endPathIdxMap.clear();
        pathIdxs_set.clear();
    }

};

#endif