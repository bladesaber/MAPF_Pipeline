#ifndef MAPF_PIPELINE_GROUPPATH_H
#define MAPF_PIPELINE_GROUPPATH_H

#include "assert.h"
#include "kdtree_xyzra.h"
#include "utils.h"

// #include "vertex_pose.h"
#include "vertex_XYZ.h"

namespace PathNameSpace{

// x, y, z, radius
typedef std::vector<std::tuple<double, double, double, double>> Path_XYZR;

class GroupPathNode
{
public:
    GroupPathNode(size_t nodeIdx, size_t groupIdx, size_t pathIdx, double x, double y, double z, double radius):
        nodeIdx(nodeIdx), groupIdx(groupIdx), x(x), y(y), z(z), radius(radius){};
    
    ~GroupPathNode(){};
    
    size_t nodeIdx;
    size_t groupIdx;
    double x, y, z, radius;

    double alpha = -999;
    double theta = -999;
    bool fixed = false;
    
    std::map<size_t, size_t> parentIdxsMap;
    std::map<size_t, size_t> childIdxsMap;

    SmootherNameSpace::VertexXYZ* vertex;

    double vertex_x(){
        return vertex->x();
    }

    double vertex_y(){
        return vertex->y();
    }

    double vertex_z(){
        return vertex->z();
    }

    void updateVertex(){
        x = vertex_x();
        y = vertex_y();
        z = vertex_z();
    }

};

class GroupPath
{
public:
    GroupPath(size_t groupIdx):groupIdx(groupIdx){};
    ~GroupPath(){
        release();
    };

    size_t groupIdx;
    std::set<size_t> pathIdxs_set;
    double max_radius = 0.0;

    std::map<size_t, size_t> startPathIdxMap; // pathIdx, nodeIdx
    std::map<size_t, size_t> endPathIdxMap;   // pathIdx, nodeIdx
    std::map<size_t, GroupPathNode*> nodeMap;  // nodeIdx, Node

    KDTree_XYZRA* nodeTree;

    void insertPath(
        size_t pathIdx, Path_XYZR& path_xyzr, 
        bool fixed_start, bool fixed_end,
        std::pair<double, double> startDire, 
        std::pair<double, double> endDire,
        bool merge_path
    );
    std::vector<size_t> extractPath(size_t pathIdx);

    KDTree_XYZRA* pathTree;
    bool setuped_pathTree = false;
    void create_pathTree();

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

        if (setuped_pathTree){
            delete pathTree;
        }
    }
};

}

#endif