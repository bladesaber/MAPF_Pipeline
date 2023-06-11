#ifndef MAPF_PIPELINE_GROUPPATH_H
#define MAPF_PIPELINE_GROUPPATH_H

#include "assert.h"
#include "math.h"

#include "kdtree_xyzra.h"
#include "utils.h"

#include "vertex_XYZ.h"

namespace PathNameSpace{

// x, y, z, radius
typedef std::vector<std::tuple<double, double, double, double>> Path_XYZR;

class FlexGraphNode
{
public:
    FlexGraphNode(size_t nodeIdx, double x, double y, double z, double radius):
        nodeIdx(nodeIdx), x(x), y(y), z(z), radius(radius){};
    ~FlexGraphNode(){};
    
    double x, y, z, radius;
    double alpha = -999;
    double theta = -999;

    size_t nodeIdx;
    bool fixed = false;
    // bool isIndependt;

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

class PathNode{
public:
    PathNode(size_t nodeIdx, size_t groupIdx, double x, double y, double z, double radius):
        nodeIdx(nodeIdx), groupIdx(groupIdx), x(x), y(y), z(z), radius(radius){};
    ~PathNode(){};

    size_t nodeIdx;
    size_t groupIdx;
    double x, y, z, radius;
    std::set<size_t> linkIdxs;
};

class GroupPath
{
public:
    GroupPath(size_t groupIdx):groupIdx(groupIdx){
        graphTree = new KDTree_XYZRA();
    };
    ~GroupPath(){
        release();
    };

    size_t groupIdx;
    double max_radius = 0.0;

    // std::vector<size_t> terminalIdxs;
    std::map<size_t, PathNode*> pathNodeMap;  // nodeIdx, Node

    // Just For Tempotary
    KDTree_XYZRA* nodeTree;

    void insertPath(Path_XYZR& path_xyzr);
    void setMaxRadius(double radius){
        max_radius = radius;
    }

    bool deepFirstSearch(size_t nodeIdx, size_t parent_nodeIdx, size_t& goal_nodeIdx, std::vector<size_t>& pathIdxs, int& count);
    std::vector<size_t> extractPath(size_t start_nodeIdx, size_t goal_nodeIdx);
    std::vector<size_t> extractPath(
        double start_x, double start_y, double start_z,
        double end_x, double end_y, double end_z
    ){
        std::vector<size_t> pathIdxs;

        int start_nodeIdx = findNodeIdx(start_x, start_y, start_z);
        if (start_nodeIdx<0){
            std::cout << "[INFO]: Can't Find StartPoint." << std::endl;
            return pathIdxs;
        }

        int goal_nodeIdx = findNodeIdx(end_x, end_y, end_z);
        if (goal_nodeIdx<0){
            std::cout << "[INFO]: Can't Find EndPoint." << std::endl;
            return pathIdxs;
        }

        return extractPath(start_nodeIdx, goal_nodeIdx);
    }

    int findNodeIdx(double x, double y, double z){
        for (auto iter: pathNodeMap){
            PathNode* node = iter.second;
            if (
                node->x == x &&
                node->y == y &&
                node->z == z
            ){
                return node->nodeIdx;
            }
        }
        return -1; 
    };

    KDTree_XYZRA* graphTree;
    std::map<size_t, FlexGraphNode*> graphNodeMap; // nodeIdx, GraphNode
    std::map<size_t, std::vector<size_t>> graphPathMap; // pathIdx, graphNodeIdxs

    bool insert_OptimizePath(
        size_t pathIdx, 
        double start_x, double start_y, double start_z,
        double end_x, double end_y, double end_z,
        std::pair<double, double> startDire, 
        std::pair<double, double> endDire,
        double startFlexRatio, double endFlexRatio
    );

private:
    void release(){
        for (auto iter : pathNodeMap)
        {
            delete iter.second;
        }
        pathNodeMap.clear();
        
        for (auto iter : graphNodeMap)
        {
            delete iter.second;
        }
        graphNodeMap.clear();

        delete graphTree;
    }
};

}

#endif