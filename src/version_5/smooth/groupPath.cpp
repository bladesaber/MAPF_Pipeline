#include "groupPath.h"

namespace SmootherNameSpace{

void GroupPath::insertPath(
    size_t pathIdx, Path_XYZR& path_xyzr, 
    std::pair<double, double> startDire, std::pair<double, double> endDire
){
    // forbide too long path
    assert(path_xyzr.size() < 300);

    pathIdxs_set.insert(pathIdx);

    if (nodeMap.size() == 0)
    {
        size_t nodeIdx = nodeMap.size();
        double x, y, z, radius;
        int lastNodeIdx = -1;

        for (size_t i = 0; i < path_xyzr.size(); i++){
            std::tie(x, y, z, radius) = path_xyzr[i];
            max_radius = std::max(max_radius, radius);
            
            GroupPathNode* cur_node = new GroupPathNode(nodeIdx, groupIdx, pathIdx, x, y, z, radius);
            nodeIdx += 1;

            if (i == 0){
                startPathIdxMap[pathIdx] = cur_node->nodeIdx;
                cur_node->alpha = std::get<0>(startDire);
                cur_node->theta = std::get<1>(startDire);

            }else if (i == path_xyzr.size()-1){
                endPathIdxMap[pathIdx] = cur_node->nodeIdx;
                cur_node->alpha = std::get<0>(endDire);
                cur_node->theta = std::get<1>(endDire);
            }

            if (lastNodeIdx >= 0){
                cur_node->parentIdxsMap[pathIdx] = lastNodeIdx;
                nodeMap[lastNodeIdx]->childIdxsMap[pathIdx] = cur_node->nodeIdx;
            }

            nodeMap[cur_node->nodeIdx] = cur_node;
            lastNodeIdx = cur_node->nodeIdx;
        }

    }else{
        nodeTree = new KDTree_XYZRA();
        for (auto iter : nodeMap)
        {
            size_t nodeIdx = iter.first;
            GroupPathNode* node = iter.second;
            nodeTree->insertNode(node->nodeIdx, node->x, node->y, node->z, node->radius, 0.0, 0.0);
        }

        size_t nodeIdx = nodeMap.size();
        double x, y, z, radius;
        KDTree_XYZRA_Res res;
        double dist;

        int lastNodeIdx = -1;
        for (size_t i = 0; i < path_xyzr.size(); i++){
            std::tie(x, y, z, radius) = path_xyzr[i];
            max_radius = std::max(max_radius, radius);

            nodeTree->nearest(x, y, z, res);
            dist = norm2_distance(
                x, y, z, res.x, res.y, res.z
            );
            
            size_t curNode_idx;
            if (dist < std::min(radius, res.data->radius) * 0.5){
                curNode_idx = res.data->idx;
                
                nodeMap[curNode_idx]->x = (x + res.x) / 2.0;
                nodeMap[curNode_idx]->y = (y + res.y) / 2.0;
                nodeMap[curNode_idx]->z = (z + res.z) / 2.0;
                nodeMap[curNode_idx]->radius = std::max(radius, res.data->radius);

            }else{
                GroupPathNode* cur_node = new GroupPathNode(nodeIdx, groupIdx, pathIdx, x, y, z, radius);
                nodeIdx += 1;

                nodeMap[cur_node->nodeIdx] = cur_node;
                curNode_idx = cur_node->nodeIdx;
            }

            if (lastNodeIdx >= 0){
                nodeMap[curNode_idx]->parentIdxsMap[pathIdx] = lastNodeIdx;
                nodeMap[lastNodeIdx]->childIdxsMap[pathIdx] = curNode_idx;
            }

            lastNodeIdx = curNode_idx;

            if (i == 0){
                startPathIdxMap[pathIdx] = curNode_idx;
                nodeMap[curNode_idx]->alpha = std::get<0>(startDire);
                nodeMap[curNode_idx]->theta = std::get<1>(startDire);

            }else if (i == path_xyzr.size()-1){
                endPathIdxMap[pathIdx] = curNode_idx;
                nodeMap[curNode_idx]->alpha = std::get<0>(endDire);
                nodeMap[curNode_idx]->theta = std::get<1>(endDire);
            }

        }
        delete nodeTree;
    }
}

std::vector<size_t> GroupPath::extractPath(size_t pathIdx){
    std::vector<size_t> path;

    size_t cur_nodeIdx = startPathIdxMap[pathIdx];
    GroupPathNode* cur_node = nodeMap[cur_nodeIdx];
    path.emplace_back(cur_nodeIdx);

    while (true)
    {
        auto iter = cur_node->childIdxsMap.find(pathIdx);
        if (iter == cur_node->childIdxsMap.end()){
            return path;
        }

        size_t childIdx = iter->second;

        cur_node = nodeMap[childIdx];
        path.emplace_back(childIdx);
    }
}

void GroupPath::updatePathDirection(){
    for (auto iter : nodeMap){
        GroupPathNode* node = iter.second;

        if (node->parentIdxsMap.size() == 0){
            continue;
        }else if (node->childIdxsMap.size() == 0){
            continue;
        }

        double alpha_sum = 0.0;
        double theta_sum = 0.0;
        double count = node->parentIdxsMap.size();
        for (auto iter : node->parentIdxsMap)
        {
            size_t pathIdx = iter.first;
            GroupPathNode* parent_node = nodeMap[iter.second];
            GroupPathNode* child_node = nodeMap[node->childIdxsMap[pathIdx]];

            double alpha0, theta0;
            std::tie(alpha0, theta0) = vec3D_to_polar3D(
                node->x - parent_node->x, node->y - parent_node->y, node->z - parent_node->z
            );

            double alpha1, theta1;
            std::tie(alpha1, theta1) = vec3D_to_polar3D(
                child_node->x - node->x, child_node->y - node->y, child_node->z - node->z
            );

            alpha_sum += (alpha0 + alpha1) * 0.5;
            theta_sum += (theta0 + theta1) * 0.5;
        }

        node->alpha = alpha_sum / count;
        node->theta = theta_sum / count;
    }
}

void GroupPath::create_pathTree(){
    if (setuped_pathTree){
        delete pathTree;
    }

    pathTree = new KDTree_XYZRA();
    for (auto iter : nodeMap)
    {
        size_t nodeIdx = iter.first;
        GroupPathNode* node = iter.second;
        pathTree->insertNode(node->nodeIdx, node->x, node->y, node->z, node->radius, node->alpha, node->theta);
    }
}

}