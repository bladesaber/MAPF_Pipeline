#include "groupPath.h"

namespace PathNameSpace{

void GroupPath::insertPath(Path_XYZR& path_xyzr)
{
    // forbide too long path
    assert(path_xyzr.size() < 300);

    if (pathNodeMap.size() == 0){
        size_t nodeIdx = pathNodeMap.size();
        double x, y, z, radius;
        int lastNodeIdx = -1;

        for (size_t i = 0; i < path_xyzr.size(); i++){
            std::tie(x, y, z, radius) = path_xyzr[i];

            PathNode* cur_node = new PathNode(nodeIdx, groupIdx, x, y, z, radius);
            nodeIdx += 1;

            if (lastNodeIdx >= 0){
                cur_node->linkIdxs.insert(lastNodeIdx);
                pathNodeMap[lastNodeIdx]->linkIdxs.insert(cur_node->nodeIdx);
            }

            pathNodeMap[cur_node->nodeIdx] = cur_node;
            lastNodeIdx = cur_node->nodeIdx;
        }
    }else{

        nodeTree = new KDTree_XYZRA();
        for (auto iter : pathNodeMap){
            PathNode* node = iter.second;
            nodeTree->insertNode(node->nodeIdx, node->x, node->y, node->z, node->radius, 0.0, 0.0);
        }

        size_t nodeIdx = pathNodeMap.size();
        double x, y, z, radius;
        KDTree_XYZRA_Res res;
        double dist;

        int lastNodeIdx = -1;
        for (size_t i = 0; i < path_xyzr.size(); i++){
            std::tie(x, y, z, radius) = path_xyzr[i];

            bool need_merge = false;
            nodeTree->nearest(x, y, z, res);
            dist = norm2_distance(x, y, z, res.x, res.y, res.z);

            // if ( dist < std::min(radius, res.data->radius) * 0.5 ){
            //     need_merge = true;
            // }
            if ( dist == 0.0 ){
                need_merge = true;
            }
            
            size_t curNode_idx;
            if (need_merge){
                curNode_idx = res.data->idx;
                // pathNodeMap[curNode_idx]->x = (x + res.x) / 2.0;
                // pathNodeMap[curNode_idx]->y = (y + res.y) / 2.0;
                // pathNodeMap[curNode_idx]->z = (z + res.z) / 2.0;
                // pathNodeMap[curNode_idx]->radius = (radius + res.data->radius) / 2.0;

            }else{
                PathNode* cur_node = new PathNode(nodeIdx, groupIdx, x, y, z, radius);
                nodeIdx += 1;
                pathNodeMap[cur_node->nodeIdx] = cur_node;
                curNode_idx = cur_node->nodeIdx;
            }

            if (lastNodeIdx >= 0){
                pathNodeMap[curNode_idx]->linkIdxs.insert(lastNodeIdx);
                pathNodeMap[lastNodeIdx]->linkIdxs.insert(curNode_idx);
            }
            lastNodeIdx = curNode_idx;
        }
        delete nodeTree;
    }
}

bool GroupPath::deepFirstSearch(size_t nodeIdx, size_t parent_nodeIdx, size_t& goal_nodeIdx, std::vector<size_t>& pathIdxs, int& count){
    // std::cout << "nodeIdx: " << nodeIdx << " parent_nodeIdx:" << parent_nodeIdx << std::endl;

    count += 1;
    if (count > pathNodeMap.size() + 2){
        return false;
    }

    if (nodeIdx == goal_nodeIdx){
        return true;
    }

    for (size_t childIdx: pathNodeMap[nodeIdx]->linkIdxs){
        if ( childIdx == parent_nodeIdx ){
            continue;
        }

        if ( deepFirstSearch(childIdx, nodeIdx, goal_nodeIdx, pathIdxs, count) ){
            pathIdxs.emplace_back(childIdx);
            return true;
        }
    }
    return false;
}

std::vector<size_t> GroupPath::extractPath(size_t start_nodeIdx, size_t goal_nodeIdx){
    std::vector<size_t> pathIdxs;
    int count = 0;

    bool success = deepFirstSearch(start_nodeIdx, 99999999, goal_nodeIdx, pathIdxs, count);
    if (!success){
        return pathIdxs;
    }
    pathIdxs.emplace_back(start_nodeIdx);

    std::reverse(pathIdxs.begin(), pathIdxs.end());
    return pathIdxs;
}

bool GroupPath::insert_OptimizePath(
    size_t pathIdx, 
    double start_x, double start_y, double start_z,
    double end_x, double end_y, double end_z,
    std::pair<double, double> startDire, std::pair<double, double> endDire
){
    std::vector<size_t> pathIdxs = extractPath(
        start_x, start_y, start_z,
        end_x, end_y, end_z
    );
    if (pathIdxs.size()==0){
        return false;
    }

    PathNode* path_node;
    KDTree_XYZRA_Res res;
    std::vector<size_t> graphNodePath;

    size_t nodeIdx = graphNodeMap.size();
    int path_size = pathIdxs.size();
    int clip_num = ceil(pathIdxs.size() * flexible_percentage);
    // std::cout << "[DEBUG] clip_num:" << clip_num << std::endl;
    
    for (size_t i = 0; i < pathIdxs.size(); i++)
    {
        path_node = pathNodeMap[pathIdxs[i]];

        if (i < clip_num || i >= path_size - clip_num)
        {
            FlexGraphNode* node = new FlexGraphNode(nodeIdx, path_node->x, path_node->y, path_node->z, path_node->radius);
            graphNodeMap[node->nodeIdx] = node;
            nodeIdx += 1;

            if ( i==0 ){
                node->fixed = true;
                node->alpha = std::get<0>(startDire);
                node->theta = std::get<1>(startDire);
            }else if (i==pathIdxs.size() - 1){
                node->fixed = true;
                node->alpha = std::get<0>(endDire);
                node->theta = std::get<1>(endDire);
            }

            graphNodePath.emplace_back(node->nodeIdx);
            
        }else{
            bool isNewNode = true;
            if (graphTree->getTreeCount() > 0){
                graphTree->nearest(path_node->x, path_node->y, path_node->z, res);

                // std::cout << "x:" << path_node->x << " y:" << path_node->y << " z:" << path_node->z;
                // std::cout << " res_x:" << res.x << " res_y:" << res.y << " res_z:" << res.z << std::endl;

                if (
                    path_node->x == res.x && 
                    path_node->y == res.y &&
                    path_node->z == res.z
                ){
                    isNewNode = false;
                }

            }

            if (isNewNode){
                FlexGraphNode* node = new FlexGraphNode(nodeIdx, path_node->x, path_node->y, path_node->z, path_node->radius);
                graphNodeMap[node->nodeIdx] = node;
                graphTree->insertNode(node->nodeIdx, node->x, node->y, node->z, node->radius, 0.0, 0.0);
                nodeIdx += 1;

                if ( i==0 ){
                    node->fixed = true;
                    node->alpha = std::get<0>(startDire);
                    node->theta = std::get<1>(startDire);
                }else if (i==pathIdxs.size() - 1){
                    node->fixed = true;
                    node->alpha = std::get<0>(endDire);
                    node->theta = std::get<1>(endDire);
                }

                graphNodePath.emplace_back(node->nodeIdx);
                
            }else{

                graphNodePath.emplace_back(res.data->idx);
            }
        }  
    }

    graphPathMap[pathIdx] = graphNodePath;
}

}