#include "groupPath.h"

void GroupPath::insertPath(size_ut pathIdx, DetailPath& detailPath, double radius){
    // forbide too long path

    assert(detailPath.size()<300);
    size_ut max_node_num = 99999;

    max_radius = std::max(max_radius, radius);
    pathIdxs_set.insert(pathIdx);

    if (nodeMap.size() > 0)
    {
        nodeTree = new KDTreeWrapper();
        for (auto iter : nodeMap)
        {
            size_ut nodeIdx = iter.first;
            GroupPathNode* node = iter.second;
            nodeTree->insertPathNode(node->nodeIdx, node->x, node->y, node->z, node->radius);
        }

        double x, y, z, length, distance;
        KDTreeRes res;
        int nodeIdx = nodeMap.size();
        size_ut lastNodeIdx = max_node_num;
        size_ut cur_nodeIdx;
        
        for (size_t i = 0; i < detailPath.size(); i++)
        {
            std::tie(x, y, z, length) = detailPath[i];
        
            nodeTree->nearest(x, y, z, res);
            distance = norm2_distance(
                x, y, z,
                res.x, res.y, res.z
            );
            if (distance < std::min(radius, res.data->radius) * 0.5)
            {
                size_ut cloestIdx = res.data->dataIdx;
        
                nodeMap[cloestIdx]->x = (x + res.x) / 2.0;
                nodeMap[cloestIdx]->y = (y + res.y) / 2.0;
                nodeMap[cloestIdx]->z = (z + res.z) / 2.0;
                nodeMap[cloestIdx]->radius = std::max(radius, res.data->radius);
                nodeMap[cloestIdx]->pathIdx_set.insert(pathIdx);
        
                if (lastNodeIdx < max_node_num){
                    nodeMap[cloestIdx]->parentIdxsMap[pathIdx] = lastNodeIdx;
                    nodeMap[lastNodeIdx]->childIdxsMap[pathIdx] = cloestIdx;
                }
        
                lastNodeIdx = cloestIdx;
                cur_nodeIdx = cloestIdx;
        
            }else{
                GroupPathNode* cur_node = new GroupPathNode(nodeIdx, pathIdx, x, y, z, radius);
                nodeMap[cur_node->nodeIdx] = cur_node;
        
                if (lastNodeIdx < max_node_num){
                    cur_node->parentIdxsMap[pathIdx] = lastNodeIdx;
                    nodeMap[lastNodeIdx]->childIdxsMap[pathIdx] = cur_node->nodeIdx;
                }
        
                nodeIdx += 1;
                lastNodeIdx = cur_node->nodeIdx;
                cur_nodeIdx = cur_node->nodeIdx;
            }
        
            if (i == 0)
            {
                startPathIdxMap[pathIdx] = cur_nodeIdx;
            }else if (i == detailPath.size()-1)
            {
                endPathIdxMap[pathIdx] = cur_nodeIdx;
            }

            if (i <= 1 || i >= detailPath.size() - 2){
                fixedNodes.insert(cur_nodeIdx);
            }
        
        }

        delete nodeTree;

    }else{
        int nodeIdx = nodeMap.size();
        double x, y, z, length;
        size_ut lastNodeIdx = max_node_num;

        for (size_t i = 0; i < detailPath.size(); i++){
            std::tie(x, y, z, length) = detailPath[i];
            GroupPathNode* cur_node = new GroupPathNode(nodeIdx, pathIdx, x, y, z, radius);

            if (i == 0)
            {
                startPathIdxMap[pathIdx] = cur_node->nodeIdx;
            }else if (i == detailPath.size()-1)
            {
                endPathIdxMap[pathIdx] = cur_node->nodeIdx;
            }

            if (i <= 1 || i >= detailPath.size() - 2){
                fixedNodes.insert(cur_node->nodeIdx);
            }
            
            if (lastNodeIdx < max_node_num){
                cur_node->parentIdxsMap[pathIdx] = lastNodeIdx;
                nodeMap[lastNodeIdx]->childIdxsMap[pathIdx] = cur_node->nodeIdx;
            }

            nodeMap[cur_node->nodeIdx] = cur_node;
            nodeIdx += 1;
            lastNodeIdx = cur_node->nodeIdx;
        }
    }
}

std::vector<std::tuple<double, double, double, double>> GroupPath::extractPath(size_ut pathIdx){
    std::vector<std::tuple<double, double, double, double>> path;

    size_ut cur_nodeIdx = startPathIdxMap[pathIdx];
    GroupPathNode* cur_node = nodeMap[cur_nodeIdx];
    path.emplace_back(std::make_tuple(cur_node->x, cur_node->y, cur_node->z, cur_node->radius));

    while (true)
    {
        auto iter = cur_node->childIdxsMap.find(pathIdx);
        if (iter == cur_node->childIdxsMap.end()){
            // there must be error
            std::cout << "The path is broken" << std::endl;
            assert(false);
            break;
        }
        size_ut childIdx = iter->second;

        cur_node = nodeMap[childIdx];
        path.emplace_back(std::make_tuple(cur_node->x, cur_node->y, cur_node->z, cur_node->radius));
        cur_nodeIdx = cur_node->nodeIdx;

        if (cur_nodeIdx == endPathIdxMap[pathIdx])
        {
            cur_node = nodeMap[cur_nodeIdx];
            path.emplace_back(std::make_tuple(cur_node->x, cur_node->y, cur_node->z, cur_node->radius));
            break;
        }
    }

    return path;
}
