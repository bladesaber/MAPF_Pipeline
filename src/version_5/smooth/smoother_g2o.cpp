#include "smoother_g2o.h"

namespace SmootherNameSpace{

bool SmootherG2O::add_vertexs(){
    unsigned int id_counter = 0;
    for (auto group_iter : groupMap)
    {
        GroupPath* groupPath = group_iter.second;
        
        std::set<size_t> fixedset;
        for (auto i : groupPath->startPathIdxMap){
            fixedset.insert(i.second);
        }
        for (auto i : groupPath->endPathIdxMap){
            fixedset.insert(i.second);
        }

        for (auto node_iter : groupPath->nodeMap)
        {
            GroupPathNode* node = node_iter.second;
            
            if (fixedset.find(node->nodeIdx) != fixedset.end()){
                node->vertex = new VertexSE3(node->x, node->y, node->z, node->alpha, node->theta, true);
            }
            else{
                node->vertex = new VertexSE3(node->x, node->y, node->z, node->alpha, node->theta, false);
            }
            
            node->vertex->setId(id_counter);
            bool success = optimizer->addVertex(node->vertex);
            if (!success){
                return false;
            }
            
            id_counter += 1;
        }
    }

    return true;
}

void SmootherG2O::build_graph(
    double elasticBand_weight,
    double crossPlane_weight, double curvature_weight,
    double obstacle_weight, double pipeConflict_weight
){
    if (!optimizer->edges().empty() || !optimizer->vertices().empty())
    {
        std::cout << "[Error]: Cannot build graph, because it is not empty. Call graphClear()!" << std::endl;
        assert(false);
    }

    bool status;
    status = add_vertexs();
    if (!status){
        std::cout << "[Error]: Adding Vertexs Fail" << std::endl;
        return;
    }
    
    if (elasticBand_weight>0){
        status = add_elasticBand(elasticBand_weight);
        if (!status){
            std::cout << "[Error]: Adding Elastic Band Edge Fail" << std::endl;
            return;
        }
    }

    if ( crossPlane_weight>0 && curvature_weight>0){
        status = add_kinematicEdge(crossPlane_weight, curvature_weight);
        if (!status){
            std::cout << "[Error]: Adding Kinematics Edge Fail" << std::endl;
            return;
        }
    }

    if (obstacle_weight>0){
        add_obstacleEdge(obstacle_weight);
    }

    if (pipeConflict_weight>0){
        add_pipeConflictEdge(pipeConflict_weight);
    }
}

bool SmootherG2O::add_elasticBand(double weight){
    for (auto group_iter : groupMap)
    {
        GroupPath* groupPath = group_iter.second;

        std::set<size_t> explored_set;
        int nodeSize = groupPath->nodeMap.size();

        assert(nodeSize < 1000);

        for (size_t pathIdx : groupPath->pathIdxs_set)
        {
            std::vector<size_t> nodeIdxs_path = groupPath->extractPath(pathIdx);

            for (size_t i = 0; i < nodeIdxs_path.size() - 1; i++)
            {
                GroupPathNode* node0 = groupPath->nodeMap[nodeIdxs_path[i]];
                GroupPathNode* node1 = groupPath->nodeMap[nodeIdxs_path[i + 1]];

                size_t tag = (std::min(node0->nodeIdx, node1->nodeIdx) + 1.0) * nodeSize + std::max(node0->nodeIdx, node1->nodeIdx);

                // std::cout << "MinIdx:" << std::min(node0->nodeIdx, node1->nodeIdx);
                // std::cout << " MaxIdx:" << std::max(node0->nodeIdx, node1->nodeIdx);
                // std::cout << " Size:" << nodeSize << " Tag:" << tag << std::endl;

                if (explored_set.find(tag) != explored_set.end()){
                    continue;
                }
                explored_set.insert(tag);
                
                EdgeElasticBand* edge = new EdgeElasticBand();
                edge->setVertex(0, node0->vertex);
                edge->setVertex(1, node1->vertex);
                
                Eigen::Matrix<double,1,1> information;
                information.fill(weight);
                edge->setInformation(information);

                bool success = optimizer->addEdge(edge);
                if (!success){
                   return false;
                }
            }
        }
    }
    return true;
}

void SmootherG2O::insertStaticObs(double x, double y, double z, double radius, double alpha, double theta){
    obsTree->insertNode(0, x, y, z, radius, alpha, theta);
}

void SmootherG2O::add_obstacleEdge(double weight){
    if (obsTree->getTreeCount() == 0){
        return;
    }
    
    std::vector<KDTree_XYZRA_Res*> resList;

    for (auto group_iter : groupMap)
    {
        GroupPath* group = group_iter.second;

        for (auto node_iter : group->nodeMap)
        {
            GroupPathNode* node = node_iter.second;

            if ((node->parentIdxsMap.size() == 0) || (node->childIdxsMap.size() == 0)){
                continue;
            }
            
            resList.clear();
            obsTree->nearest_range(node->x, node->y, node->z, node->radius + 0.1, resList);

            for (KDTree_XYZRA_Res* res : resList)
            {
                EdgeObstacle* edge = new EdgeObstacle(
                    Eigen::Vector3d(res->x, res->y, res->z), 
                    res->data->radius + node->radius, 
                    0.1
                );
                edge->setVertex(0, node->vertex);

                Eigen::Matrix<double,1,1> information;
                information.fill(weight);
                edge->setInformation(information);

                optimizer->addEdge(edge);

                delete res;
            }
        }
    }
}

void SmootherG2O::add_pipeConflictEdge(double weight){
    if (groupMap.size() <= 1){
        return;
    }

    std::vector<KDTree_XYZRA_Res*> resList;

    for (size_t i=0; i < groupMap.size(); i++)
    {
        GroupPath* group_i = groupMap[i];

        for (size_t j=i+1; j < groupMap.size(); j++){
            GroupPath* group_j = groupMap[j];

            for (auto iter : group_i->nodeMap)
            {
                GroupPathNode* node = iter.second;

                resList.clear();
                group_j->pathTree->nearest_range(node->x, node->y, node->z, 
                    group_i->max_radius + group_j->max_radius, resList
                );

                for (KDTree_XYZRA_Res* res : resList)
                {
                    double dist = norm2_distance(
                        node->x, node->y, node->z,
                        res->x, res->y, res->z
                    );
                    if (dist <= (node->radius + res->data->radius) * 1.2 )
                    {
                        EdgePipeConflict* edge = new EdgePipeConflict(node->radius + res->data->radius, 0.1);
                        edge->setVertex(0, node->vertex);
                        edge->setVertex(1, (group_j->nodeMap[res->data->idx])->vertex);

                        Eigen::Matrix<double,1,1> information;
                        information.fill(weight);
                        edge->setInformation(information);

                        optimizer->addEdge(edge);
                    }

                    delete res;
                }
            }
        }
    }
}

bool SmootherG2O::add_kinematicEdge(
    double crossPlane_weight, double curvature_weight
){
    for (auto group_iter : groupMap)
    {
        GroupPath* groupPath = group_iter.second;

        std::set<size_t> explored_set;
        int nodeSize = groupPath->nodeMap.size();

        assert(nodeSize < 1000);

        for (size_t pathIdx : groupPath->pathIdxs_set)
        {
            std::vector<size_t> nodeIdxs_path = groupPath->extractPath(pathIdx);

            for (size_t i = 0; i < nodeIdxs_path.size() - 1; i++)
            {
                GroupPathNode* node0 = groupPath->nodeMap[nodeIdxs_path[i]];
                GroupPathNode* node1 = groupPath->nodeMap[nodeIdxs_path[i + 1]];

                size_t tag = (std::min(node0->nodeIdx, node1->nodeIdx) + 1.0) * nodeSize + std::max(node0->nodeIdx, node1->nodeIdx);
                if (explored_set.find(tag) != explored_set.end()){
                    continue;
                }
                explored_set.insert(tag);

                double radius = std::max(node0->radius, node1->radius);
                EdgeKinematics* edge = new EdgeKinematics(1.0 / (3.0 * radius));
                edge->setVertex(0, node0->vertex);
                edge->setVertex(1, node1->vertex);

                Eigen::Matrix<double,2,2> information;
                information.fill(0.0);
                information(0, 0) = crossPlane_weight;
                information(1, 1) = curvature_weight;
                edge->setInformation(information);

                bool success = optimizer->addEdge(edge);
                if (!success)
                {
                    return false;
                }   
            }
        }
    }
    return true;
}

}
