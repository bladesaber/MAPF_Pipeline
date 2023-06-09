#include "smootherXYZ_g2o.h"

namespace SmootherNameSpace{

bool SmootherXYZG2O::add_vertexs(){
    size_t vertex_num = 0;

    for (auto group_iter : groupMap)
    {
        GroupPath* groupPath = group_iter.second;
        
        for (auto node_iter : groupPath->graphNodeMap)
        {
            FlexGraphNode* node = node_iter.second;
            
            node->vertex = new VertexXYZ(node->x, node->y, node->z, node->fixed);
            node->vertex->setId(vertex_num);
            vertex_num += 1;

            bool success = optimizer->addVertex(node->vertex);
            if (!success){
                return false;
            }            
        }
    }
    return true;
}

bool SmootherXYZG2O::build_graph(
    double elasticBand_weight,
    double kinematic_weight,
    double obstacle_weight,
    double pipeConflict_weight,
    double boundary_weight
){
    if (!optimizer->edges().empty() || !optimizer->vertices().empty())
    {
        std::cout << "[Error]: Cannot build graph, because it is not empty. Call graphClear()!" << std::endl;
        return false;
    }

    bool status;
    status = add_vertexs();
    if (!status){
        std::cout << "[Error]: Adding Vertexs Fail" << std::endl;
        return false;
    }

    if (boundary_weight>0){
        status = add_boundaryEdge(boundary_weight);
        if (!status){
            std::cout << "[Error]: Adding Boundaey Band Edge Fail" << std::endl;
            return false;
        }
    }
    
    if (elasticBand_weight>0){
        status = add_elasticBand(elasticBand_weight);
        if (!status){
            std::cout << "[Error]: Adding Elastic Band Edge Fail" << std::endl;
            return false;
        }
    }

    if (kinematic_weight>0){
        status = add_kinematicEdge(kinematic_weight);
        if (!status){
            std::cout << "[Error]: Adding Kinematics Edge Fail" << std::endl;
            return false;
        }
    }

    if (obstacle_weight>0)
    {
        status = add_obstacleEdge(obstacle_weight);
        if (!status){
            std::cout << "[Error]: Adding Obstacle Edge Fail" << std::endl;
            return false;
        }
    }

    if (pipeConflict_weight>0){
        status = add_pipeConflictEdge(pipeConflict_weight);
        if (!status){
            std::cout << "[Error]: Adding Obstacle Edge Fail" << std::endl;
            return false;
        }
    }

    return true;
}

/*
void SmootherXYZG2O::loss_info(
    double elasticBand_weight, 
    double kinematic_weight,
    double obstacle_weight,
    double pipeConflict_weight
){
    // GroupPathNode* node0;
    // GroupPathNode* node1;
    // GroupPathNode* node2;

    // for (auto group_iter : groupMap)
    // {
    //     GroupPath* groupPath = group_iter.second;

    //     for (size_t pathIdx : groupPath->pathIdxs_set)
    //     {
    //         std::vector<size_t> nodeIdxs_path = groupPath->extractPath(pathIdx);

    //         std::cout << "GroupIdx:" << groupPath->groupIdx << " PathIdx:" << pathIdx << std::endl;
    //         for (size_t i = 0; i < nodeIdxs_path.size(); i++)
    //         {
    //             std::cout << "[DEBUG] Idx:" << i << std::endl;
    //             double loss;

    //             if (kinematic_weight>0)
    //             {
    //                 if ( i>0 && i<nodeIdxs_path.size() - 1){

    //                     node0 = groupPath->nodeMap[nodeIdxs_path[i - 1]];
    //                     node1 = groupPath->nodeMap[nodeIdxs_path[i]];
    //                     node2 = groupPath->nodeMap[nodeIdxs_path[i + 1]];

    //                     if (i == 1)
    //                     {
    //                         double dz = std::sin(node0->theta);
    //                         double dx = std::cos(node0->theta) * std::cos(node0->alpha);
    //                         double dy = std::cos(node0->theta) * std::sin(node0->alpha);
    //                         std::cout << "  orientation ( dx:" << dx << " dy:" << dy << " dz:" << dz << ")" << std::endl;

    //                         Eigen::Vector3d orientation = Eigen::Vector3d(dx, dy, dz);
    //                         loss = EdgeXYZ_VertexKinematics::lost_calc(node0->vertex, node1->vertex, orientation, true);
    //                         std::cout << "  Kinematic Vertex Loss:" << loss << " Infomation:" << kinematic_weight << std::endl;

    //                     }else if ( i == nodeIdxs_path.size() - 2){
    //                         double dz = std::sin(node2->theta);
    //                         double dx = std::cos(node2->theta) * std::cos(node2->alpha);
    //                         double dy = std::cos(node2->theta) * std::sin(node2->alpha);
    //                         std::cout << "  orientation ( dx:" << dx << " dy:" << dy << " dz:" << dz << ")" << std::endl;

    //                         Eigen::Vector3d orientation = Eigen::Vector3d(dx, dy, dz);
    //                         loss = EdgeXYZ_VertexKinematics::lost_calc(node1->vertex, node2->vertex, orientation, true);
    //                         std::cout << "  Kinematic Vertex Loss:" << loss  << " Infomation:" << kinematic_weight << std::endl;

    //                     }

    //                     loss = EdgeXYZ_Kinematics::lost_calc(node0->vertex, node1->vertex, node2->vertex, true);
    //                     std::cout << "  Kinematic Edge Loss:" << loss  << " Infomation:" << kinematic_weight << std::endl;
    //                 }
    //             }

    //             if (elasticBand_weight>0)
    //             {
    //                 if ( i<nodeIdxs_path.size() - 1){
    //                     node0 = groupPath->nodeMap[nodeIdxs_path[i]];
    //                     node1 = groupPath->nodeMap[nodeIdxs_path[i + 1]];
    //                     loss = EdgeXYZ_ElasticBand::lost_calc(node0->vertex, node1->vertex);
    //                     std::cout << "  ElasticBand Edge Loss:" << loss  << " Infomation:" << elasticBand_weight << std::endl;
    //                 }
    //             }

    //             if (obstacle_weight>0)
    //             {
    //                 node0 = groupPath->nodeMap[nodeIdxs_path[i]];

    //                 if ( 
    //                     ((node0->parentIdxsMap.size() != 0) || (node0->childIdxsMap.size() != 0)) && 
    //                     (obsTree->getTreeCount() > 0) 
    //                 ){
    //                     std::vector<KDTree_XYZRA_Res*> resList;
    //                     obsTree->nearest_range(node0->x, node0->y, node0->z, node0->radius * 1.5, resList);
    //                     for (KDTree_XYZRA_Res* res : resList){
    //                         loss = EdgeXYZ_Obstacle::lost_calc(
    //                             node0->vertex, Eigen::Vector3d(res->x, res->y, res->y), node0->radius 
    //                         );
    //                         std::cout << "  Obstacle Edge Loss:" << loss;
    //                         std::cout << " x:" << res->x << " y:" << res->y << " z:" << res->z << " Infomation:" << obstacle_weight << std::endl;
    //                         delete res;
    //                     }
    //                 }
    //             }

    //         }
    //     }
    // }
}
*/

bool SmootherXYZG2O::add_elasticBand(double elasticBand_weight){
    for (auto group_iter : groupMap){
        GroupPath* groupPath = group_iter.second;

        std::set<size_t> explored_set;
        int nodeSize = groupPath->graphNodeMap.size();
        assert(nodeSize < 3000);

        for (auto graphPath_iter : groupPath->graphPathMap)
        {
            std::vector<size_t> pathIdx = graphPath_iter.second;

            for (size_t i=0; i<pathIdx.size() - 1; i++)
            {
                FlexGraphNode* node0 = groupPath->graphNodeMap[pathIdx[i]];
                FlexGraphNode* node1 = groupPath->graphNodeMap[pathIdx[i + 1]];

                size_t tag = (std::min(node0->nodeIdx, node1->nodeIdx) + 1.0) * nodeSize + std::max(node0->nodeIdx, node1->nodeIdx);
                if (explored_set.find(tag) != explored_set.end()){
                    continue;
                }
                explored_set.insert(tag);

                EdgeXYZ_ElasticBand* edge = new EdgeXYZ_ElasticBand();
                edge->setVertex(0, node0->vertex);
                edge->setVertex(1, node1->vertex);

                Eigen::Matrix<double,1,1> information;
                // information.fill( elasticBand_weight * current_loss );
                information.fill( elasticBand_weight );
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

bool SmootherXYZG2O::add_kinematicEdge(double kinematic_weight){
    for (auto group_iter : groupMap)
    {
        GroupPath* groupPath = group_iter.second;

        std::set<size_t> explored_set;
        int nodeSize = groupPath->graphNodeMap.size();
        assert(nodeSize < 3000 && nodeSize>2);

        for (auto graphPath_iter: groupPath->graphPathMap)
        {
            std::vector<size_t> pathIdx = graphPath_iter.second;

            for (size_t i = 1; i < pathIdx.size() - 1; i++)
            {
                FlexGraphNode* node0 = groupPath->graphNodeMap[pathIdx[i - 1]];
                FlexGraphNode* node1 = groupPath->graphNodeMap[pathIdx[i]];
                FlexGraphNode* node2 = groupPath->graphNodeMap[pathIdx[i + 1]];

                size_t tag = (
                    std::min(node0->nodeIdx, node2->nodeIdx) + 1.0) * nodeSize * nodeSize + 
                    std::max(node0->nodeIdx, node2->nodeIdx) * nodeSize + 
                    node1->nodeIdx;
                if (explored_set.find(tag) != explored_set.end()){
                    continue;
                }
                explored_set.insert(tag);

                bool success;
                // double current_loss;
                if ( node0->fixed )
                {
                    double dz = std::sin(node0->theta);
                    double dx = std::cos(node0->theta) * std::cos(node0->alpha);
                    double dy = std::cos(node0->theta) * std::sin(node0->alpha);
                    Eigen::Vector3d orientation = Eigen::Vector3d(dx, dy, dz);

                    // std::cout << "[DEBGUG]: BeginOrientation ( dx:" << dx << " dy:" << dy << " dz:" << dz << ")" << std::endl;

                    EdgeXYZ_VertexKinematics* edge = new EdgeXYZ_VertexKinematics(orientation, "BeginOrientation");
                    edge->setVertex(0, node0->vertex);
                    edge->setVertex(1, node1->vertex);

                    // current_loss = EdgeXYZ_VertexKinematics::lost_calc(node0->vertex, node1->vertex, orientation);

                    Eigen::Matrix<double,1,1> information;
                    // information.fill( kinematic_weight * current_loss );
                    information.fill( kinematic_weight );
                    edge->setInformation(information);

                    success = optimizer->addEdge(edge);
                    if (!success){
                        return false;
                    }  
                    
                }else if ( node2->fixed ){
                    double dz = std::sin(node2->theta);
                    double dx = std::cos(node2->theta) * std::cos(node2->alpha);
                    double dy = std::cos(node2->theta) * std::sin(node2->alpha);
                    Eigen::Vector3d orientation = Eigen::Vector3d(dx, dy, dz);

                    // std::cout << "[DEBGUG]: EndOrientation ( dx:" << dx << " dy:" << dy << " dz:" << dz << ")" << std::endl;

                    EdgeXYZ_VertexKinematics* edge = new EdgeXYZ_VertexKinematics(orientation, "EndOrientation");
                    edge->setVertex(0, node1->vertex);
                    edge->setVertex(1, node2->vertex);

                    // current_loss = EdgeXYZ_VertexKinematics::lost_calc(node1->vertex, node2->vertex, orientation);

                    Eigen::Matrix<double,1,1> information;
                    // information.fill( kinematic_weight * current_loss );
                    information.fill( kinematic_weight );
                    edge->setInformation(information); 

                    success = optimizer->addEdge(edge);
                    if (!success){
                        return false;
                    }
                }
                
                EdgeXYZ_Kinematics* edge = new EdgeXYZ_Kinematics();
                    
                edge->setVertex(0, node0->vertex);
                edge->setVertex(1, node1->vertex);
                edge->setVertex(2, node2->vertex);

                // current_loss = EdgeXYZ_Kinematics::lost_calc(node0->vertex, node1->vertex, node2->vertex);

                Eigen::Matrix<double,1,1> information;
                // information.fill( kinematic_weight * current_loss );
                information.fill( kinematic_weight );
                edge->setInformation(information); 

                success = optimizer->addEdge(edge);
                if (!success){
                    return false;
                }
            }
        }
    }
    return true;
}

bool SmootherXYZG2O::add_obstacleEdge(double obstacle_weight){
    if (obsTree->getTreeCount() == 0){
        return true;
    }
    
    std::vector<KDTree_XYZRA_Res*> resList;
    for (auto group_iter : groupMap)
    {
        GroupPath* group = group_iter.second;

        for (auto node_iter : group->graphNodeMap)
        {
            FlexGraphNode* node = node_iter.second;

            if ( node->fixed ){
                continue;
            }
            
            resList.clear();
            obsTree->nearest_range(node->x, node->y, node->z, node->radius * obstacle_detection_scale, resList);

            for (KDTree_XYZRA_Res* res : resList)
            {
                EdgeXYZ_Obstacle* edge = new EdgeXYZ_Obstacle(
                    Eigen::Vector3d(res->x, res->y, res->z), (node->radius + res->data->radius) * 1.025
                );
                edge->setVertex(0, node->vertex);

                Eigen::Matrix<double,1,1> information;
                information.fill(obstacle_weight);
                edge->setInformation(information);

                delete res;
                bool success = optimizer->addEdge(edge);
                if (!success){
                   return false;
                }
            }
        }
    }
    return true;
}

bool SmootherXYZG2O::add_pipeConflictEdge(double pipeConflict_weight){
    if (groupMap.size() <= 1){
        return true;
    }

    std::vector<KDTree_XYZRA_Res*> resList;
    for (size_t i=0; i < groupMap.size(); i++){
        GroupPath* group_i = groupMap[i];

        for (size_t j=i+1; j < groupMap.size(); j++){
            GroupPath* group_j = groupMap[j];

            for (auto iter : group_i->graphNodeMap)
            {
                FlexGraphNode* node = iter.second;

                resList.clear();
                group_j->graphTree->nearest_range(
                    node->x, node->y, node->z, 
                    (group_i->max_radius + group_j->max_radius) * pipeConflict_detection_scale, resList
                );

                for (KDTree_XYZRA_Res* res : resList)
                {
                    double dist = norm2_distance(node->x, node->y, node->z, res->x, res->y, res->z);
                    if (dist <= ( node->radius + res->data->radius) * pipeConflict_detection_scale )
                    {
                        EdgeXYZ_PipeConflict* edge = new EdgeXYZ_PipeConflict( (node->radius + res->data->radius)*1.01 );
                        edge->setVertex(0, node->vertex);
                        edge->setVertex(1, (group_j->graphNodeMap[res->data->idx])->vertex);

                        Eigen::Matrix<double,1,1> information;
                        information.fill(pipeConflict_weight);
                        edge->setInformation(information);

                        bool success = optimizer->addEdge(edge);
                        if (!success){
                            return false;
                        }
                    }

                    delete res;
                }
            }
        }
    }
    return true;
}

bool SmootherXYZG2O::add_boundaryEdge(double boundary_weight){
    for (auto group_iter : groupMap)
    {
        GroupPath* group = group_iter.second;

        for (auto node_iter : group->graphNodeMap)
        {
            FlexGraphNode* node = node_iter.second;

            if ( node->fixed ){
                continue;
            }

            EdgeXYZ_Boundary* edge = new EdgeXYZ_Boundary(min_x, min_y, min_z, max_x, max_y, max_z);
            edge->setVertex(0, node->vertex);

            Eigen::Matrix<double,1,1> information;
            information.fill(boundary_weight);
            edge->setInformation(information);

            bool success = optimizer->addEdge(edge);
            if (!success){
               return false;
            }            
        }
    }
    return true;
}

}
