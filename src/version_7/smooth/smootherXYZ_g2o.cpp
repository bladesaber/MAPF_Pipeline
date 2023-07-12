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
    // std::cout << "[Info]: Adding Vertexs ......" << std::endl;
    if (!status){
        std::cout << "[Error]: Adding Vertexs Fail" << std::endl;
        return false;
    }

    if (boundary_weight>0)
    {
        // std::cout << "[Info]: Adding Boundary Edge ......" << std::endl;
        status = add_boundaryEdge(boundary_weight);
        if (!status){
            std::cout << "[Error]: Adding Boundary Edge Fail" << std::endl;
            return false;
        }
    }
    
    if (elasticBand_weight>0)
    {
        // std::cout << "[Info]: Adding Elastic Band Edge ......" << std::endl;
        status = add_elasticBand(elasticBand_weight);
        if (!status){
            std::cout << "[Error]: Adding Elastic Band Edge Fail" << std::endl;
            return false;
        }
    }

    if (kinematic_weight>0)
    {
        // std::cout << "[Info]: Adding Kinematics Edge ......" << std::endl;
        status = add_kinematicEdge(kinematic_weight);
        if (!status){
            std::cout << "[Error]: Adding Kinematics Edge Fail" << std::endl;
            return false;
        }
    }

    if (obstacle_weight>0)
    {
        // std::cout << "[Info]: Adding Obstacle Edge ......" << std::endl;
        status = add_obstacleEdge(obstacle_weight);
        if (!status){
            std::cout << "[Error]: Adding Obstacle Edge Fail" << std::endl;
            return false;
        }
    }

    if (pipeConflict_weight>0)
    {
        // std::cout << "[Info]: Adding PipeConflict Edge ......" << std::endl;
        status = add_pipeConflictEdge(pipeConflict_weight);
        if (!status){
            std::cout << "[Error]: Adding PipeConflict Edge Fail" << std::endl;
            return false;
        }
    }

    return true;
}

bool SmootherXYZG2O::add_elasticBand(double elasticBand_weight){
    for (auto group_iter : groupMap){
        GroupPath* groupPath = group_iter.second;

        std::set<size_t> explored_set;
        explored_set.clear();
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
                
                EdgeXYZ_ElasticBand* edge = new EdgeXYZ_ElasticBand(elasticBand_kSpring);
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
                    double dx, dy, dz; 
                    std::tie(dx, dy, dz) = polar2Vec(node0->alpha, node0->theta);
                    Eigen::Vector3d orientation = Eigen::Vector3d(dx, dy, dz);
                    // std::cout << "[DEBGUG]: BeginOrientation ( dx:" << dx << " dy:" << dy << " dz:" << dz << ")" << std::endl;

                    double length = norm2_distance(node0->x, node0->y, node0->z, node1->x, node1->y, node1->z);
                    double cosTheta_target = 1.0 - std::cos( 1.0 / (3.0 * node1->radius) * length);

                    EdgeXYZ_VertexKinematics* edge = new EdgeXYZ_VertexKinematics(orientation, cosTheta_target, vertexKinematic_kSpring);
                    edge->setVertex(0, node0->vertex);
                    edge->setVertex(1, node1->vertex);

                    Eigen::Matrix<double,1,1> information;
                    information.fill( kinematic_weight );
                    edge->setInformation(information);

                    success = optimizer->addEdge(edge);
                    if (!success){
                        return false;
                    }  
                    
                }else if ( node2->fixed ){
                    double dx, dy, dz; 
                    std::tie(dx, dy, dz) = polar2Vec(node2->alpha, node2->theta);
                    Eigen::Vector3d orientation = Eigen::Vector3d(dx, dy, dz);
                    // std::cout << "[DEBGUG]: EndOrientation ( dx:" << dx << " dy:" << dy << " dz:" << dz << ")" << std::endl;

                    double length = norm2_distance(node1->x, node1->y, node1->z, node2->x, node2->y, node2->z);
                    double cosTheta_target = 1.0 - std::cos( 1.0 / (3.0 * node1->radius) * length);

                    EdgeXYZ_VertexKinematics* edge = new EdgeXYZ_VertexKinematics(orientation, cosTheta_target, vertexKinematic_kSpring);
                    edge->setVertex(0, node1->vertex);
                    edge->setVertex(1, node2->vertex);

                    Eigen::Matrix<double,1,1> information;
                    information.fill( kinematic_weight );
                    edge->setInformation(information); 

                    success = optimizer->addEdge(edge);
                    if (!success){
                        return false;
                    }
                }
                
                double length = std::min(
                    norm2_distance(node0->x, node0->y, node0->z, node1->x, node1->y, node1->z),
                    norm2_distance(node1->x, node1->y, node1->z, node2->x, node2->y, node2->z)
                );
                double cosTheta_target = 1.0 - std::cos( 1.0 / (3.0 * std::min(node1->radius, node2->radius)) * length);

                EdgeXYZ_Kinematics* edge = new EdgeXYZ_Kinematics(cosTheta_target, edgeKinematic_kSpring);
                edge->setVertex(0, node0->vertex);
                edge->setVertex(1, node1->vertex);
                edge->setVertex(2, node2->vertex);

                Eigen::Matrix<double,1,1> information;
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
            
            // 膨胀搜索主要是为了可以执行多步优化
            resList.clear();
            obsTree->nearest_range(node->x, node->y, node->z, node->radius * obstacle_detection_scale, resList);

            for (KDTree_XYZRA_Res* res : resList)
            {
                double dist = norm2_distance(node->x, node->y, node->z, res->x, res->y, res->z);
                if ( dist > (node->radius + res->data->radius) * 1.5 ){
                    delete res;
                    continue;
                }

                EdgeXYZ_Obstacle* edge = new EdgeXYZ_Obstacle(
                    Eigen::Vector3d(res->x, res->y, res->z), (node->radius + res->data->radius) * 1.5
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

            // EdgeXYZ_Boundary* edge = new EdgeXYZ_Boundary(min_x, min_y, min_z, max_x, max_y, max_z, node->radius);
            EdgeXYZ_Boundary* edge = new EdgeXYZ_Boundary(min_x, min_y, min_z, max_x, max_y, max_z, 0.0);
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

void SmootherXYZG2O::loss_report(
    size_t groupIdx,
    size_t pathIdx,
    double elasticBand_weight, 
    double kinematic_weight,
    double obstacle_weight,
    double pipeConflict_weight,
    double boundary_weight
){
    GroupPath* groupPath = groupMap[groupIdx];
    std::vector<size_t> nodeIdxs_path = groupPath->graphPathMap[pathIdx];

    FlexGraphNode* node0;
    FlexGraphNode* node1;
    FlexGraphNode* node2;
    double dx, dy, dz;
    double kinematic_cost = 0.0;
    double elasticBand_cost = 0.0;

    for (size_t i = 0; i < nodeIdxs_path.size(); i++)
    {
        double loss;

        std::cout << "[INFO] Idx:" << i << std::endl;
        if (kinematic_weight>0)
        {
            if ( i>0 && i<nodeIdxs_path.size() - 1){

                node0 = groupPath->graphNodeMap[nodeIdxs_path[i - 1]];
                node1 = groupPath->graphNodeMap[nodeIdxs_path[i]];
                node2 = groupPath->graphNodeMap[nodeIdxs_path[i + 1]];

                if (i == 1)
                {
                    double dx, dy, dz; 
                    std::tie(dx, dy, dz) = polar2Vec(node0->alpha, node0->theta);
                    Eigen::Vector3d orientation = Eigen::Vector3d(dx, dy, dz);
                    // std::cout << "  NormOrient ( dx:" << dx << " dy:" << dy << " dz:" << dz << ")" << std::endl;

                    double length = norm2_distance(node0->x, node0->y, node0->z, node1->x, node1->y, node1->z);
                    double cosTheta_target = 1.0 - std::cos( 1.0 / (3.0 * node1->radius) * length);
                    loss = EdgeXYZ_VertexKinematics::lost_calc(node0->vertex, node1->vertex, orientation, cosTheta_target, vertexKinematic_kSpring);
                    
                    kinematic_cost += loss;

                }else if ( i == nodeIdxs_path.size() - 2){
                    double dx, dy, dz; 
                    std::tie(dx, dy, dz) = polar2Vec(node2->alpha, node2->theta);
                    Eigen::Vector3d orientation = Eigen::Vector3d(dx, dy, dz);
                    // std::cout << "  NormOrient ( dx:" << dx << " dy:" << dy << " dz:" << dz << ")" << std::endl;

                    double length = norm2_distance(node1->x, node1->y, node1->z, node2->x, node2->y, node2->z);
                    double cosTheta_target = 1.0 - std::cos( 1.0 / (3.0 * node1->radius) * length);
                    loss = EdgeXYZ_VertexKinematics::lost_calc(node1->vertex, node2->vertex, orientation, cosTheta_target, vertexKinematic_kSpring);
                    
                    kinematic_cost += loss;
                }

                double length = std::min(
                    norm2_distance(node0->x, node0->y, node0->z, node1->x, node1->y, node1->z),
                    norm2_distance(node1->x, node1->y, node1->z, node2->x, node2->y, node2->z)
                );
                double cosTheta_target = std::cos( 1.0 / (3.0 * std::min(node1->radius, node2->radius)) * length);
                // double cosTheta_target = std::cos( 1.0 / (3.0 * std::min(node1->radius, node2->radius)) );

                loss = EdgeXYZ_Kinematics::lost_calc(
                    node0->vertex, node1->vertex, node2->vertex, cosTheta_target, edgeKinematic_kSpring
                );

                kinematic_cost += loss;
            }
        }

        if (elasticBand_weight>0)
        {
            if ( i < nodeIdxs_path.size() - 1){
                node0 = groupPath->graphNodeMap[nodeIdxs_path[i]];
                node1 = groupPath->graphNodeMap[nodeIdxs_path[i + 1]];
                loss = EdgeXYZ_ElasticBand::lost_calc(node0->vertex, node1->vertex, elasticBand_kSpring);
                elasticBand_cost += loss;
            }
        }
    }

    std::cout << "KinematicCost:" << kinematic_cost << " ElasticBandCost:" << elasticBand_cost << std::endl;
}

}
