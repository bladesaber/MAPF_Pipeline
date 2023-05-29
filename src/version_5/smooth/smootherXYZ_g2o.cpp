#include "smootherXYZ_g2o.h"

namespace SmootherNameSpace{

bool SmootherXYZG2O::add_vertexs(){
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
                node->vertex = new VertexXYZ(node->x, node->y, node->z, true);
            }
            else{
                node->vertex = new VertexXYZ(node->x, node->y, node->z, false);
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

void SmootherXYZG2O::build_graph(
    double elasticBand_weight,
    double kinematic_weight,
    double obstacle_weight,
    double pipeConflict_weight
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

    if (kinematic_weight>0){
        status = add_kinematicEdge(kinematic_weight);
        if (!status){
            std::cout << "[Error]: Adding Kinematics Edge Fail" << std::endl;
            return;
        }
    }

    if (obstacle_weight>0)
    {
        status = add_obstacleEdge(obstacle_weight);
        if (!status){
            std::cout << "[Error]: Adding Obstacle Edge Fail" << std::endl;
            return;
        }
    }
}

void SmootherXYZG2O::loss_info(
    double elasticBand_weight, 
    double kinematic_weight,
    double obstacle_weight,
    double pipeConflict_weight
){
    GroupPathNode* node0;
    GroupPathNode* node1;
    GroupPathNode* node2;

    for (auto group_iter : groupMap)
    {
        GroupPath* groupPath = group_iter.second;

        for (size_t pathIdx : groupPath->pathIdxs_set)
        {
            std::vector<size_t> nodeIdxs_path = groupPath->extractPath(pathIdx);

            std::cout << "GroupIdx:" << groupPath->groupIdx << " PathIdx:" << pathIdx << std::endl;
            for (size_t i = 0; i < nodeIdxs_path.size(); i++)
            {
                std::cout << "[DEBUG] Idx:" << i << std::endl;
                double loss;

                if (kinematic_weight>0)
                {
                    if ( i>0 && i<nodeIdxs_path.size() - 1){

                        node0 = groupPath->nodeMap[nodeIdxs_path[i - 1]];
                        node1 = groupPath->nodeMap[nodeIdxs_path[i]];
                        node2 = groupPath->nodeMap[nodeIdxs_path[i + 1]];

                        if (i == 1)
                        {
                            double dz = std::sin(node0->theta);
                            double dx = std::cos(node0->theta) * std::cos(node0->alpha);
                            double dy = std::cos(node0->theta) * std::sin(node0->alpha);
                            std::cout << "  orientation ( dx:" << dx << " dy:" << dy << " dz:" << dz << ")" << std::endl;

                            Eigen::Vector3d orientation = Eigen::Vector3d(dx, dy, dz);
                            loss = EdgeXYZ_VertexKinematics::lost_calc(node0->vertex, node1->vertex, orientation, true);
                            std::cout << "  Kinematic Vertex Loss:" << loss << " Infomation:" << kinematic_weight << std::endl;

                        }else if ( i == nodeIdxs_path.size() - 2){
                            double dz = std::sin(node2->theta);
                            double dx = std::cos(node2->theta) * std::cos(node2->alpha);
                            double dy = std::cos(node2->theta) * std::sin(node2->alpha);
                            std::cout << "  orientation ( dx:" << dx << " dy:" << dy << " dz:" << dz << ")" << std::endl;

                            Eigen::Vector3d orientation = Eigen::Vector3d(dx, dy, dz);
                            loss = EdgeXYZ_VertexKinematics::lost_calc(node1->vertex, node2->vertex, orientation, true);
                            std::cout << "  Kinematic Vertex Loss:" << loss  << " Infomation:" << kinematic_weight << std::endl;

                        }

                        loss = EdgeXYZ_Kinematics::lost_calc(node0->vertex, node1->vertex, node2->vertex, true);
                        std::cout << "  Kinematic Edge Loss:" << loss  << " Infomation:" << kinematic_weight << std::endl;
                    }
                }

                if (elasticBand_weight>0)
                {
                    if ( i<nodeIdxs_path.size() - 1){
                        node0 = groupPath->nodeMap[nodeIdxs_path[i]];
                        node1 = groupPath->nodeMap[nodeIdxs_path[i + 1]];
                        loss = EdgeXYZ_ElasticBand::lost_calc(node0->vertex, node1->vertex);
                        std::cout << "  ElasticBand Edge Loss:" << loss  << " Infomation:" << elasticBand_weight << std::endl;
                    }
                }

                if (obstacle_weight>0)
                {
                    node0 = groupPath->nodeMap[nodeIdxs_path[i]];

                    if ( 
                        ((node0->parentIdxsMap.size() != 0) || (node0->childIdxsMap.size() != 0)) && 
                        (obsTree->getTreeCount() > 0) 
                    ){
                        std::vector<KDTree_XYZRA_Res*> resList;
                        obsTree->nearest_range(node0->x, node0->y, node0->z, node0->radius * 1.5, resList);
                        for (KDTree_XYZRA_Res* res : resList){
                            loss = EdgeXYZ_Obstacle::lost_calc(
                                node0->vertex, Eigen::Vector3d(res->x, res->y, res->y), node0->radius 
                            );
                            std::cout << "  Obstacle Edge Loss:" << loss;
                            std::cout << " x:" << res->x << " y:" << res->y << " z:" << res->z << " Infomation:" << obstacle_weight << std::endl;
                            delete res;
                        }
                    }
                }

            }
        }
    }
}

bool SmootherXYZG2O::add_elasticBand(double elasticBand_weight){
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
                
                EdgeXYZ_ElasticBand* edge = new EdgeXYZ_ElasticBand();
                edge->setVertex(0, node0->vertex);
                edge->setVertex(1, node1->vertex);

                // double current_loss = EdgeXYZ_ElasticBand::lost_calc(node0->vertex, node1->vertex);
                
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
        int nodeSize = groupPath->nodeMap.size();

        assert(nodeSize < 1000);

        for (size_t pathIdx : groupPath->pathIdxs_set)
        {
            std::vector<size_t> nodeIdxs_path = groupPath->extractPath(pathIdx);

            for (size_t i = 1; i < nodeIdxs_path.size() - 1; i++)
            {
                GroupPathNode* node0 = groupPath->nodeMap[nodeIdxs_path[i - 1]];
                GroupPathNode* node1 = groupPath->nodeMap[nodeIdxs_path[i]];
                GroupPathNode* node2 = groupPath->nodeMap[nodeIdxs_path[i + 1]];

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
                if (i == 1)
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
                    
                }else if ( i == nodeIdxs_path.size() - 2){
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

        for (auto node_iter : group->nodeMap)
        {
            GroupPathNode* node = node_iter.second;

            if ((node->parentIdxsMap.size() == 0) || (node->childIdxsMap.size() == 0)){
                continue;
            }
            
            resList.clear();
            obsTree->nearest_range(node->x, node->y, node->z, node->radius * 3.0, resList);

            for (KDTree_XYZRA_Res* res : resList)
            {
                EdgeXYZ_Obstacle* edge = new EdgeXYZ_Obstacle(
                    Eigen::Vector3d(res->x, res->y, res->z), node->radius * 1.025
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
    for (size_t i=0; i < groupMap.size(); i++)
    {
        GroupPath* group_i = groupMap[i];

        for (size_t j=i+1; j < groupMap.size(); j++){
            GroupPath* group_j = groupMap[j];

            for (auto iter : group_i->nodeMap)
            {
                GroupPathNode* node = iter.second;

                resList.clear();
                group_j->pathTree->nearest_range(
                    node->x, node->y, node->z, 
                    (group_i->max_radius + group_j->max_radius) * 1.5, resList
                );

                for (KDTree_XYZRA_Res* res : resList)
                {
                    double dist = norm2_distance(
                        node->x, node->y, node->z,
                        res->x, res->y, res->z
                    );
                    if (dist <= ( node->radius + res->data->radius) * 1.5 )
                    {
                        EdgeXYZ_PipeConflict* edge = new EdgeXYZ_PipeConflict( (node->radius + res->data->radius)*1.025 );
                        edge->setVertex(0, node->vertex);
                        edge->setVertex(1, (group_j->nodeMap[res->data->idx])->vertex);

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

Path_XYZR SmootherXYZG2O::detailSamplePath(Path_XYZR& path_xyzr, double stepLength){
    Path_XYZR new_detail_path;

    double lastX, lastY, lastZ, lastRadius;
    double curX, curY, curZ, curRadius;

    std::tie(lastX, lastY, lastZ, lastRadius) = path_xyzr[0];

    double distance, real_stepLength;
    size_t num;
    double vecX, vecY, vecZ, vecRaiuds;
    double cur_length = 0.0;

    for (size_t i = 1; i < path_xyzr.size(); i++){
        std::tie(curX, curY, curZ, curRadius) = path_xyzr[i];

        distance = norm2_distance(
            lastX, lastY, lastZ,
            curX, curY, curZ
        );

        if (distance < stepLength){
            continue;
        }
        
        num = std::ceil(distance / stepLength);
        real_stepLength = distance / (double)num;
        vecX = (curX - lastX) / distance;
        vecY = (curY - lastY) / distance;
        vecZ = (curZ - lastZ) / distance;
        vecRaiuds = (curRadius - lastRadius) / (double)num;

        // for last point
        if (i == path_xyzr.size() - 1){
            num += 1;
        }

        for (size_t j = 0; j < num; j++)
        {
            new_detail_path.emplace_back(std::make_tuple(
                lastX + vecX * (j * real_stepLength),
                lastY + vecY * (j * real_stepLength),
                lastZ + vecZ * (j * real_stepLength),
                curRadius + vecRaiuds * j
            ));
        }

        lastX = curX;
        lastY = curY;
        lastZ = curZ;
    }

    return new_detail_path;
}

}
