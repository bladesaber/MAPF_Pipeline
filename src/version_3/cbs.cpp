#include "cbs.h"

void CBSNode::updateFirstConflict(
    AgentInfo* agent1,
        
    double conflict1_x, double conflict1_y, double conflict1_z, 
    double conflict1_radius, double conflict1_length, 
        
    size_ut conflict_agentIdx,
    double conflict2_x, double conflict2_y, double conflict2_z, 
    double conflict2_radius, double conflict2_length
){

    // // ------ Just For Debug
    // std::cout << "Debug3:" << std::endl;
    // std::cout << "Agent1:" << agent1->agentIdx;
    // std::cout << " x:" <<conflict1_x << " y:" << conflict1_y << " z:" << conflict1_z;
    // std::cout << " radius:" << conflict1_radius << " length:" << conflict1_length << std::endl;
    // std::cout << "Conflict Agent:" << conflict_agentIdx;
    // std::cout << " x:" <<conflict2_x << " y:" << conflict2_y << " z:" << conflict2_z;
    // std::cout << " radius:" << conflict2_radius << " length:" << conflict2_length << std::endl;
    // // -------------------------------------------

    if (!agent1->isConflict)
    {
        agent1->firstConflict = Conflict(
            agent1->agentIdx, conflict1_length,
            conflict1_x, conflict1_y, conflict1_z, conflict1_radius,
            conflict_agentIdx, conflict2_length,
            conflict2_x, conflict2_y, conflict2_z, conflict2_radius
        );
        agent1->isConflict = true;
        
    }else{
        if (conflict1_length < agent1->firstConflict.conflict1_length)
        {
            agent1->firstConflict = Conflict(
                agent1->agentIdx, conflict1_length,
                conflict1_x, conflict1_y, conflict1_z, conflict1_radius,
                conflict_agentIdx, conflict2_length,
                conflict2_x, conflict2_y, conflict2_z, conflict2_radius
            );
        }
    }

    // agent1->firstConflict.info();
}

void CBSNode::findAllAgentConflict(){
    AgentInfo* agent_i;
    AgentInfo* agent_j;
    
    for (auto iter : agentMap)
    {
        agent_i = iter.second;
        agent_i->isConflict = false;
        agent_i->conflictSet.clear();
    }

    std::shared_ptr<DetailPath> path_i;
    std::shared_ptr<KDTreeWrapper> tree_j;

    double x, y, z, length;
    KDTreeRes res;
    double distance;

    for (size_ut i = 0; i < num_of_agents; i++)
    {
        agent_i = agentMap[i];
        path_i = agent_i->detailPath;

        for (size_ut j = i + 1; j < num_of_agents; j++)
        {
            agent_j = agentMap[j];
            tree_j = agent_j->pathTree;

            for (size_t k = 0; k < path_i->size(); k++){
                std::tie(x, y, z, length) = path_i->at(k);
                tree_j->nearest(x, y, z, res);

                distance = norm2_distance(
                    x, y, z,
                    res.x, res.y, res.z
                );

                if (distance >= agent_i->radius + agent_j->radius){
                    continue;
                }

                // ----- Discrete Conflict Set
                // 因为agent是在步长为1的离散空间行走的，所以直接使用整数
                agent_i->conflictSet.insert(
                    std::make_tuple(
                        (int)round(res.x),
                        (int)round(res.y),
                        (int)round(res.z)
                    )
                );
                agent_j->conflictSet.insert(
                    std::make_tuple(
                        (int)round(x),
                        (int)round(y),
                        (int)round(z)
                    )
                );

                // // ------ Just For Debug
                // std::cout << "Debug2:" << std::endl;
                // std::cout << "   AgentIdx:" << agent_i->agentIdx << " x:" << res.x << " y:" << res.y << " z:" << res.z << " radius:" << res.data->radius << " length:" << length << std::endl;
                // std::cout << "   AgentIdx:" << agent_j->agentIdx << " x:" << x << " y:" << y << " z:" << z << " radius:" << agent_i->radius << " length:" << res.data->length << std::endl;
                // // -----------------------------------------

                updateFirstConflict(
                    agent_i,
                    res.x, res.y, res.z, res.data->radius, length,
                    agent_j->agentIdx,
                    x, y, z, agent_i->radius,res.data->length
                );
                updateFirstConflict(
                    agent_j,
                    x, y, z, agent_i->radius, res.data->length, 
                    agent_i->agentIdx,
                    res.x, res.y, res.z, agent_j->radius, length
                );
            }
        }
    }

    for (auto iter : agentMap)
    {
        agent_i = iter.second;
        agent_i->conflictNum = agent_i->conflictSet.size();
        agent_i->conflictSet.clear();
    }
}

void CBS::pushNode(CBSNode* node){
    open_list.push(node);
}

CBSNode* CBS::popNode(){
    CBSNode* node = open_list.top();
    open_list.pop();
    return node;
}

DetailPath CBS::sampleDetailPath(Path& path, Instance& instance, double stepLength){
    DetailPath detail_path;

    if (path.size() == 0)
    {
        return detail_path;
    }
    
    double lastX, lastY, lastZ;
    std::tie(lastX, lastY, lastZ) = instance.getCoordinate(path[0]);

    double curX, curY, curZ;
    double vecX, vecY, vecZ;
    double distance, real_stepLength;
    size_t num;
    double cur_length = 0.0;
    for (size_t i = 1; i < path.size(); i++)
    {
        std::tie(curX, curY, curZ) = instance.getCoordinate(path[i]);

        distance = norm2_distance(
            lastX, lastY, lastZ,
            curX, curY, curZ
        );

        num = std::ceil(distance / stepLength);
        real_stepLength = distance / (double)num;

        vecX = (curX - lastX) / distance;
        vecY = (curY - lastY) / distance;
        vecZ = (curZ - lastZ) / distance;

        // for last point
        if (i == path.size() - 1)
        {
            num += 1;
        }
        
        // std::cout << "lastX: " << lastX << " lastY: " << lastY << " lastZ: " << lastZ << std::endl;
        // std::cout << "curX: " << curX << " curY: " << curY << " curZ: " << curZ << std::endl;
        // std::cout << "vecX: " << vecX << " vecY: " << vecY << " vecZ: " << vecZ << std::endl;

        for (size_t j = 0; j < num; j++)
        {
            // std::cout << "[" <<  lastX + vecX * (j * real_stepLength) << ", " 
            //           << lastY + vecY * (j * real_stepLength) << ", "
            //           << lastZ + vecZ * (j * real_stepLength) << "]" << std::endl;

            detail_path.emplace_back(std::make_tuple(
                lastX + vecX * (j * real_stepLength),
                lastY + vecY * (j * real_stepLength),
                lastZ + vecZ * (j * real_stepLength),
                cur_length
            ));
            cur_length += real_stepLength;

        }

        lastX = curX;
        lastY = curY;
        lastZ = curZ;
    }

    return detail_path;
}

void CBS::compute_Heuristics(CBSNode* node){
    node->h_val = 0.;

    AgentInfo* agent;
    for (size_ut i = 0; i < node->num_of_agents; i++)
    {
        agent = node->agentMap[i];
        node->h_val += (double)agent->conflictNum / 2.0 * heuristics_mupltier;
    }
}

void CBS::compute_Gval(CBSNode* node){
    double x, y ,z, length;
    node->g_val = 0;

    for (auto iter : node->agentMap)
    {
        std::tie(x, y, z, length) = (*iter.second->detailPath).back();
        node->g_val += length;
    }
}

bool CBS::update_AgentPath(Instance& instance, CBSNode* node, size_ut agentIdx){
    AgentInfo* agent = node->agentMap[agentIdx];
    Path path = search_engines[agentIdx]->findPath(
        *agent->constrains,
        instance,
        agent->startPos,
        agent->endPos
    );

    if (path.size() == 0){
        return false;
    }

    node->update_DetailPath_And_Tree(
        agentIdx,
        sampleDetailPath(path, instance, stepLength)
    );

    runtime_search = search_engines[agentIdx]->runtime_search;

    return true;
}

bool CBS::isGoal(CBSNode* node){
    for (auto iter : node->agentMap)
    {
        if (iter.second->isConflict){
            return false;
        }
    }
    return true;
}