#include "cbs.h"

void CBSNode::updateFirstConflict(
    double x, double y, double z, 
    double radius, double length,
    AgentInfo* agent
){
    if (!agent->isConflict)
    {
        agent->firstConflictLength = length;
        agent->firstConflict = std::make_tuple(x, y, z, radius);
        agent->isConflict = true;

    }else{
        if (length < agent->firstConflictLength)
        {
            agent->firstConflictLength = length;
            agent->firstConflict = std::make_tuple(x, y, z, radius);
        }
    }
}

void CBSNode::updateAgentConflict(size_t agentIdx){
    AgentInfo* agent = &(agentMap[agentIdx]);
    DetailPath path = agent->detailPath;

    double x, y, z, length;
    KDTreeRes res;
    double distance;
    AgentInfo* other_agent;
    KDTreeWrapper otherAgent_tree;
    double num_of_conflict;

    agent->isConflict = false;

    for (auto iter : agentMap)
    {
        if (iter.first == agentIdx){
            continue;
        }
        
        other_agent = &agentMap[iter.first];
        num_of_conflict = 0;
        otherAgent_tree = other_agent->pathTree;

        for (size_t i = 0; i < path.size(); i++)
        {
            std::tie(x, y, z, length) = path[i];
            otherAgent_tree.nearest(x, y, z, res);

            distance = norm2_distance(
                x, y, z,
                res.x, res.y, res.z
            );

            if (distance >= agent->radius + other_agent->radius){
                continue;
            }

            num_of_conflict += 1;
            updateFirstConflict(
                res.x, res.y, res.z, other_agent->radius, length, agent
            );
        }

        agent->costMap[other_agent->agentIdx] = num_of_conflict;
        other_agent->costMap[agent->agentIdx] = num_of_conflict;

    }
}

void CBSNode::findAllAgentConflict(){
    double x, y, z, length;
    KDTreeRes res;
    double distance;

    AgentInfo* agent_i;
    AgentInfo* agent_j;
    DetailPath path_i;
    KDTreeWrapper tree_j;

    for (auto iter : agentMap)
    {
        agent_i = &iter.second;
        agent_i->isConflict = false;
        for (size_ut i = 0; i < num_of_agents; i++)
        {
            if (i != agent_i->agentIdx)
            {
                agent_i->costMap[i] = 0;
            }
        }
    }

    for (size_ut i = 0; i < num_of_agents; i++)
    {
        agent_i = &agentMap[i];
        path_i = agent_i->detailPath;

        for (size_ut j = i; j < num_of_agents; j++)
        {
            agent_j = &agentMap[j];
            tree_j = agent_j->pathTree;

            for (size_t k = 0; k < path_i.size(); i++){
                std::tie(x, y, z, length) = path_i[i];
                tree_j.nearest(x, y, z, res);

                distance = norm2_distance(
                    x, y, z,
                    res.x, res.y, res.z
                );

                if (distance >= agent_i->radius + agent_j->radius){
                    continue;
                }

                agent_i->costMap[agent_j->agentIdx] += 1;
                agent_j->costMap[agent_i->agentIdx] += 1;
                
                updateFirstConflict(
                    res.x, res.y, res.z, agent_j->radius, length, agent_i
                );
                updateFirstConflict(
                    x, y, z, agent_i->radius, res.data->length, agent_j
                );
            }
        }
    }
}

double CBSNode::getHeuristics(){
    double h_val = 0.;

    AgentInfo* agent;
    for (size_ut i = 0; i < num_of_agents; i++)
    {
        agent = &agentMap[i];
        if (!agent->isConflict)
        {
            continue;
        }
        
        for (size_ut j = i; j < num_of_agents; j++){
            h_val += agent->costMap[j];
        }
    }

    return h_val;
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

            cur_length += real_stepLength;
            detail_path.emplace_back(std::make_tuple(
                lastX + vecX * (j * real_stepLength),
                lastY + vecY * (j * real_stepLength),
                lastZ + vecZ * (j * real_stepLength),
                cur_length
            ));
        }

        lastX = curX;
        lastY = curY;
        lastZ = curZ;
    }

    return detail_path;
}