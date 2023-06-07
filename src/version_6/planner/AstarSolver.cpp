#include "AstarSolver.h"

namespace PlannerNameSpace{

AStarNode* AStarSolver::getNextNode(
    size_t neighbour_loc,
    AStarNode* lastNode, 
    ConstraintTable& constrain_table, 
    Instance& instance
){
    double next_g_val = getCost(instance, lastNode, neighbour_loc);
    double next_h_val = getHeuristic(instance, neighbour_loc, goal_locs);

    // std::cout << "DEBUG next_g_val:" << next_g_val << " next_h_val:" << next_h_val << std::endl;

    AStarNode* next_node = new AStarNode(
        neighbour_loc, // location
        next_g_val, // g val
        next_h_val, // h val
        lastNode, // parent
        lastNode->timestep + 1, // timestep
        0, // num_of_conflicts
        false // in_openlist
    );

    // Theta* Direct Connect
    if (with_AnyAngle)
    {
        AStarNode* oldParent = lastNode->parent;
        if (oldParent != nullptr)
        {
            double lineStart_x, lineStart_y, lineStart_z;
            double lineEnd_x, lineEnd_y, lineEnd_z;

            std::tie(lineStart_x, lineStart_y, lineStart_z) = instance.getCoordinate(oldParent->location);
            std::tie(lineEnd_x, lineEnd_y, lineEnd_z) = instance.getCoordinate(neighbour_loc);

            bool validOnSight = constrain_table.islineOnSight(
                lineStart_x, lineStart_y, lineStart_z,
                lineEnd_x, lineEnd_y, lineEnd_z,
                radius
            );

            if (validOnSight)
            {
                double g_val_onSight = oldParent->g_val + instance.getEulerDistance(neighbour_loc, oldParent->location);

                // std::cout << "DEBUG validOnSight g_val_onSight:" << g_val_onSight << " next_g_val:" << next_g_val << std::endl;
                
                // please use (less) here rather than (less or equal)
                if (g_val_onSight < next_g_val)
                {
                    
                    // ------ debug
                    // debugPrint(next_node, instance, "[Interm]: Skip Node:");
                    // --------

                    next_node->g_val = g_val_onSight;
                    next_node->parent = oldParent;
                    // next_node->num_of_conflicts = ;

                    return next_node;
                }
            }
        }
    }

    // ------ debug
    // debugPrint(next_node, instance, "[Interm]: Near Node:");
    // --------

    return next_node;
}

double AStarSolver::getCost(Instance& instance, AStarNode* cur_node, size_t next_loc){
    double cost = cur_node->g_val;

    // Trip Cost
    cost += 1.0;

    // Angle Cost
    if (cur_node->parent != nullptr && with_OrientCost )
    {
        size_t parent_x, parent_y, parent_z;
        std::tie(parent_x, parent_y, parent_z) = instance.getCoordinate(cur_node->parent->location);

        size_t cur_x, cur_y, cur_z;
        std::tie(cur_x, cur_y, cur_z) = instance.getCoordinate(cur_node->location);

        size_t next_x, next_y, next_z;
        std::tie(next_x, next_y, next_z) = instance.getCoordinate(next_loc);
        
        if (cur_x - parent_x != next_x - cur_x){
            cost += 1.0;
        }
        if (cur_y - parent_y != next_y - cur_y){
            cost += 1.0;
        }
        if (cur_z - parent_z != next_z - cur_z){
            cost += 1.0;
        }
    }

    return cost;
}

bool AStarSolver::validMove(Instance& instance, ConstraintTable& constrain_table, int curr, int next) const {
    // 离散检测
    // double x, y, z;
    // std::tie(x, y, z) = instance.getCoordinate(next);
    // if(constrain_table.isConstrained(x, y, z, radius)){
    //     return false;
    // }

    // 连续检测
    double lineStart_x, lineStart_y, lineStart_z;
    double lineEnd_x, lineEnd_y, lineEnd_z;
    std::tie(lineStart_x, lineStart_y, lineStart_z) = instance.getCoordinate(curr);
    std::tie(lineEnd_x, lineEnd_y, lineEnd_z) = instance.getCoordinate(next);

    if(constrain_table.isConstrained(
        lineStart_x, lineStart_y, lineStart_z,
        lineEnd_x, lineEnd_y, lineEnd_z, radius
    )){
        return false;
    }
    return true;
}

bool AStarSolver::isValidSetting(Instance& instance, ConstraintTable& constrain_table, size_t loc){
    double x, y, z;
    std::tie(x, y, z) = instance.getCoordinate(loc);
    if(constrain_table.isConstrained(x, y, z, radius)){
         return false;
    }
    return true;
}

Path AStarSolver::findPath(
    double radius, std::vector<ConstrainType> constraints, Instance& instance,
    size_t start_loc, std::vector<size_t>& goal_locs, bool check_EndPosValid
){
    // temporary Params Setting
    this->radius = radius;
    this->start_loc = start_loc;
    this->goal_locs = goal_locs;

    num_expanded = 0;
	num_generated = 0;
    auto start_time = clock();

    // ------ Create Constrain Table
    ConstraintTable constrain_table = ConstraintTable();
    for (auto constraint : constraints){
        constrain_table.insert2CT(constraint);
    }
    runtime_build_CT = (double) (clock() - start_time) / CLOCKS_PER_SEC;

    // ------ Init Path
    Path path;

    // ------ Check startPos or endPos Valid
    if (!isValidSetting(instance, constrain_table, start_loc)){
        std::cout << "[DEBUG]: StartPos is not Valid" << std::endl;
        return path;
    }
    if (check_EndPosValid){
        for (size_t goal_loc: goal_locs){
            if (!isValidSetting(instance, constrain_table, goal_loc)){
                std::cout << "[DEBUG]: EndPos is not Valid" << std::endl;
                return path;
            }
        }
    }
    
    // ------ Init Searching Setting
    AStarNode* start_node = new AStarNode(
        start_loc,  // location
		0,  // g val
		getHeuristic(instance, start_loc, goal_locs),  // h val
		nullptr,  // parent
		0,  // timestep
		0, // num_of_conflicts
	    false // in_openlist
    );
    pushNode(start_node);
	allNodes_table.insert(start_node);

    start_time = clock();
    while ( !open_list.empty() ){
        AStarNode* cur_node = popNode();

        if ( isGoal(cur_node->location) )
        {
            updatePath(cur_node, path);
            break;
        }

        num_expanded++;
        for (int neighbour_loc : instance.getNeighbors(cur_node->location)){
            if (cur_node->parent != nullptr){
                if (neighbour_loc == cur_node->parent->location){
                    continue;
                }
            }

            bool valid_move = this->validMove(instance, constrain_table, cur_node->location, neighbour_loc);
            if (!valid_move){
                continue;
            }

            AStarNode* next_node = getNextNode(
                neighbour_loc, 
                cur_node,
                constrain_table,
                instance
            );

            auto it = allNodes_table.find(next_node);
            if (it == allNodes_table.end()){
                pushNode(next_node);
                allNodes_table.insert(next_node);

                // ------ debug
                // debugPrint(next_node, instance, "New Next node:");
                // --------

                continue;
            }

            AStarNode* existing_next = *it;
            if (
                next_node->getFVal() < existing_next->getFVal() 
                // || 
                // (
                //     next_node->getFVal() <= existing_next->getFVal() + bandwith && 
                //     next_node->num_of_conflicts < existing_next->num_of_conflicts
                // )
            ){
                existing_next->copy(*next_node);

                if (!existing_next->in_openlist) // if its in the closed list (reopen)
                {
                    pushNode(existing_next);

                    // ------ debug
                    // debugPrint(next_node, instance, "Reopen node:");
                    // ------

                }else{
                    open_list.increase(existing_next->open_handle);
                }
            }

            delete next_node;
        }
    }

    releaseNodes();

    runtime_search = (double) (clock() - start_time) / CLOCKS_PER_SEC;

    return path;
}

Path AStarSolver::findPath(
    double radius, ConstraintTable& constrain_table, Instance& instance,
    size_t start_loc, std::vector<size_t>& goal_locs, bool check_EndPosValid
){
    // temporary Params Setting
    this->radius = radius;
    this->start_loc = start_loc;
    this->goal_locs = goal_locs;

    num_expanded = 0;
	num_generated = 0;
    auto start_time = clock(); 

    // ------ Init Path
    Path path;

    // ------ Check startPos or endPos Valid
    if (!isValidSetting(instance, constrain_table, start_loc)){
        std::cout << "[DEBUG]: StartPos is not Valid" << std::endl;
        return path;
    }
    if (check_EndPosValid){
        for (size_t goal_loc: goal_locs){
            if (!isValidSetting(instance, constrain_table, goal_loc)){
                std::cout << "[DEBUG]: EndPos is not Valid" << std::endl;
                return path;
            }
        }
    }
    
    // ------ Init Searching Setting
    AStarNode* start_node = new AStarNode(
        start_loc,  // location
		0,  // g val
		getHeuristic(instance, start_loc, goal_locs),  // h val
		nullptr,  // parent
		0,  // timestep
		0, // num_of_conflicts
	    false // in_openlist
    );
    pushNode(start_node);
	allNodes_table.insert(start_node);

    start_time = clock();
    while ( !open_list.empty() ){
        AStarNode* cur_node = popNode();

        if ( isGoal(cur_node->location) )
        {
            updatePath(cur_node, path);
            break;
        }

        num_expanded++;
        for (int neighbour_loc : instance.getNeighbors(cur_node->location)){
            if (cur_node->parent != nullptr){
                if (neighbour_loc == cur_node->parent->location){
                    continue;
                }
            }

            bool valid_move = this->validMove(instance, constrain_table, cur_node->location, neighbour_loc);
            if (!valid_move){
                continue;
            }

            AStarNode* next_node = getNextNode(
                neighbour_loc, 
                cur_node,
                constrain_table,
                instance
            );

            auto it = allNodes_table.find(next_node);
            if (it == allNodes_table.end()){
                pushNode(next_node);
                allNodes_table.insert(next_node);

                // ------ debug
                // debugPrint(next_node, instance, "New Next node:");
                // --------

                continue;
            }

            AStarNode* existing_next = *it;
            if (
                next_node->getFVal() < existing_next->getFVal() 
                // || 
                // (
                //     next_node->getFVal() <= existing_next->getFVal() + bandwith && 
                //     next_node->num_of_conflicts < existing_next->num_of_conflicts
                // )
            ){
                existing_next->copy(*next_node);

                if (!existing_next->in_openlist) // if its in the closed list (reopen)
                {
                    pushNode(existing_next);

                    // ------ debug
                    // debugPrint(next_node, instance, "Reopen node:");
                    // ------

                }else{
                    open_list.increase(existing_next->open_handle);
                }
            }

            delete next_node;
        }
    }

    releaseNodes();

    runtime_search = (double) (clock() - start_time) / CLOCKS_PER_SEC;

    return path;
}


}
