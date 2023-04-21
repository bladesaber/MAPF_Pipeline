#include "angleAstar.h"

bool AngleAStar::validMove(Instance& instance, ConstraintTable& constrain_table, int curr, int next) const {
    double x, y, z;
    std::tie(x, y, z) = instance.getCoordinate(next);

    if(constrain_table.isConstrained(x, y, z, radius)){
        return false;
    }
    return true;
}

void AngleAStar::updatePath(const AStarNode* goal_node, Path& path){
    auto curr = goal_node;
    while (curr != nullptr)
    {
        path.emplace_back(curr->location);
        curr = curr->parent;
    }
    std::reverse(path.begin(), path.end());
}

void AngleAStar::pushNode(AStarNode* node){
    num_generated++;

    node->open_handle = open_list.push(node);
	node->in_openlist = true;

    // if (this->focus_optimal)
    // {
    //     if (node->getFVal() <= lower_bound)
    //     {
    //         node->focal_handle = focal_list.push(node);
    //     }   
    // }
}

AStarNode* AngleAStar::popNode(){
    // AStarNode* node;
    // if (focus_optimal)
    // {
    //     node = focal_list.top();
    //     focal_list.pop();
	//     open_list.erase(node->open_handle);

    // }else{
    //     node = open_list.top();
    //     open_list.pop();
    // }

    AStarNode* node = open_list.top();
    open_list.pop();

	node->in_openlist = false;
    return node;
}

double AngleAStar::getHeuristic(Instance& instance, int loc1, int loc2){
    return instance.getManhattanDistance(loc1, loc2);
    // return instance.getEulerDistance(loc1, loc2);
};

AStarNode* AngleAStar::getAnyAngleNode(
    int neighbour_loc, 
    int goal_loc,
    AStarNode* lastNode, 
    ConstraintTable& constrain_table, 
    Instance& instance
){
    AStarNode* oldParent = lastNode->parent;

    int next_timestep = lastNode->timestep + 1;
    double next_g_val = lastNode->g_val + 1.0;
    double next_h_val = getHeuristic(instance, neighbour_loc, goal_loc);
    // int next_internal_conflicts = lastNode->num_of_conflicts + constrain_table.getNumOfConflictsForStep(curr->location, neighbour_loc);
    int next_internal_conflicts = 0;

    AStarNode* next_node = new AStarNode(
        neighbour_loc, // location
        next_g_val, // g val
        next_h_val, // h val
        lastNode, // parent
        next_timestep, // timestep
        next_internal_conflicts, // num_of_conflicts
        false // in_openlist
    );

    if (oldParent != nullptr)
    {
        bool validOnSight = constrain_table.islineOnSight(instance, oldParent->location, neighbour_loc, this->radius);
        if (validOnSight)
        {
            double g_val_onSight = oldParent->g_val + instance.getEulerDistance(neighbour_loc, oldParent->location);
            
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

    // ------ debug
    // debugPrint(next_node, instance, "[Interm]: Near Node:");
    // --------

    return next_node;
}

Path& AngleAStar::findPath(
    std::vector<ConstrainType> constraints,
    Instance& instance,
    const std::tuple<int, int, int>& start_state, 
    const std::tuple<int, int, int>& goal_state
){
    num_expanded = 0;
	num_generated = 0;
    auto start_time = clock();

    // ------ Create Constrain Table
    ConstraintTable constrain_table = ConstraintTable();
    for (auto constraint : constraints)
    {
        constrain_table.insert2CT(constraint);
    }
    runtime_build_CT = (double) (clock() - start_time) / CLOCKS_PER_SEC;

    // ------ Init Search
    int start_loc = instance.linearizeCoordinate(start_state);
    int goal_loc = instance.linearizeCoordinate(goal_state);

    Path path;

    AStarNode* start_node = new AStarNode(
        start_loc,  // location
		0,  // g val
		getHeuristic(instance, start_loc, goal_loc),  // h val
		nullptr,  // parent
		0,  // timestep
		0, // num_of_conflicts
	    false // in_openlist
    );

    pushNode(start_node);
	allNodes_table.insert(start_node);

    // ------ Begin Search
    start_time = clock();
    while (!open_list.empty()){
        AStarNode* curr = popNode();

        if (curr->location == goal_loc)
        {
            updatePath(curr, path);
            break;
        }

        num_expanded++;
        for (int neighbour_loc : instance.getNeighbors(curr->location)){
            if (curr->parent != nullptr){
                if (neighbour_loc == curr->parent->location){
                    continue;
                }   
            }
            
            bool valid_move = this->validMove(instance, constrain_table, curr->location, neighbour_loc);
            if (!valid_move){
                continue;
            }

            AStarNode* next_node = getAnyAngleNode(
                neighbour_loc, 
                goal_loc,
                curr,
                constrain_table,
                instance
            );

            auto it = allNodes_table.find(next_node);
            if (it == allNodes_table.end())
            {
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

void AngleAStar::releaseNodes(){
    open_list.clear();
	for (auto node: allNodes_table){
		delete node;
    }
	allNodes_table.clear();
}