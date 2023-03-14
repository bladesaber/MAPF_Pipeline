//
// Created by quan on 23-3-13.
//

#include "spaceTimeAstar.h"

bool SpaceTimeAStar::validMove(Instance& instance, ConstraintTable& constrain_table, int curr, int next) const {
    if (next < 0 || next >= instance.map_size){
        return false;
    }
    if(constrain_table.isConstrained(next)){
        return false;
    }
    return true;
}

void SpaceTimeAStar::pushNode(AStarNode* node){
    num_generated++;
    if (this->focus_optimal)
    {
        node->open_handle = open_list.push(node);
	    node->in_openlist = true;
        if (node->getFVal() <= this->lower_bound)
        {
            node->focal_handle = focal_list.push(node);
        }
        
    }else{
        node->open_handle = open_list.push(node);
	    node->in_openlist = true;
    }
}

AStarNode* SpaceTimeAStar::popNode(){
    if (this->focus_optimal)
    {
        auto node = focal_list.top();
        focal_list.pop();
	    open_list.erase(node->open_handle);
	    node->in_openlist = false;
        return node;
    }else{
        auto node = open_list.top();
        open_list.pop();
        node->in_openlist = false;
        return node;
    }
}

void SpaceTimeAStar::updateFocalList(){
    if (this->focus_optimal)
    {
        auto open_head = open_list.top();
        if (open_head->getFVal() > min_f_val)
        {
		    int new_min_f_val = (int) open_head->getFVal();
		    int new_lower_bound = std::max(lower_bound, new_min_f_val);
            for (auto n : open_list)
            {
                if (n->getFVal() > lower_bound && n->getFVal() <= new_lower_bound){
                    n->focal_handle = focal_list.push(n);
                }
            }
            min_f_val = new_min_f_val;
            lower_bound = new_lower_bound;
        }
    }
}

void SpaceTimeAStar::updatePath(const AStarNode* goal_node, Path& path){
    path.resize(goal_node->timestep + 1);
    auto curr = goal_node;
    while (curr != nullptr)
    {
        path[curr->timestep].location = curr->location;
        curr = curr->parent;
    }
    
}

Path SpaceTimeAStar::findPath(
    const CBSNode& cbs_node, Instance& instance,
    const std::pair<int, int> start_state, std::pair<int, int> goal_state

){
    num_expanded = 0;
	num_generated = 0;
    timestep = 0;

    // build constraint table
    ConstraintTable constrain_table = ConstraintTable();
	auto starrt_time = clock();
    constrain_table.buildCT(cbs_node, agent_idx);
    runtime_build_CT = (double) (clock() - starrt_time) / CLOCKS_PER_SEC;

    starrt_time = clock();
    constrain_table.buildCAT(cbs_node, agent_idx);
    runtime_build_CAT = (double) (clock() - starrt_time) / CLOCKS_PER_SEC;

    int start_loc = instance.linearizeCoordinate(start_state);
    int goal_loc = instance.linearizeCoordinate(goal_state);

    auto start_node = new AStarNode(
        start_loc,  // location
		0,  // g val
		getHeuristic(instance, start_loc, goal_loc),  // h val
		nullptr,  // parent
		timestep,  // timestep
		0, // num_of_conflicts
	    false // in_openlist
    );

    num_generated++;
	start_node->open_handle = open_list.push(start_node);
	start_node->focal_handle = focal_list.push(start_node);
	start_node->in_openlist = true;
	allNodes_table.insert(start_node);

    Path path;
    while (!open_list.empty())
    {
        updateFocalList(); // update FOCAL if min f-val increased
        auto* curr = popNode();

        if (curr->location == goal_loc)
        {
            updatePath(curr, path);
            break;
        }
        
        for (int neighbour_loc : instance.getNeighbors(curr->location))
        {
            bool valid_move = this->validMove(instance, constrain_table, curr->location, neighbour_loc);
            if (valid_move)
            {
                int next_timestep = curr->timestep + 1;
                
                int next_g_val = curr->g_val + 1;
                int next_h_val = getHeuristic(instance, curr->location, goal_loc);
                int next_internal_conflicts = curr->num_of_conflicts + constrain_table.getNumOfConflictsForStep(
                    curr->location, neighbour_loc
                );

                AStarNode* next_node = new AStarNode(
                    neighbour_loc, // location
                    next_g_val, // g val
                    next_h_val, // h val
                    curr, // parent
                    next_timestep, // timestep
                    next_internal_conflicts, // num_of_conflicts
                    false // in_openlist
                );

                auto it = allNodes_table.find(next_node);
                if (it == allNodes_table.end())
                {
                    pushNode(next_node);
                    allNodes_table.insert(next_node);
                    continue;
                }
                
                AStarNode* existing_next = *it;
                if (
                    existing_next->getFVal() > next_node->getFVal() ||
                    (existing_next->getFVal() == next_node->getFVal() && 
                    existing_next->num_of_conflicts > next_node->num_of_conflicts)
                )
                {
                    if (!existing_next->in_openlist) // if its in the closed list (reopen)
                    {
                        existing_next->copy(*next_node);
                        pushNode(existing_next);
                    }else{
                        bool update_open = false;
                        if (existing_next->getFVal() > next_g_val + next_h_val){
                            update_open = true;
                        }

                        if (focus_optimal)
                        {
                            // check if it was above the focal bound before and now below (thus need to be inserted)
                            bool add_to_focal = false;  
                            // check if it was inside the focal and needs to be updated (because f-val changed)
                            bool update_in_focal = false;
                            
                            if ((next_g_val + next_h_val) <= lower_bound){
                                if (existing_next->getFVal() > lower_bound){
                                    add_to_focal = true;
                                }else{
                                    update_in_focal = true;
                                }
                            }
                            
                            existing_next->copy(*next_node);

                            if (update_open){
                                open_list.increase(existing_next->open_handle);
                            }
                            if (add_to_focal){
                                existing_next->focal_handle = focal_list.push(existing_next);
                            }
                            if (update_in_focal){
                                focal_list.update(existing_next->focal_handle); 
                            }

                        }else{
                            existing_next->copy(*next_node);
                            if (update_open){
                                open_list.increase(existing_next->open_handle);
                            }
                        }
                    }
                }
                delete next_node;
            }
        }
    }
    releaseNodes();
    delete &constrain_table;

    return path;
}

void SpaceTimeAStar::releaseNodes(){
    open_list.clear();
	focal_list.clear();
	for (auto node: allNodes_table){
		delete node;
    }
	allNodes_table.clear();
}
