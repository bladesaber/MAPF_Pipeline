//
// Created by quan on 23-3-13.
//

#include "spaceTimeAstar.h"

bool SpaceTimeAStar::validMove(Instance& instance, ConstraintTable& constrain_table, int curr, int next) const {
    if(constrain_table.isConstrained(next)){
        return false;
    }
    return true;
}

void SpaceTimeAStar::pushNode(AStarNode* node){
    num_generated++;

    node->open_handle = open_list.push(node);
	node->in_openlist = true;

    if (this->focus_optimal)
    {
        if (node->getFVal() <= lower_bound)
        {
            node->focal_handle = focal_list.push(node);
        }   
    }
}

AStarNode* SpaceTimeAStar::popNode(){
    AStarNode* node;
    if (focus_optimal)
    {
        node = focal_list.top();
        focal_list.pop();
	    open_list.erase(node->open_handle);

    }else{
        node = open_list.top();
        open_list.pop();
    }

	node->in_openlist = false;
    return node;
}

void SpaceTimeAStar::updateFocalList(){
    if (this->focus_optimal)
    {
        auto open_head = open_list.top();
        if (open_head->getFVal() > min_f_val)
        {
		    double new_min_f_val = open_head->getFVal();
		    double new_lower_bound = std::max(lower_bound, new_min_f_val * focus_w);
            for (auto n : open_list)
            {
                // 避免无效遍历
                if(n->getFVal() > new_lower_bound){
                    break;
                }

                // 之前未在 Focal list 内
                if (n->getFVal() > lower_bound){
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
        path[curr->timestep] = curr->location;
        curr = curr->parent;
    }
}

// template<typename Instanct_type, typename State_type>
Path SpaceTimeAStar::findPath(
    std::map<int, Path>& paths,
    std::map<int, std::vector<Constraint>>& constraints,
    Instance& instance,
    const std::pair<int, int>& start_state, 
    const std::pair<int, int>& goal_state
    // Instanct_type& instance, 
    // const State_type& start_state, 
    // const State_type& goal_state
){
    num_expanded = 0;
	num_generated = 0;

    // build constraint table
    // if use new please delete personally
    // ConstraintTable* constrain_table_ptr = new ConstraintTable();
    // ConstraintTable constrain_table = *constrain_table_ptr;
    ConstraintTable constrain_table = ConstraintTable();

	auto starrt_time = clock();
    auto it = constraints.find(agent_idx);
    if (it != constraints.end())
    {
        constrain_table.insertConstrains2CT(it->second);
    }
    runtime_build_CT = (double) (clock() - starrt_time) / CLOCKS_PER_SEC;

    // build conflict avoid table
    starrt_time = clock();
    for (auto it = paths.begin(); it != paths.end(); it++) {
        if (it->first != agent_idx){
            constrain_table.insertPath2CAT(it->second);
        }
    }
    runtime_build_CAT = (double) (clock() - starrt_time) / CLOCKS_PER_SEC;

    // state: (row(y), col(x))
    int start_loc = instance.linearizeCoordinate(start_state);
    int goal_loc = instance.linearizeCoordinate(goal_state);

    auto start_node = new AStarNode(
        start_loc,  // location
		0,  // g val
		getHeuristic(instance, start_loc, goal_loc),  // h val
		nullptr,  // parent
		0,  // timestep
		0, // num_of_conflicts
	    false // in_openlist
    );

    min_f_val = start_node->getFVal();
    lower_bound = min_f_val * focus_w;
    pushNode(start_node);
	allNodes_table.insert(start_node);

    starrt_time = clock();
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
        
        num_expanded++;
        for (int neighbour_loc : instance.getNeighbors(curr->location))
        {
            bool valid_move = this->validMove(instance, constrain_table, curr->location, neighbour_loc);
            if (valid_move)
            {
                int next_timestep = curr->timestep + 1;
                int next_g_val = curr->g_val + 1;
                int next_h_val = getHeuristic(instance, neighbour_loc, goal_loc);
                int next_internal_conflicts = curr->num_of_conflicts + constrain_table.getNumOfConflictsForStep(curr->location, neighbour_loc);

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
                    // ------ debug
                    // debugPrint(next_node, instance, "Next node:");
                    // --------

                    pushNode(next_node);
                    allNodes_table.insert(next_node);
                    continue;
                }
                
                AStarNode* existing_next = *it;
                if (
                    next_node->getFVal() < existing_next->getFVal() ||
                    (
                        next_node->getFVal() <= existing_next->getFVal() + bandwith && 
                        next_node->num_of_conflicts < existing_next->num_of_conflicts
                    )
                )
                {
                    if (!existing_next->in_openlist) // if its in the closed list (reopen)
                    {
                        
                        // ------ debug
                        // debugPrint(next_node, instance, "Reopen node:");
                        // ------

                        existing_next->copy(*next_node);
                        pushNode(existing_next);

                    }else{
                        bool update_open = false;
                        double existing_next_f_val = existing_next->getFVal();
                        if (existing_next_f_val > next_g_val + next_h_val){
                            update_open = true;
                        }

                        existing_next->copy(*next_node);

                        if (focus_optimal)
                        {
                            // check if it was above the focal bound before and now below (thus need to be inserted)
                            bool add_to_focal = false;  
                            // check if it was inside the focal and needs to be updated (because f-val changed)
                            bool update_in_focal = false;
                            
                            if ((next_g_val + next_h_val) <= lower_bound){
                                if (existing_next_f_val > lower_bound){
                                    add_to_focal = true;
                                }else{
                                    update_in_focal = true;
                                }
                            }
                            
                            if (add_to_focal){
                                existing_next->focal_handle = focal_list.push(existing_next);
                            }else if (update_in_focal){
                                focal_list.update(existing_next->focal_handle); 
                            }
                        }

                        if (update_open){
                            open_list.increase(existing_next->open_handle);
                        }

                        // ------ debug
                        // debugPrint(next_node, instance, "Update node:");
                        // ------

                    }
                }
                delete next_node;
            }
        }
    }
    releaseNodes();
    // delete constrain_table_ptr;

    runtime_search = (double) (clock() - starrt_time) / CLOCKS_PER_SEC;

    return path;
}

Path SpaceTimeAStar::findPath(
    std::map<int, Path>& paths,
    std::map<int, std::vector<Constraint>>& constraints,
    Instance3D& instance,
    const std::tuple<int, int, int>& start_state, 
    const std::tuple<int, int, int>& goal_state
){
    num_expanded = 0;
	num_generated = 0;

    // build constraint table
    // if use new please delete personally
    // ConstraintTable* constrain_table_ptr = new ConstraintTable();
    // ConstraintTable constrain_table = *constrain_table_ptr;
    ConstraintTable constrain_table = ConstraintTable();

	auto starrt_time = clock();
    auto it = constraints.find(agent_idx);
    if (it != constraints.end())
    {
        constrain_table.insertConstrains2CT(it->second);
    }
    runtime_build_CT = (double) (clock() - starrt_time) / CLOCKS_PER_SEC;

    // build conflict avoid table
    starrt_time = clock();
    for (auto it = paths.begin(); it != paths.end(); it++) {
        if (it->first != agent_idx){
            constrain_table.insertPath2CAT(it->second);
        }
    }
    runtime_build_CAT = (double) (clock() - starrt_time) / CLOCKS_PER_SEC;

    // State: (row(y), col(x), z)
    int start_loc = instance.linearizeCoordinate(start_state);
    int goal_loc = instance.linearizeCoordinate(goal_state);

    auto start_node = new AStarNode(
        start_loc,  // location
		0,  // g val
		getHeuristic(instance, start_loc, goal_loc),  // h val
		nullptr,  // parent
		0,  // timestep
		0, // num_of_conflicts
	    false // in_openlist
    );

    min_f_val = start_node->getFVal();
    lower_bound = min_f_val * focus_w;
    pushNode(start_node);
	allNodes_table.insert(start_node);

    starrt_time = clock();
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
        
        num_expanded++;
        for (int neighbour_loc : instance.getNeighbors(curr->location))
        {
            bool valid_move = this->validMove(instance, constrain_table, curr->location, neighbour_loc);
            if (valid_move)
            {
                int next_timestep = curr->timestep + 1;
                int next_g_val = curr->g_val + 1;
                int next_h_val = getHeuristic(instance, neighbour_loc, goal_loc);
                int next_internal_conflicts = curr->num_of_conflicts + constrain_table.getNumOfConflictsForStep(curr->location, neighbour_loc);

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

                    // ------ debug
                    // debugPrint(next_node, instance, "Next node:");
                    // --------

                    continue;
                }
                
                AStarNode* existing_next = *it;
                if (
                    next_node->getFVal() < existing_next->getFVal() || 
                    (
                        next_node->getFVal() <= existing_next->getFVal() + bandwith && 
                        next_node->num_of_conflicts < existing_next->num_of_conflicts
                    )
                )
                {
                    if (!existing_next->in_openlist) // if its in the closed list (reopen)
                    {
                        existing_next->copy(*next_node);
                        pushNode(existing_next);

                        // ------ debug
                        // debugPrint(next_node, instance, "Reopen node:");
                        // ------

                    }else{
                        bool update_open = false;
                        double existing_next_f_val = existing_next->getFVal();
                        if (existing_next_f_val > next_g_val + next_h_val){
                            update_open = true;
                        }

                        existing_next->copy(*next_node);

                        if (focus_optimal)
                        {
                            // check if it was above the focal bound before and now below (thus need to be inserted)
                            bool add_to_focal = false;  
                            // check if it was inside the focal and needs to be updated (because f-val changed)
                            bool update_in_focal = false;
                            
                            if ((next_g_val + next_h_val) <= lower_bound){
                                if (existing_next_f_val > lower_bound){
                                    add_to_focal = true;
                                }else{
                                    update_in_focal = true;
                                }
                            }
                            
                            if (add_to_focal){
                                existing_next->focal_handle = focal_list.push(existing_next);
                            }else if (update_in_focal){
                                focal_list.update(existing_next->focal_handle); 
                            }
                        }

                        if (update_open){
                            open_list.increase(existing_next->open_handle);
                        }

                        // ------ debug
                        // debugPrint(next_node, instance, "updateOpen node:");
                        // ------

                    }
                }
                delete next_node;
            }
        }

    }

    releaseNodes();
    // delete constrain_table_ptr;

    runtime_search = (double) (clock() - starrt_time) / CLOCKS_PER_SEC;

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
