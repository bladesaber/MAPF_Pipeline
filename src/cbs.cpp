#include "cbs.h"

CBS::CBS(const Instance& instance, int num_of_agents, 
        std::vector<std::pair<int, int>>& start_states,
        std::vector<std::pair<int, int>>& goal_states
    ):num_of_agents(num_of_agents), instance(instance), start_states(start_states), goal_states(goal_states)
{
    for (int i = 0; i < num_of_agents; i++)
	{
		search_engines[i] = new SpaceTimeAStar(i);
	}
}

bool CBS::solve(double time_limit){
    this->time_limit = time_limit;

    clock_t start = clock();

}

CBSNode* CBS::generateRoot(){
    CBSNode* start_node = new CBSNode();
    start_node->g_val = 0;

    for (int agent_idx = 0; agent_idx < num_of_agents; agent_idx++)
    {   
        start_node->paths[agent_idx] = search_engines[agent_idx]->findPath(
            *start_node, instance, start_states[agent_idx], goal_states[agent_idx]
        );
        if (start_node->paths[agent_idx].empty())
        {
            std::cout << "No path exists for agent " << agent_idx << std::endl;
			return nullptr;
        }
        int path_size = start_node->paths[agent_idx].size();
        start_node->makespan = std::max(start_node->makespan, path_size);
        start_node->g_val += path_size;
        num_LL_expanded += search_engines[agent_idx]->num_expanded;
		num_LL_generated += search_engines[agent_idx]->num_generated;
    }
    
    start_node->h_val = 0;
	start_node->depth = 0;
	start_node->open_handle = open_list.push(start_node);
	start_node->focal_handle = focal_list.push(start_node);

    num_HL_generated++;
	// start_node->time_generated = num_HL_generated;
	allNodes_table.push_back(start_node);

    min_f_val = std::max(min_f_val, (double) start_node->g_val);
	focal_list_threshold = min_f_val * focal_w;

    return start_node;
}