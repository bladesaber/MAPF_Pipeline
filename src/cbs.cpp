#include "cbs.h"

void CBS::updateFocalList(){
    if (focal_optimal)
    {
        CBSNode* open_head = open_list.top();
        if (open_head->getFVal() > min_f_val)
        {
            min_f_val = open_head->getFVal();
            double new_focal_list_threshold = min_f_val * focal_w;
            for (auto n : open_list){
                // 避免无效遍历
                if(n->getFVal() > new_focal_list_threshold){
                    break;
                }

                // 之前未在 Focal list 内
                if (n->getFVal() > focal_list_threshold){
                    n->focal_handle = focal_list.push(n);
                }
            }

            focal_list_threshold = new_focal_list_threshold;
        }
    }   
}

void CBS::pushNode(CBSNode* node){
    node->open_handle = open_list.push(node);
	num_HL_generated++;
    
    if (focal_optimal)
    {
        if (node->getFVal() <= focal_list_threshold)
        {
            node->focal_handle = focal_list.push(node);
        }
    }
    // 交由Python管理
    // allNodes_table.push_back(node);
}

CBSNode* CBS::popNode(){
    CBSNode* node;
    if (focal_optimal)
    {
        node = focal_list.top();
		focal_list.pop();
		open_list.erase(node->open_handle);

    }else{
        node = open_list.top();
        open_list.pop();
    }
    return node;
}

void CBS::solvePath(CBSNode& node, int agent){
    Path path = search_engines[agent]->findPath(
        node.paths, 
        node.constraints, this->instance, 
        this->start_states[agent], 
        this->goal_states[agent]
    );
    node.updatePath(agent, path);
    node.runtime_build_CT = search_engines[agent]->runtime_build_CT;
    node.runtime_build_CAT = search_engines[agent]->runtime_build_CAT;
    node.runtime_search = search_engines[agent]->runtime_search;
}

// Conflict* CBS::findConflicts(CBSNode& node){
//     std::map<size_t, int> path_map;
    
//     std::vector<int> agent_list(num_of_agents);
// 	for (int i = 0; i < num_of_agents; i++)
// 	{
// 		agent_list[i] = i;
// 	}
//     // std::random_device rd;
// 	// std::mt19937 g(rd());
// 	// std::shuffle(std::begin(agents), std::end(agents), g);
//     std::random_shuffle(agent_list.begin(), agent_list.end());

//     for (int agent_idx : agent_list){
//         for(size_t loc : node.paths[agent_idx]){
//             auto it = path_map.find(loc);
//             if (it != path_map.end())
//             {
//                 int other_agent_idx = it->second;
//                 Conflict* conflict = new Conflict();
//                 conflict->vertexConflict(agent_idx, other_agent_idx, loc);
//                 return conflict;
//             }

//             path_map[loc] = agent_idx;
//         }
//     }
//     return nullptr;
// }

// CBSNode* CBS::generateRoot(){
//     CBSNode* start_node = new CBSNode();
//     start_node->g_val = 0;

//     for (int agent_idx = 0; agent_idx < num_of_agents; agent_idx++)
//     {   
//         start_node->paths[agent_idx] = search_engines[agent_idx]->findPath(
//             *start_node, instance, start_states[agent_idx], goal_states[agent_idx]
//         );
//         if (start_node->paths[agent_idx].empty())
//         {
//             std::cout << "No path exists for agent " << agent_idx << std::endl;
// 			return nullptr;
//         }
//         int path_size = start_node->paths[agent_idx].size();
//         start_node->makespan = std::max(start_node->makespan, path_size);
//         start_node->g_val += path_size;
//         num_LL_expanded += search_engines[agent_idx]->num_expanded;
// 		num_LL_generated += search_engines[agent_idx]->num_generated;
//     }
    
//     start_node->h_val = 0;
// 	start_node->depth = 0;

//     num_HL_generated++;
// 	// start_node->time_generated = num_HL_generated;
//     start_node->curr_conflict = findConflicts(*start_node);

//     min_f_val = (double) start_node->g_val;
// 	focal_list_threshold = min_f_val * focal_w;

//     return start_node;
// }

// bool CBS::generateChild(CBSNode* node, CBSNode* parent){
//     // node->parent = parent;
// 	node->g_val = parent->g_val;
// 	node->makespan = parent->makespan;
// 	node->depth = parent->depth + 1;

//     std::map<int, Path> paths(parent->paths);
//     node->paths = paths;

//     std::map<int, std::vector<Constraint>> constraints(parent->constraints);
//     node->constraints = constraints;

//     // todo unfinish

// }