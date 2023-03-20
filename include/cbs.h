#ifndef MAPF_PIPELINE_CBS_H
#define MAPF_PIPELINE_CBS_H

#include "common.h"
#include "instance.h"
#include "spaceTimeAstar.h"
#include "conflict.h"
#include "cbsNode.h"

class CBS{
public:
    CBS(int num_of_agents, Instance3D& instance, 
        std::map<int, std::tuple<int, int, int>>& start_states,
        std::map<int, std::tuple<int, int, int>>& goal_states
    ): instance(instance), start_states(start_states), goal_states(goal_states)
    {
        for (int agent_idx = 0; agent_idx < num_of_agents; agent_idx++)
        {
            search_engines[agent_idx] = new SpaceTimeAStar(agent_idx);
        }
    };
    ~CBS(){
        this->clear();
    };

    uint64_t num_HL_expanded = 0;
	uint64_t num_HL_generated = 0;
    uint64_t num_LL_expanded = 0;
	uint64_t num_LL_generated = 0;

    std::map<int, std::tuple<int, int, int>> start_states;
    std::map<int, std::tuple<int, int, int>> goal_states;

    bool focal_optimal = false;
    double time_limit;
    Instance3D& instance;

    void solvePath(CBSNode& node, int agent);

    // 由于为特定对象指针，pybind不会进行自动类型转换，因此可使用
    void updateFocalList();
    void pushNode(CBSNode* node);
    CBSNode* popNode();
    bool is_openList_empty(){
        return this->open_list.empty();
    }

    inline void clear(){
        this->clearSearchEngines();
        this->releaseNodes();
    };

    void print(){
        instance.print();
        for (auto iter = start_states.begin(); iter != start_states.end(); iter++){
            std::cout << "   Agent: " << iter->first << " Start(";
            std::cout << std::get<0>(iter->second) << ", " << std::get<1>(iter->second) << ", " << std::get<2>(iter->second) << ")";
            std::cout << std::endl;
        }
        for (auto iter = goal_states.begin(); iter != goal_states.end(); iter++){
            std::cout << "   Agent: " << iter->first << " Goal(";
            std::cout << std::get<0>(iter->second) << ", " << std::get<1>(iter->second) << ", " << std::get<2>(iter->second) << ")";
            std::cout << std::endl;
        }
    }

private:
    double min_f_val;
	double focal_list_threshold;
    double focal_w;

    std::map<int, SpaceTimeAStar*> search_engines;

    boost::heap::pairing_heap< CBSNode*, boost::heap::compare<CBSNode::compare_node> > open_list;
	boost::heap::pairing_heap< CBSNode*, boost::heap::compare<CBSNode::secondary_compare_node> > focal_list;
	std::list<CBSNode*> allNodes_table;

    void clearSearchEngines(){
        for (auto iter = this->search_engines.begin(); iter != search_engines.end(); iter++)
        {
            delete iter->second;
        }
	    search_engines.clear();
    };
    inline void releaseNodes(){
        open_list.clear();
        focal_list.clear();
        for (auto node : allNodes_table)
            delete node;
        allNodes_table.clear();
    }
    
};

#endif //MAPF_PIPELINE_CBS_H