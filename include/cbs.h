#ifndef MAPF_PIPELINE_CBS_H
#define MAPF_PIPELINE_CBS_H

#include "common.h"
#include "instance.h"
#include "spaceTimeAstar.h"
#include "cbsNode.h"

class CBS{
public:
    CBS(const Instance& instance, int num_of_agents, 
        std::vector<std::pair<int, int>>& start_states,
        std::vector<std::pair<int, int>>& goal_states
    );
    ~CBS(){};

    int num_of_agents;
    Instance instance;
    std::vector<std::pair<int, int>> start_states;
    std::vector<std::pair<int, int>> goal_states;

    uint64_t num_HL_expanded = 0;
	uint64_t num_HL_generated = 0;
    uint64_t num_LL_expanded = 0;
	uint64_t num_LL_generated = 0;

    bool solve(double time_limit);

private:
    double time_limit;
    double min_f_val;
	double focal_list_threshold;
    double focal_w;

    std::vector<SpaceTimeAStar*> search_engines;

    CBSNode* generateRoot();

    boost::heap::pairing_heap< CBSNode*, boost::heap::compare<CBSNode::compare_node> > open_list;
	boost::heap::pairing_heap< CBSNode*, boost::heap::compare<CBSNode::secondary_compare_node> > focal_list;
	std::list<CBSNode*> allNodes_table;


};

#endif //MAPF_PIPELINE_CBS_H