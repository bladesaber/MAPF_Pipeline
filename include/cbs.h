#ifndef MAPF_PIPELINE_CBS_H
#define MAPF_PIPELINE_CBS_H

#include "common.h"
#include "instance.h"
#include "spaceTimeAstar.h"
#include "conflict.h"

class CBSNode{
public:
    CBSNode(){};
    ~CBSNode(){};

	// agent, Constrains
    std::map<int, std::vector<Constraint>> constraints;
    // agent, Path
	std::map<int, Path> paths;
	Conflict* curr_conflict;

    int g_val;
	int h_val;
	int depth; // depth of this CT node
	int makespan = 0; // makespan over all paths
    int tie_breaking = 0;
	// CBSNode* parent;

    struct compare_node 
	{
		bool operator()(const CBSNode* n1, const CBSNode* n2) const 
		{
			return n1->g_val + n1->h_val >= n2->g_val + n2->h_val;
		}
	};
    struct secondary_compare_node 
	{
		bool operator()(const CBSNode* n1, const CBSNode* n2) const 
		{
			if (n1->tie_breaking == n2->tie_breaking)
				return rand() % 2;
			return n1->tie_breaking >= n2->tie_breaking;
		}
	};

    typedef boost::heap::pairing_heap<CBSNode*, boost::heap::compare<CBSNode::compare_node>>::handle_type Open_handle_t;
	typedef boost::heap::pairing_heap<CBSNode*, boost::heap::compare<CBSNode::secondary_compare_node>>::handle_type Focal_handle_t;
    Open_handle_t open_handle;
	Focal_handle_t focal_handle;

	inline double getFVal() const {
        return g_val + h_val;
    }
};


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

    bool focal_optimal = false;
    bool solve(double time_limit);

private:
    double time_limit;
    double min_f_val;
	double focal_list_threshold;
    double focal_w;
    bool solution_found = false;

    std::vector<SpaceTimeAStar*> search_engines;

    CBSNode* generateRoot();
    Conflict* findConflicts(CBSNode& node);
    bool generateChild(CBSNode* node, CBSNode* parent);

    boost::heap::pairing_heap< CBSNode*, boost::heap::compare<CBSNode::compare_node> > open_list;
	boost::heap::pairing_heap< CBSNode*, boost::heap::compare<CBSNode::secondary_compare_node> > focal_list;
	std::list<CBSNode*> allNodes_table;

    void updateFocalList();
    void pushNode(CBSNode* node);
    CBSNode* popNode();
};

#endif //MAPF_PIPELINE_CBS_H