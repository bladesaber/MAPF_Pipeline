#ifndef MAPF_PIPELINE_CBS_H
#define MAPF_PIPELINE_CBS_H

#include "common.h"
#include "instance.h"
#include "utils.h"
#include "kdtreeWrapper.h"

class AgentInfo{
public:
    size_ut agentIdx;
    double radius;

    // Path path;
    DetailPath detailPath;

    std::vector<ConstrainType> constrains;
    KDTreeWrapper pathTree;

    bool isConflict = false;
    ConstrainType firstConflict;
    double firstConflictLength;

    std::map<size_ut, size_t> costMap;

    AgentInfo(){};
    AgentInfo(size_ut agentIdx, double radius):agentIdx(agentIdx), radius(radius){};

};

class CBSNode{
public:
    CBSNode(size_t num_of_agents):num_of_agents(num_of_agents){};
    ~CBSNode(){};

    size_t num_of_agents;

    std::map<size_ut, AgentInfo> agentMap;

    double g_val = 0.0;
	double h_val = 0.0;
	int depth;

    void updateAgentConflict(size_t agentIdx);
    void findAllAgentConflict();
    void updateFirstConflict(
        double x, double y, double z, 
        double radius, double length, AgentInfo* agent
    );

    double getHeuristics();

    struct compare_node 
	{
		bool operator()(const CBSNode* n1, const CBSNode* n2) const 
		{
			return n1->g_val + n1->h_val >= n2->g_val + n2->h_val;
		}
	};
    
    inline double getFVal() const{
        return g_val + h_val;
    }

    // typedef boost::heap::pairing_heap<CBSNode*, boost::heap::compare<CBSNode::compare_node>>::handle_type Open_handle_t;
    // Open_handle_t open_handle;

};

class CBS{
public:
    CBS(){};
    ~CBS(){};

    DetailPath sampleDetailPath(Path& path, Instance& instance, double stepLength);

    void pushNode(CBSNode* node);
    CBSNode* popNode();
    bool is_openList_empty(){
        return this->open_list.empty();
    }

private:
    boost::heap::pairing_heap< CBSNode*, boost::heap::compare<CBSNode::compare_node> > open_list;

    inline void releaseNodes(){
        open_list.clear();
    }
    
};

#endif