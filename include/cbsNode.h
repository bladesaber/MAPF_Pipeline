//
// Created by quan on 23-3-14.
//

#ifndef MAPF_PIPELINE_CBSNODE_H
#define MAPF_PIPELINE_CBSNODE_H

#include "conflict.h"

class CBSNode{
public:
    CBSNode(){};
    ~CBSNode(){};

    CBSNode* parent;
    std::map<int, std::vector<Constraint>> constraints;
    std::map<int, Path> paths;

    int g_val;
	int h_val;
	int depth; // depth of this CT node
	int makespan = 0; // makespan over all paths
    int tie_breaking = 0;

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

};

#endif //MAPF_PIPELINE_CBSNODE_H
