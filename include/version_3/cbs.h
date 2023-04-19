#ifndef MAPF_PIPELINE_CBS_H
#define MAPF_PIPELINE_CBS_H

#include "common.h"
#include "instance.h"
#include "utils.h"

class CBSNode{
public:
    CBSNode(){};
    ~CBSNode(){};

    double g_val = 0.0;
	double h_val = 0.0;
	int depth;

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