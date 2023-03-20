#ifndef MAPF_PIPELINE_CBSNODE_H
#define MAPF_PIPELINE_CBSNODE_H

#include "common.h"

class CBSNode{
public:
    CBSNode(){};

    ~CBSNode(){
		this->clear();
	};

	// agent, Constrains
    std::map<int, std::vector<Constraint>> constraints;
    // agent, Path
	std::map<int, Path> paths;

    int g_val;
	int h_val = 0;
	int depth; // depth of this CT node
	int makespan = 0; // makespan over all paths
    int tie_breaking = 0;
	// CBSNode* parent;

	double runtime_build_CT = 0; // runtime of building constraint table
	double runtime_build_CAT = 0; // runtime of building conflict avoidance table
    double runtime_search = 0; // runtime of Astar search

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

	inline double getFVal() const{
        return g_val + h_val;
    }
	
	void updateConstraint(int agent, const std::vector<Constraint>& constraint){
		// this->constraints.erase(agent);
		this->constraints[agent] = constraint;
	}

	void insertConstraint(int agent, const Constraint& constraint){
		this->constraints[agent].emplace_back(constraint);
	}

	void updatePath(int agent, const Path& path){
		// this->paths.erase(agent);
		this->paths[agent] = path;
	}

	void updateMakespan(int agent = -1){
		if (agent < 0)
		{
			for(auto iter = paths.begin(); iter != paths.end(); iter++){
				this->makespan = std::max(this->makespan, (int)iter->second.size());
			}
		}else{
			this->makespan = std::max(this->makespan, (int)paths[agent].size());
		}
	}

	void updateGval(){
		g_val = 0;
		for(auto iter = paths.begin(); iter != paths.end(); iter++){
			g_val += (int)iter->second.size();
		}
	}

	void copy(CBSNode& other_node){
		this->constraints = std::map<int, std::vector<Constraint>>(other_node.constraints);
		this->paths = std::map<int, Path>(other_node.paths);
		this->g_val = other_node.g_val;
		this->h_val = other_node.h_val;
		this->makespan = other_node.makespan;
	}

	Path getPath(int agent){
		return paths[agent];
	}

	void clear(){
		// todo 这里我不太清楚是否进行释放 ??
		this->constraints.clear();
		this->paths.clear();
	}

};

#endif //MAPF_PIPELINE_CBSNODE_H