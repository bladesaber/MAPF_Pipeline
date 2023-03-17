//
// Created by quan on 23-3-13.
//

#ifndef MAPF_PIPELINE_SPACETIMEASTAR_H
#define MAPF_PIPELINE_SPACETIMEASTAR_H

#include "iostream"
#include "common.h"
#include "instance.h"
#include "constraintTable.h"

class AStarNode
{
public:
    int location;
	int g_val;
	int h_val = 0;
    int timestep = 0;
	int num_of_conflicts = 0;
    bool in_openlist;

	AStarNode* parent;

    struct compare_node{
        // returns true if n1 > n2 (note -- this gives us *min*-heap).
        bool operator()(const AStarNode* n1, const AStarNode* n2) const
        {
            if (n1->g_val + n1->h_val == n2->g_val + n2->h_val)
            {
                if (n1->h_val == n2->h_val)
                    return rand() % 2;
                return n1->h_val >= n2->h_val;
            }
            return n1->g_val + n1->h_val >= n2->g_val + n2->h_val;
        }
    };

    struct secondary_compare_node
	{
		bool operator()(const AStarNode* n1, const AStarNode* n2) const // returns true if n1 > n2
		{
			if (n1->num_of_conflicts == n2->num_of_conflicts)
			{
				if (n1->g_val == n2->g_val)
				{
					return rand() % 2 == 0;
				}
				return n1->g_val <= n2->g_val;  // break ties towards larger g_vals
			}
			return n1->num_of_conflicts >= n2->num_of_conflicts;  // n1 > n2 if it has more conflicts
		}
	};

    // The following is used by for generating the hash value of a nodes
    struct NodeHasher
	{
		size_t operator()(const AStarNode* n) const
		{
			size_t loc_hash = std::hash<int>()(n->location);
			// size_t timestep_hash = std::hash<int>()(n->timestep);
			// return (loc_hash ^ (timestep_hash << 1));
            return loc_hash;
		}
	};

    // The following is used for checking whether two nodes are equal
	// we say that two nodes, s1 and s2, are equal if both are non-NULL and agree on the id and timestep
	struct eqnode
	{
		bool operator()(const AStarNode* s1, const AStarNode* s2) const
		{
			// return (s1 == s2) || 
            //     (s1 && s2 && s1->location == s2->location && s1->timestep == s2->timestep);
            return (s1 == s2) || (s1->location == s2->location);
		}
	};

    AStarNode():location(0), g_val(0), h_val(0), parent(nullptr), timestep(0), num_of_conflicts(0), in_openlist(false){};
    AStarNode(int loc, int g_val, int h_val, AStarNode* parent, int timestep, int num_of_conflicts, bool in_openlist):
        location(loc), g_val(g_val), h_val(h_val), parent(parent), timestep(timestep), num_of_conflicts(num_of_conflicts), in_openlist(in_openlist){};
    ~AStarNode(){};

    typedef boost::heap::pairing_heap<AStarNode*, boost::heap::compare<AStarNode::compare_node>>::handle_type Open_handle_t;
	typedef boost::heap::pairing_heap<AStarNode*, boost::heap::compare<AStarNode::secondary_compare_node> >::handle_type Focal_handle_t;
	Open_handle_t open_handle;
	Focal_handle_t focal_handle;

    inline double getFVal() const {
        return g_val + h_val;
    }

    void copy(const AStarNode& node){
        parent = node.parent;
        location = node.location;
        
        // special for skip-Astar
        // g_val = node.g_val;
        g_val = std::min(node.g_val, g_val);

        h_val = node.h_val;
        timestep = node.timestep;
        num_of_conflicts = node.num_of_conflicts;
    }

};

class SpaceTimeAStar{
public:
    SpaceTimeAStar(int agent_idx):agent_idx(agent_idx){};
    ~SpaceTimeAStar(){};

    int num_expanded = 0;
	int num_generated = 0;
    double runtime_build_CT = 0; // runtime of building constraint table
	double runtime_build_CAT = 0; // runtime of building conflict avoidance table
    double runtime_search = 0; // runtime of Astar search

    bool focus_optimal = false;
    double focus_w = 1.0;

    // 用于通道回避A-star
    double bandwith = 0.0;

    int getHeuristic(Instance& instance, int loc1, int loc2){
        return instance.getManhattanDistance(loc1, loc2);
    };
    int getHeuristic(Instance& instance, const std::pair<int, int>& loc1, const std::pair<int, int>& loc2){
        return instance.getManhattanDistance(loc1, loc2);
    };

    bool validMove(Instance& instance, ConstraintTable& constrain_table, int curr, int next) const;
    
    // template<typename Instanct_type, typename State_type>
    // Path findPath(
    //     std::map<int, Path>& paths,
    //     std::map<int, std::vector<Constraint>>& constraints,
    //     Instanct_type& instance, 
    //     const State_type& start_state, 
    //     const State_type& goal_state
    // );
    Path findPath(
        std::map<int, Path>& paths,
        std::map<int, std::vector<Constraint>>& constraints,
        Instance& instance, 
        const std::pair<int, int>& start_state, 
        const std::pair<int, int>& goal_state
    );
    Path findPath(
        std::map<int, Path>& paths,
        std::map<int, std::vector<Constraint>>& constraints,
        Instance3D& instance,
        const std::tuple<int, int, int>& start_state, 
        const std::tuple<int, int, int>& goal_state
    );

    void updatePath(const AStarNode* goal, Path& path);

    template<typename Instance_type>
    void debugPrint(const AStarNode* next_node, Instance_type& instance, std::string tag){
        std::cout << tag;
        instance.printCoordinate(next_node->location);
        std::cout << "->";
        instance.printCoordinate(next_node->parent->location);
        std::cout << std::endl;
    };

    void releaseNodes();

private:
    int agent_idx;

    typedef boost::heap::pairing_heap<AStarNode*, boost::heap::compare<AStarNode::compare_node>> Heap_open_t;
    typedef boost::heap::pairing_heap<AStarNode*, boost::heap::compare<AStarNode::secondary_compare_node> > Heap_focal_t;
    Heap_open_t open_list;
	Heap_focal_t focal_list;

    double min_f_val; // minimal f value in OPEN
	double lower_bound; // Threshold for FOCAL

    // define typedef for hash_map
	typedef boost::unordered_set<AStarNode*, AStarNode::NodeHasher, AStarNode::eqnode> hashtable_t;
	hashtable_t allNodes_table;

    void pushNode(AStarNode* node);
    AStarNode* popNode();
    void updateFocalList();

};

#endif //MAPF_PIPELINE_SPACETIMEASTAR_H
