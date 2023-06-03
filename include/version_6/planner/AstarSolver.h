#ifndef MAPF_PIPELINE_ASTARSOLVER_H
#define MAPF_PIPELINE_ASTARSOLVER_H

#include "common.h"
#include "instance.h"
#include "constrainTable.h"

namespace PlannerNameSpace{

typedef std::vector<size_t> Path;
typedef std::vector<std::tuple<double, double, double, double>> Path_XYZR;

class AStarNode
{
public:
    size_t location;
    size_t timestep = 0;
	int num_of_conflicts = 0;
    bool in_openlist;

    double g_val = 0;
	double h_val = 0;

    AStarNode* parent = nullptr;

    AStarNode():
        location(0), g_val(0), h_val(0), parent(nullptr), 
        timestep(0), num_of_conflicts(0), in_openlist(false)
    {};
    AStarNode(
        size_t loc, double g_val, double h_val, 
        AStarNode* parent, size_t timestep, 
        int num_of_conflicts, bool in_openlist
    ):location(loc), g_val(g_val), h_val(h_val), parent(parent), timestep(timestep), 
      num_of_conflicts(num_of_conflicts), in_openlist(in_openlist)
    {};
    
    ~AStarNode(){};

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
            return (s1 == s2) || (s1->location == s2->location);
		}
	};

    typedef boost::heap::pairing_heap<AStarNode*, boost::heap::compare<AStarNode::compare_node>>::handle_type Open_handle_t;
	Open_handle_t open_handle;

    inline double getFVal() const {
        return g_val + h_val;
    }

    void copy(const AStarNode& node){
        parent = node.parent;
        location = node.location;
        
        // TODO special for skip-Astar
        g_val = node.g_val;
        // g_val = std::min(node.g_val, g_val);

        h_val = node.h_val;
        timestep = node.timestep;
        num_of_conflicts = node.num_of_conflicts;
    }

};

class AStarSolver{
public:
    AStarSolver(bool with_AnyAngle, bool with_OrientCost):
        with_AnyAngle(with_AnyAngle), with_OrientCost(with_OrientCost)
        {};
    ~AStarSolver(){};

    int num_expanded = 0;
	int num_generated = 0;
    double runtime_build_CT = 0; // runtime of building constraint table
	// double runtime_build_CAT = 0; // runtime of building conflict avoidance table
    double runtime_search = 0; // runtime of Astar search

    bool with_OrientCost = true;
    bool with_AnyAngle = false;

    void updatePath(const AStarNode* goal_node, Path& path){
        auto curr = goal_node;
        while (curr != nullptr)
        {
            path.emplace_back(curr->location);
            curr = curr->parent;
        }
        std::reverse(path.begin(), path.end());
    }

    AStarNode* getNextNode(
        size_t neighbour_loc,
        AStarNode* lastNode, 
        ConstraintTable& constrain_table, 
        Instance& instance
    );

    Path findPath(
        double radius,
        std::vector<ConstrainType> constraints, Instance& instance,
        size_t start_loc, 
        std::vector<size_t>& goal_locs
    );

    bool validMove(Instance& instance, ConstraintTable& constrain_table, int curr, int next) const;

    double getHeuristic(Instance& instance, size_t loc, std::vector<size_t>& goal_locs){
        double min_heuristic = 999999.0;
        for (size_t goal_loc : goal_locs)
        {
            min_heuristic = std::min(min_heuristic, instance.getManhattanDistance(loc, goal_loc));
        }
        return min_heuristic;
    }

    double getCost(Instance& instance, AStarNode* cur_node, size_t next_loc);

    bool isGoal(size_t loc){
        for (size_t goal_loc : goal_locs){
            if (loc == goal_loc){
                return true;
            }
        }
        return false;
    }

private:
    // temporary Params
    double radius;
    size_t start_loc;
    std::vector<size_t> goal_locs;

    typedef boost::heap::pairing_heap<AStarNode*, boost::heap::compare<AStarNode::compare_node>> Heap_open_t;
    Heap_open_t open_list;

    typedef boost::unordered_set<AStarNode*, AStarNode::NodeHasher, AStarNode::eqnode> hashtable_t;
	hashtable_t allNodes_table;

    void pushNode(AStarNode* node){
        num_generated++;
        node->open_handle = open_list.push(node);
        node->in_openlist = true;
    }
    AStarNode* popNode(){
        AStarNode* node = open_list.top();
        open_list.pop();

        node->in_openlist = false;
        return node;
    }

    void releaseNodes(){
        open_list.clear();
        for (auto node: allNodes_table){
            delete node;
        }
        allNodes_table.clear();
    }
};

}

#endif