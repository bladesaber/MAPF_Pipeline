#ifndef MAPF_PIPELINE_ANGLE_ASTAR_H
#define MAPF_PIPELINE_ANGLE_ASTAR_H

#include "common.h"
#include "instance.h"
#include "constrainTable.h"

class AStarNode
{
public:
    int location;
    int timestep = 0;
	int num_of_conflicts = 0;
    bool in_openlist;

    double g_val = 0;
	double h_val = 0;

    AStarNode* parent;

    AStarNode():location(0), g_val(0), h_val(0), parent(nullptr), timestep(0), num_of_conflicts(0), in_openlist(false){};
    AStarNode(int loc, double g_val, double h_val, AStarNode* parent, int timestep, int num_of_conflicts, bool in_openlist):
        location(loc), g_val(g_val), h_val(h_val), parent(parent), timestep(timestep), 
        num_of_conflicts(num_of_conflicts), in_openlist(in_openlist){};
    
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

class AngleAStar{
public:
    AngleAStar(double radius):radius(radius){};
    ~AngleAStar(){};

    double radius;

    int num_expanded = 0;
	int num_generated = 0;
    double runtime_build_CT = 0; // runtime of building constraint table
	// double runtime_build_CAT = 0; // runtime of building conflict avoidance table
    double runtime_search = 0; // runtime of Astar search

    double getHeuristic(Instance& instance, int loc1, int loc2);

    bool validMove(Instance& instance, ConstraintTable& constrain_table, int curr, int next) const;

    void updatePath(const AStarNode* goal, Path& path);

    AStarNode* getAnyAngleNode(
        int neighbour_loc, 
        int goal_loc,
        AStarNode* lastNode, 
        ConstraintTable& constrain_table, 
        Instance& instance
    );

    Path findPath(
        std::vector<ConstrainType> constraints,
        Instance& instance,
        const std::tuple<int, int, int>& start_state, 
        const std::tuple<int, int, int>& goal_state
    );

    void releaseNodes();

    void debugPrint(const AStarNode* next_node, Instance& instance, std::string tag){
        std::tuple<int, int, int> coodr = instance.getCoordinate(next_node->location);

        std::cout << tag;
        std::cout << next_node->location << ":(x:" << std::get<0>(coodr) << ", y:" << std::get<1>(coodr) << ", z:" << std::get<2>(coodr);
        std::cout << ", f:" << next_node->getFVal() << ", h:" << next_node->h_val << ", g:" << next_node->g_val << ")";
        std::cout << "<-";

        coodr = instance.getCoordinate(next_node->parent->location);
        std::cout << next_node->parent->location << ":(x:" << std::get<0>(coodr) << ", y:" << std::get<1>(coodr) << ", z:" << std::get<2>(coodr);
        std::cout << ", f:" << next_node->parent->getFVal() << ", h:" << next_node->parent->h_val << ", g:" << next_node->parent->g_val << ")";
        
        std::cout << std::endl;
    };

private:
    typedef boost::heap::pairing_heap<AStarNode*, boost::heap::compare<AStarNode::compare_node>> Heap_open_t;
    Heap_open_t open_list;

    typedef boost::unordered_set<AStarNode*, AStarNode::NodeHasher, AStarNode::eqnode> hashtable_t;
	hashtable_t allNodes_table;

    void pushNode(AStarNode* node);
    AStarNode* popNode();

};

#endif