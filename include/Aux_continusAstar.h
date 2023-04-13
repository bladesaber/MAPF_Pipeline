#ifndef MAPF_PIPELINE_AUX_CONTINUSASTAR_H
#define MAPF_PIPELINE_AUX_CONTINUSASTAR_H

#include "Aux_common.h"
#include "Aux_dubins.h"

class HybridAstarNode
{
public:
    HybridAstarNode(double x, double y, double z, double alpha, double beta, HybridAstarNode* parent=nullptr, bool in_openlist=false):
        x(x), y(y), z(z), alpha(alpha), beta(beta), in_openlist(in_openlist), parent(parent){};
    ~HybridAstarNode(){};

    double x;
    double y;
    double z;
    double alpha;
    double beta;

    int x_round;
    int y_round;
    int z_round;
    int alpha_round;
    int beta_round;

    double g_val;
    double h_val;
    int timestep;
    std::string hashTag;

    bool in_openlist;

    /*
    // store dubins path if exist
    std::tuple<DubinsPath, DubinsPath> dubins_solutions;
    bool invert_yz;
    std::vector<std::tuple<double, double, double>> dubinsPath3D;
    double dubinsLength3D;
    bool findValidDubinsPath = false;
    */

    HybridAstarNode* parent;
    std::string parentTag;

    void setRoundCoodr(int x, int y, int z, int alpha, int beta){
        this->x_round = x;
        this->y_round = y;
        this->z_round = z;
        this->alpha_round = alpha;
        this->beta_round = beta;

        this->hashTag = getHashTag();
    }

    void setRoundCoodr(std::tuple<int, int, int, int, int> pos){
        this->x_round = std::get<0>(pos);
        this->y_round = std::get<1>(pos);
        this->z_round = std::get<2>(pos);
        this->alpha_round = std::get<3>(pos);
        this->beta_round = std::get<4>(pos);

        this->hashTag = getHashTag();
    }

    std::tuple<double, double, double, double, double> getCoodr(){
        return std::make_tuple(x, y, z, alpha, beta);
    }
    
    std::tuple<int, int, int, int, int> getRoundCoodr(){
        return std::make_tuple(x_round, y_round, z_round, alpha_round, beta_round);
    }

    void copy(HybridAstarNode* rhs){
        x = rhs->x;
        y = rhs->y;
        z = rhs->z;
        alpha = rhs->alpha;
        beta = rhs->beta;
        g_val = rhs->g_val;
        h_val = rhs->h_val;
        parent = rhs->parent;
    }

    /*
    bool operator == (const HybridAstarNode& rhs) const{
        return x_round == rhs.x_round &&
               y_round == rhs.y_round &&
               z_round == rhs.z_round &&
               alpha_round == rhs.alpha_round &&
               beta_round == rhs.beta_round;
    }
    */

    struct compare_node{
        // returns true if n1 > n2 (note -- this gives us *min*-heap).
        bool operator()(const HybridAstarNode* n1, const HybridAstarNode* n2) const
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

    // Manage By Python
    // // The following is used by for generating the hash value of a nodes
    // struct NodeHasher
	// {
	// 	size_t operator()(const HybridAstarNode* n) const
	// 	{
    //         std::string s = "x:" + std::to_string(n->x_round);
    //         s += "y:" + std::to_string(n->y_round);
    //         s += "z:" + std::to_string(n->z_round);
    //         s += "alpha:" + std::to_string(n->alpha_round);
    //         s += "beta:" + std::to_string(n->beta_round);
    //
	// 		size_t pos_hash = std::hash<std::string>()(s);
    //         return pos_hash;
	// 	}
	// };
    //
    // // The following is used for checking whether two nodes are equal
	// // we say that two nodes, s1 and s2, are equal if both are non-NULL and agree on the id and timestep
	// struct eqnode
	// {
	// 	bool operator()(const HybridAstarNode* s1, const HybridAstarNode* s2) const
	// 	{
    //         return s1 == s2;
	// 	}
	// };

    std::string getHashTag() const{
        std::string s = "x(" + std::to_string(x_round) + ")";
        s += "y(" + std::to_string(y_round) + ")";
        s += "z(" + std::to_string(z_round) + ")";
        s += "alpha(" + std::to_string(alpha_round) + ")";
        s += "beta(" + std::to_string(beta_round) + ")";
        return s;
    }

    inline double getFVal() const {
        return g_val + h_val;
    }

    typedef boost::heap::pairing_heap<HybridAstarNode*, boost::heap::compare<HybridAstarNode::compare_node>>::handle_type Open_handle_t;
	Open_handle_t open_handle;

    bool equal(HybridAstarNode* rhs){
        return x_round == rhs->x_round &&
               y_round == rhs->y_round &&
               z_round == rhs->z_round &&
               alpha_round == rhs->alpha_round &&
               beta_round == rhs->beta_round;
    }

};

class HybridAstar
{
public:
    HybridAstar(){};
    ~HybridAstar(){};

    int num_expanded = 0;
	int num_generated = 0;
    // double runtime_build_CT = 0; // runtime of building constraint table
	// double runtime_build_CAT = 0; // runtime of building conflict avoidance table
    double runtime_search = 0; // runtime of Astar search

    void pushNode(HybridAstarNode* node);
    HybridAstarNode* popNode();

    bool is_openList_empty(){
        return this->open_list.empty();
    }

    void release(){
        open_list.clear();
    }

private:
    typedef boost::heap::pairing_heap<HybridAstarNode*, boost::heap::compare<HybridAstarNode::compare_node>> Heap_open_t;
    Heap_open_t open_list;

    // Manage By Python
    // typedef boost::unordered_set<HybridAstarNode*, HybridAstarNode::NodeHasher, HybridAstarNode::eqnode> hashtable_t;
    // hashtable_t allNodes_table;
};

#endif