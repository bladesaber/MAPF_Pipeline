//
// Created by admin123456 on 2024/5/31.
//

#ifndef MAPF_PIPELINE_A_STAR_ALGO_H
#define MAPF_PIPELINE_A_STAR_ALGO_H

#include "common.h"
#include "constraint_avoid_table.h"
#include "collision_utils.h"
#include "grid_utils.h"
#include "state_utils.h"

using namespace std;

class AStarNode {
public:
    size_t loc_flag;
    size_t timestep = 0;
    int num_of_conflicts = 0;
    bool in_openlist;

    double g_val = 0;
    double h_val = 0;
    int straight_x, straight_y, straight_z;

    AStarNode *parent = nullptr;

    AStarNode(
            size_t loc_flag, double g_val, double h_val,
            int straight_x, int straight_y, int straight_z
    ) : loc_flag(loc_flag), g_val(g_val), h_val(h_val), num_of_conflicts(0),
        straight_x(straight_x), straight_y(straight_y), straight_z(straight_z),
        parent(nullptr), timestep(0), in_openlist(false) {};

    AStarNode(
            size_t loc_flag, double g_val, double h_val, AStarNode *parent, size_t timestep,
            int num_of_conflicts, bool in_openlist, int straight_x, int straight_y, int straight_z
    ) : loc_flag(loc_flag), g_val(g_val), h_val(h_val), parent(parent), timestep(timestep),
        num_of_conflicts(num_of_conflicts), in_openlist(in_openlist),
        straight_x(straight_x), straight_y(straight_y), straight_z(straight_z) {};

    ~AStarNode() {};

    struct compare_node {
        // returns true if n1 > n2 (note -- this gives us *min*-heap).
        bool operator()(const AStarNode *n1, const AStarNode *n2) const {
            if (n1->g_val + n1->h_val == n2->g_val + n2->h_val) {
                if (n1->h_val == n2->h_val)
                    return rand() % 2;
                return n1->h_val >= n2->h_val;
            }
            return n1->g_val + n1->h_val > n2->g_val + n2->h_val;
        }
    };

    struct NodeHasher {
        size_t operator()(const AStarNode *n) const {
            // return loc_hash = std::hash<int>()(n->location);
            return n->loc_flag;
        }
    };

    struct equal_node {
        bool operator()(const AStarNode *s1, const AStarNode *s2) const {
            return (s1 == s2) || (s1->loc_flag == s2->loc_flag);
        }
    };

    typedef boost::heap::pairing_heap<AStarNode *, boost::heap::compare<AStarNode::compare_node>>::handle_type Open_handle_t;
    Open_handle_t open_handle;

    inline double get_f_val() const {
        return g_val + h_val;
    }

    void copy(const AStarNode &node) {
        parent = node.parent;
        loc_flag = node.loc_flag;
        g_val = node.g_val;
        h_val = node.h_val;
        timestep = node.timestep;
        num_of_conflicts = node.num_of_conflicts;
        straight_x = node.straight_x;
        straight_y = node.straight_y;
        straight_z = node.straight_z;
    }
};

class PathResult {
public:
    PathResult() {}

    PathResult(Grid::DiscreteGridEnv *grid) : grid(grid) {};

    ~PathResult() {};

    void update_result(const AStarNode *goal_node, double search_radius);

    vector<size_t> get_path_flags() const { return path_flags; }

    void get_path_xyz(vector<CellXYZ> &path_set) const;

    void get_path_xyzr(vector<CellXYZR> &path_set) const;

    void get_path_xyzrl(vector<CellXYZRL> &path_set) const;

    void get_path_grid(set<size_t> &path_set) const;

    double get_radius() const { return radius; }

    double get_length() const { return path_length; }

    vector<double> get_step_length() const { return path_step_lengths; }

    vector<CellXYZR> get_path() const {
        vector<CellXYZR> path;
        double x, y, z;
        for (size_t loc_flag: path_flags) {
            tie(x, y, z) = grid->flag2xyz(loc_flag);
            path.emplace_back(make_tuple(x, y, z, radius));
        }
        return path;
    }

private:
    double radius;
    vector<size_t> path_flags;
    vector<double> path_step_lengths;
    double path_length = -1;

    Grid::DiscreteGridEnv *grid;
};


class StandardAStarSolver {
public:
    StandardAStarSolver(
            Grid::DiscreteGridEnv &grid, CollisionDetector &obstacle_detector, CollisionDetector &dynamic_detector,
            Grid::DynamicStepStateDetector &state_detector
    ) : grid(grid), obstacle_detector(obstacle_detector), dynamic_detector(dynamic_detector),
        state_detector(state_detector) {}

    ~StandardAStarSolver() {
        reset();
    }

    size_t num_expanded = 0;
    size_t num_generated = 0;
    double search_time_cost = -1.0;

    void update_configuration(
            double pipe_radius, int search_step_scale, vector<tuple<int, int, int>> &grid_expand_candidates,
            bool use_curvature_cost, double curvature_cost_weight,
            bool use_avoid_table=true, bool use_theta_star=false
    ) {
        radius = pipe_radius;
        step_scale = search_step_scale;
        expand_candidates.clear();
        expand_candidates.assign(grid_expand_candidates.begin(), grid_expand_candidates.end());
        with_curvature_cost = use_curvature_cost;
        curvature_weight = curvature_cost_weight;
        use_constraint_avoid_table = use_avoid_table;
        with_theta_star = use_theta_star;
        if (use_constraint_avoid_table && with_theta_star){
            with_theta_star = false;
            cout << "[INFO]: Since using constraint avoid table, theta star must be set to false." << endl;
        }
    }

    bool find_path(PathResult &res_path, size_t max_iter, const ConflictAvoidTable *avoid_table);

    AStarNode *get_next_node(size_t neighbour_loc_flag, AStarNode *parent_node);

    double compute_h_cost(size_t loc_flag);

    double compute_move_orient(AStarNode *parent, size_t next_loc_flag);

    int compute_num_of_conflicts(size_t loc0_flag, size_t loc1_flag);

    tuple<int, int, int, double> compute_move_info(AStarNode *parent_node, size_t loc_flag);

    void reset() {
        open_list.clear();
        for (auto node: all_nodes_table) { delete node; }
        all_nodes_table.clear();
    }

    Grid::DiscreteGridEnv *get_grid() const { return &grid; }

    CollisionDetector *get_obstacle_detector() const { return &obstacle_detector; }

    CollisionDetector *get_dynamic_detector() const { return &dynamic_detector; }

    Grid::DynamicStepStateDetector *get_state_detector() const { return &state_detector; }

private:
    double radius;
    int step_scale;
    int current_scale;
    vector<tuple<int, int, int>> expand_candidates;
    bool use_constraint_avoid_table;

    bool with_theta_star = false;
    bool with_curvature_cost = false;
    double curvature_weight = 5.0;

    Grid::DiscreteGridEnv &grid;
    CollisionDetector &obstacle_detector;
    CollisionDetector &dynamic_detector;
    Grid::DynamicStepStateDetector &state_detector;
    const ConflictAvoidTable *conflict_avoid_table;

    typedef boost::heap::pairing_heap<AStarNode *, boost::heap::compare<AStarNode::compare_node>> Heap_open_t;
    Heap_open_t open_list;

    typedef boost::unordered_set<AStarNode *, AStarNode::NodeHasher, AStarNode::equal_node> hashtable_t;
    hashtable_t all_nodes_table;

    void pushNode(AStarNode *node) {
        num_generated++;
        node->open_handle = open_list.push(node);
        node->in_openlist = true;
    }

    AStarNode *popNode() {
        AStarNode *node = open_list.top();
        open_list.pop();
        node->in_openlist = false;
        return node;
    }
};

#endif //MAPF_PIPELINE_A_STAR_ALGO_H
