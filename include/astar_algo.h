//
// Created by admin123456 on 2024/5/31.
//

/*
 * 1. state_detector 加入方向，除了开始节点，中间节点作为开始点时方向为0，注意compute_straight_line时的0除错误
 * 2. 使用曲率半径作为参考
 * */

#ifndef MAPF_PIPELINE_A_STAR_SEARCHER_H
#define MAPF_PIPELINE_A_STAR_SEARCHER_H

#include "common.h"
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
            int num_of_conflicts, int straight_x, int straight_y, int straight_z
    ) : loc_flag(loc_flag), g_val(g_val), h_val(h_val), num_of_conflicts(num_of_conflicts),
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
            if (n1->get_f_val() == n2->get_f_val()) {
                if (n1->h_val == n2->h_val)
                    return rand() % 2;
                return n1->h_val >= n2->h_val;
            }
            return n1->get_f_val() > n2->get_f_val();
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

    inline double get_f_val() const { return g_val + h_val; }

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
    PathResult() {};

    ~PathResult() {};

    void update_result(Grid::StandardGridEnv &grid, const AStarNode *goal_node, double search_radius) {
        radius = search_radius;
        path_flags.clear();
        path_steps.clear();
        path_length = 0.0;

        int step = 0;
        double last_x, last_y, last_z, x, y, z, step_length;
        auto curr = goal_node;
        while (curr != nullptr) {
            if (step == 0) {
                tie(last_x, last_y, last_z) = grid.flag2xyz(curr->loc_flag);

            } else {
                tie(x, y, z) = grid.flag2xyz(curr->loc_flag);
                step_length = norm2_dist(last_x, last_y, last_z, x, y, z);
                path_length += step_length;
                last_x = x, last_y = y, last_z = z;
                path_steps.emplace_back(step_length);

            }
            step += 1;

            path_flags.emplace_back(curr->loc_flag);
            curr = curr->parent;
        }

        reverse(path_flags.begin(), path_flags.end());
        reverse(path_steps.begin(), path_steps.end());
        path_steps.insert(path_steps.begin(), mean(path_steps)); // 由于第1个节点无步长，因此估算1个
    }

    vector<size_t> &get_path_flags() {
        return path_flags;
    }

    void get_path_xyz(Grid::StandardGridEnv &grid, vector<CellXYZ> &path_details) {
        double x, y, z;
        for (size_t loc_flag: path_flags) {
            tie(x, y, z) = grid.flag2xyz(loc_flag);
            path_details.emplace_back(make_tuple(x, y, z));
        }
    }

    void get_path_xyzr(Grid::StandardGridEnv &grid, vector<CellXYZR> &path_set) {
        double x, y, z;
        for (size_t loc_flag: path_flags) {
            tie(x, y, z) = grid.flag2xyz(loc_flag);
            path_set.emplace_back(make_tuple(x, y, z, radius));
        }
    }

    void get_path_xyzrl(Grid::StandardGridEnv &grid, vector<CellXYZRL> &path_set) {
        double x, y, z;
        for (int i = 0; i < path_flags.size(); i++) {
            tie(x, y, z) = grid.flag2xyz(path_flags[i]);
            path_set.emplace_back(make_tuple(x, y, z, radius, path_steps[i]));
        }
    }

    double get_radius() { return radius; }

    double get_path_length() { return path_length; }

private:
    double radius;
    vector<size_t> path_flags;
    vector<double> path_steps;
    double path_length = -1;
};

class StandardAStarSolver {
public:
    StandardAStarSolver(
            Grid::StandardGridEnv &grid, CollisionDetector &obstacle_detector, CollisionDetector &dynamic_detector,
            Grid::StandardStateDetector state_detector
    ) : grid(grid), obstacle_detector(obstacle_detector), dynamic_detector(dynamic_detector),
        state_detector(state_detector) {}

    ~StandardAStarSolver() { reset(); }

    size_t num_expanded = 0;
    size_t num_generated = 0;
    double search_time_cost;

    void update_configuration(
            double pipe_radius, int x_step_scale, int y_step_scale, int z_step_scale,
            list<string> grid_expand_methods, bool use_theta_star, bool use_curvature_cost
    ) {
        radius = pipe_radius;
        x_scale = x_step_scale;
        y_scale = y_step_scale;
        z_scale = z_step_scale;
        expand_methods = grid_expand_methods;
        with_theta_star = use_theta_star;
        with_curvature_cost = use_curvature_cost;
    }

    bool find_path(PathResult &res_path, size_t max_iter);

    AStarNode *get_next_node(size_t neighbour_loc_flag, AStarNode *parent_node);

    double compute_h_cost(size_t loc_flag);

    double compute_moving_orient(AStarNode *parent, size_t next_loc_flag) {
        int parent_x, parent_y, parent_z;
        tie(parent_x, parent_y, parent_z) = grid.flag2grid(parent->loc_flag);

        int node_x, node_y, node_z;
        tie(node_x, node_y, node_z) = grid.flag2grid(next_loc_flag);

        int vec_x = node_x - parent_x;
        int vec_y = node_y - parent_y;
        int vec_z = node_z - parent_z;

        return line2line_cos(vec_x, vec_y, vec_z, parent->straight_x, parent->straight_y, parent->straight_z);
    }

    tuple<int, int, int, double> compute_moving_info(AStarNode *parent_node, size_t loc_flag);

    void reset() {
        open_list.clear();
        for (auto node: all_nodes_table) { delete node; }
        all_nodes_table.clear();
    }

private:
    double radius;
    int x_scale, y_scale, z_scale;
    list<string> expand_methods;

    bool with_theta_star = false;
    bool with_curvature_cost = false;

    Grid::StandardGridEnv &grid;
    CollisionDetector &obstacle_detector;
    CollisionDetector &dynamic_detector;
    Grid::StandardStateDetector state_detector;

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

#endif //MAPF_PIPELINE_A_STAR_SEARCHER_H
