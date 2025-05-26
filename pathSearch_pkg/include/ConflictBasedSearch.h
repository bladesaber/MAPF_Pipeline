#ifndef CONFLICT_BASED_SEARCH_H
#define CONFLICT_BASED_SEARCH_H

// #define DEBUG

#include "BaseHead.h"
#include "TaskManager.h"
#include "SearchGrid.h"
#include "SetPlanner.h"
#include "ConstraintSelector.h"
#include <boost/heap/pairing_heap.hpp>
#include <boost/unordered_set.hpp>

namespace SearchSpace
{
    class ConflictBaseSearchNode
    {
    protected:
        struct CbsGroupData
        {
            std::string name;
            MathSpace::WeightedKdTree tree;
            MathSpace::NetworkGraph graph;
            std::vector<LocXyz> nodes;
        };

    public:
        struct NodeHasher
        {
            size_t operator()(const ConflictBaseSearchNode *n) const
            {
                return static_cast<size_t>(n->m_id);
            }
        };

        struct NodeEqual
        {
            bool operator()(const ConflictBaseSearchNode *lhs, const ConflictBaseSearchNode *rhs) const
            {
                return (lhs->m_id == rhs->m_id);
            }
        };

        struct NodeCompare
        {
            // returns true if n1 > n2 (note -- this gives us *min*-heap).
            bool operator()(const ConflictBaseSearchNode *lhs, const ConflictBaseSearchNode *rhs) const
            {
                if (lhs->f_cost == rhs->f_cost)
                {
                    return (lhs->f_cost - lhs->g_cost) > (rhs->f_cost - rhs->g_cost);
                }
                return lhs->f_cost > rhs->f_cost;
            }
        };

        double g_cost;
        double f_cost;

        std::unordered_map<std::string, std::shared_ptr<PathSetData>> result_map;
        std::unordered_map<std::string, std::shared_ptr<ObstacleList>> dynamic_obstacle_map;

        ConflictBaseSearchNode(const int16_t id) : m_id(id) {}

        ~ConflictBaseSearchNode()
        {
            result_map.clear(); // share_ptr will automatically decrease their reference counts
            dynamic_obstacle_map.clear();
        }

        void add_dynamic_obstacle(std::string run_group, ObstacleData new_obstacle)
        {
            #ifdef DEBUG
            for (auto &obstacle : *dynamic_obstacle_map[run_group])
            {
                if (obstacle.loc == new_obstacle.loc && obstacle.radius == new_obstacle.radius)
                {
                    throw std::runtime_error("Cannot add same obstacle to the same group");
                }
            }
            #endif

            // ------ generate new obstacle list based on old obstacle list ------
            auto new_obstacle_list = std::make_shared<ObstacleList>(*dynamic_obstacle_map[run_group]); // Deep copy
            new_obstacle_list->push_back(new_obstacle);

            dynamic_obstacle_map[run_group] = nullptr;           // release old share_ptr obstacle list
            dynamic_obstacle_map[run_group] = new_obstacle_list; // save new share_ptr obstacle list
        }

        void reset_result(std::string run_group, PathSetData &new_result)
        {
            result_map[run_group] = nullptr;                                   // release old share_ptr result
            result_map[run_group] = std::make_shared<PathSetData>(new_result); // save new share_ptr result
        }

        void get_group_names(std::vector<std::string> &names)
        {
            for (auto &iter : result_map)
            {
                names.push_back(iter.first);
            }
        }

        void detect_conflict_between_groups(
            const CbsGroupData &group_data_i, 
            const CbsGroupData &group_data_j,
            std::vector<Conflict> &detected_conflicts);

        bool search(
            SearchSpace::SetPlanner &set_planner, const GridSpace::SearchGrid *grid, TaskSpace::TaskSet *task_set)
        {
            // ------ create new mixed grid set ------
            GridSpace::SearchGridSet grid_set(grid);

            // ------ set dynamic obstacles ------
            for (auto &obstacle : *dynamic_obstacle_map[task_set->name])
            {
                grid_set.set_obstacle(obstacle.loc, obstacle.radius);
            }

            // ------ set constraint table ------
            for (auto &iter : result_map)
            {
                if (iter.first != task_set->name)
                {
                    for (auto &path : iter.second->m_paths)
                    {
                        for (auto &loc : path.m_path)
                        {
                            grid_set.set_constraint(loc, path.m_radius);
                        }
                    }
                }
            }

            // ------ preprocess grid set ------
            grid_set.preprocess();

            // ------ set tasks ------
            set_planner.set_tasks(*task_set);

            // ------ search ------
            PathSetData new_result;
            bool is_solvable = set_planner.search(&grid_set, new_result);

            if (is_solvable)
            {
                new_result.estimate_cost();
                reset_result(task_set->name, new_result);

                std::vector<std::string> group_names;
                get_group_names(group_names);
                m_conflict_set = ConflictSet(group_names);

                is_free_conflict = this->find_division_constraint(m_conflict_set, m_constraint);

                g_cost = 0;
                for (auto &iter : result_map)
                {
                    g_cost += iter.second->m_cost;
                }
                f_cost = g_cost * m_g_weight + compute_heuristic_cost(m_conflict_set) * m_h_weight;

                m_conflict_set.clear(); // Only for low memory usage
            }
            return is_solvable;
        }

        double compute_heuristic_cost(ConflictSet &conflict_set);

        virtual bool find_division_constraint(ConflictSet &conflict_set, Constraint &selected_constraint) = 0;

        boost::heap::pairing_heap<
            ConflictBaseSearchNode *,
            boost::heap::compare<ConflictBaseSearchNode::NodeCompare>>::handle_type m_openHandle;

        static void set_weights(double g_weight, double h_weight)
        {
            m_g_weight = g_weight;
            m_h_weight = h_weight;
        }

        bool is_free_conflict = true;
        ConflictSet m_conflict_set;
        Constraint m_constraint;

    protected:
        int16_t m_id;
        static double m_g_weight;
        static double m_h_weight;
    };

    class ConflictBaseSearch
    {
    public:
        ConflictBaseSearch(
            GridSpace::SearchGrid *grid, const double g_weight = 1.0, const double h_weight = 1.0, 
            const int16_t max_iteration = 10000) : m_grid(grid), m_max_iteration(max_iteration)
        {
            ConflictBaseSearchNode::set_weights(g_weight, h_weight);
        }

        ~ConflictBaseSearch()
        {
            reset();
        }

        void add_task_set(TaskSpace::TaskSet *task_set, SearchSpace::SetPlanner *planner)
        {
            m_task_set_map[task_set->name] = task_set;
            m_planner_map[task_set->name] = planner;
        }

        bool init_root_node(const std::map<std::string, std::vector<ObstacleData>> &dynamic_obstacle_map, CbsResult *result)
        {
            m_node_count = 0;
            ConflictBaseSearchNode *root_node = create_node();

            // ------ create dynamic obstacle map ------
            for (auto &iter : m_task_set_map)
            {
                root_node->dynamic_obstacle_map[iter.first] = std::make_shared<ObstacleList>(ObstacleList());
                if (dynamic_obstacle_map.find(iter.first) != dynamic_obstacle_map.end()) {
                    for (const auto &obstacle : dynamic_obstacle_map.at(iter.first))
                    {
                        root_node->dynamic_obstacle_map[iter.first]->push_back(obstacle);
                    }
                }
            }

            // ------ search all task sets ------
            for (auto &iter : m_task_set_map)
            {
                if (!root_node->search(*m_planner_map[iter.first], m_grid, iter.second))
                {
                    return false;
                }
            }

            // ------ if all task sets are solvable, push root node to open heap ------
            push_node(root_node);

            // ------ add result to result ------
            for (auto &iter : root_node->result_map){
                result->add_result(iter.first, *iter.second);
            }

            return true;
        }

        bool extend_node(
            ConflictBaseSearchNode *reference_node,
            std::string run_group,
            const LocXyz &obs_loc,
            const double obs_radius)
        {
            ConflictBaseSearchNode *new_node = create_node();

            // ------ share all results from reference node ------
            new_node->result_map = reference_node->result_map;
            new_node->dynamic_obstacle_map = reference_node->dynamic_obstacle_map;

            // ------ create new obstacle data ------
            ObstacleData new_obstacle(obs_loc, obs_radius);
            new_node->add_dynamic_obstacle(run_group, new_obstacle);

            if (new_node->search(*m_planner_map[run_group], m_grid, m_task_set_map[run_group]))
            {
                // if success, push node to open heap
                push_node(new_node);
                return true;
            }
            else
            {
                // If search failed no result, delete the node
                delete new_node;
                return false;
            }
        }

        bool search(CbsResult *result)
        {
            if (!result) return false;
            bool is_solvable = false;
            int16_t iteration = 0;
            while (!m_open_heap.empty() && iteration < m_max_iteration)
            {
                iteration++;
                ConflictBaseSearchNode *node = pop_node();

                if (node->is_free_conflict)
                {
                    for (auto &iter : node->result_map)
                    {
                        result->add_result(iter.first, *iter.second);
                    }
                    is_solvable = true;
                    break;
                }
                else
                {
                    // extend node 1
                    extend_node(node, node->m_constraint.run_group0, node->m_constraint.obs0_loc, node->m_constraint.obs0_radius);

                    // extend node 2
                    extend_node(node, node->m_constraint.run_group1, node->m_constraint.obs1_loc, node->m_constraint.obs1_radius);
                }
                delete node;
            }

            reset();

            #ifdef DEBUG
            std::cout << "Finish search at iteration: " << iteration << std::endl;
            #endif

            return is_solvable;
        }
        
        // ------ Just for python binding ------
        bool single_step_search(CbsResult *result)
        {
            result->reset();
            if (!m_open_heap.empty())
            {
                ConflictBaseSearchNode *node = pop_node();

                for (auto &iter : node->result_map){
                    result->add_result(iter.first, *iter.second);
                    result->m_conflict_size = node->m_conflict_set.m_conflict_size;
                }

                if (node->is_free_conflict){
                    return true;
                }
                else
                {
                    // extend node 1
                    extend_node(node, node->m_constraint.run_group0, node->m_constraint.obs0_loc, node->m_constraint.obs0_radius);
                    
                    // extend node 2
                    extend_node(node, node->m_constraint.run_group1, node->m_constraint.obs1_loc, node->m_constraint.obs1_radius);
                }
                delete node;
            }
            return false;
        }

        bool is_empty()
        {
            return m_open_heap.empty();
        }

        // ------ End of Just for python binding ------

    protected:
        GridSpace::SearchGrid *m_grid;

        boost::heap::pairing_heap<
            ConflictBaseSearchNode *,
            boost::heap::compare<ConflictBaseSearchNode::NodeCompare>>
            m_open_heap;

        uint16_t m_node_count = 0;
        int16_t m_max_iteration;

        std::map<std::string, TaskSpace::TaskSet *> m_task_set_map;
        std::map<std::string, SearchSpace::SetPlanner *> m_planner_map;

        void reset()
        {
            while (!m_open_heap.empty())
            {
                ConflictBaseSearchNode *node = m_open_heap.top();
                m_open_heap.pop();
                delete node;
            }
        }

        void push_node(ConflictBaseSearchNode *node)
        {
            node->m_openHandle = m_open_heap.push(node);
        }

        ConflictBaseSearchNode *pop_node()
        {
            ConflictBaseSearchNode *node = m_open_heap.top();
            m_open_heap.pop();
            return node;
        }

        virtual ConflictBaseSearchNode *create_node() = 0;
    };

    template <typename ConstraintSelectorType>
    class CbsNode : public ConflictBaseSearchNode
    {
        static_assert(std::is_base_of<ConstraintSelector, ConstraintSelectorType>::value, "ConstraintSelectorType must inherit from ConstraintSelector");

    public:
        CbsNode(const int16_t iteration) : ConflictBaseSearchNode(iteration) {}

        bool find_division_constraint(ConflictSet &conflict_set, Constraint &selected_constraint) override;
    };

    template <typename ConstraintSelectorType>
    class CbsSearch : public ConflictBaseSearch
    {
        static_assert(std::is_base_of<ConstraintSelector, ConstraintSelectorType>::value, "ConstraintSelectorType must inherit from ConstraintSelector");

    public:
        CbsSearch(
            GridSpace::SearchGrid *grid, const double g_weight = 1.0, const double h_weight = 1.0, const int16_t max_iteration = 10000) : ConflictBaseSearch(grid, g_weight, h_weight, max_iteration) {}

    protected:
        ConflictBaseSearchNode *create_node() override
        {
            return new CbsNode<ConstraintSelectorType>(m_node_count++);
        }
    };
}

#endif
