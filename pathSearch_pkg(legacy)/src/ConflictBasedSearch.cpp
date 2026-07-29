#include "ConflictBasedSearch.h"

namespace SearchSpace
{
    double ConflictBaseSearchNode::m_g_weight = 1.0;
    double ConflictBaseSearchNode::m_h_weight = 1.0;

    double ConflictBaseSearchNode::compute_heuristic_cost(ConflictSet &conflict_set)
    {
        double h_cost = 0;
        for (auto &iter : conflict_set.m_group_cost)
        {
            h_cost += iter.second;
        }
        return h_cost;
        
    }

    void ConflictBaseSearchNode::detect_conflict_between_groups(
        const CbsGroupData &group_data_i, 
        const CbsGroupData &group_data_j,
        std::vector<Conflict> &detected_conflicts)
    {
        std::vector<std::vector<int>> indices;
        std::vector<std::vector<double>> distances;

        group_data_i.tree.search_knn(group_data_j.nodes, 1, indices, distances);
        for (size_t j_index = 0; j_index < indices.size(); j_index++)
        {
            if (indices[j_index].size() == 0)
                continue;
            size_t i_index = indices[j_index][0];

            double distance = distances[j_index][0];
            double radius0 = group_data_i.tree.get_weight(i_index);
            double radius1 = group_data_j.tree.get_weight(j_index);

            // TODO: Very important, it should be consist with definition of conflict detection in SearchGrid::is_obstacle
            if (distance < radius0 + radius1)
            {
                Conflict conflict;
                conflict.group0 = group_data_i.name;
                conflict.loc0 = group_data_i.nodes[i_index];
                conflict.radius0 = radius0;
                conflict.group1 = group_data_j.name;
                conflict.loc1 = group_data_j.nodes[j_index];
                conflict.radius1 = radius1;
                conflict.conflict_cost = radius0 + radius1 - distance;
                detected_conflicts.push_back(conflict);
            }
        }
    }

    template <typename ConstraintSelectorType>
    bool CbsNode<ConstraintSelectorType>::find_division_constraint(
        ConflictSet &conflict_set, Constraint &selected_constraint)
    {        
        std::vector<std::string> names;
        names.reserve(result_map.size());
        for (const auto& result : result_map) {
            names.push_back(result.first);
        }

        // Current implementation rebuilds trees/graphs for every node
        std::unordered_map<std::string, CbsGroupData> group_data_map;

        // Pre-allocate map to avoid race conditions
        for (const auto& name : names) {
            group_data_map[name] = CbsGroupData();
            group_data_map[name].name = name;
        }

        // Process each group
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < names.size(); i++)
        {
            const std::string& name = names[i];
            std::vector<PathData> paths = result_map[name]->m_paths;

            // ------ Build graph ------
            for (auto &path : paths)
            {
                for (size_t j = 0; j < path.get_size() - 1; j++)
                {
                    double distance = MathSpace::get_distance(path.m_path[j], path.m_path[j + 1]);
                    group_data_map[name].graph.add_edge(path.m_path[j], path.m_path[j + 1], distance);
                }
            }

            // ------ Only consider the nodes on the path not the start and end nodes ------
            size_t tree_size = 0;
            for (auto &path : paths)
            {
                tree_size += path.get_size() - 2;
            }
            group_data_map[name].tree.init_tree(tree_size);

            // ------ Build tree and Save nodes ------
            size_t tree_index = 0;
            for (auto &path : paths)
            {
                for (size_t i = 1; i < path.get_size() - 1; i++)
                {
                    group_data_map[name].tree.add_leaf(tree_index, path.m_path[i], path.m_radius);
                    group_data_map[name].nodes.push_back(path.m_path[i]);
                    tree_index++;
                }
            }
            group_data_map[name].tree.build_tree();
        }

        // ------ Find conflicts ------
        int num_threads = std::min(omp_get_max_threads(), 4);
        omp_set_num_threads(num_threads);
        std::vector<std::vector<Conflict>> thread_local_conflicts(num_threads);
        #pragma omp parallel for
        for (size_t i = 0; i < names.size(); i++) {
            int thread_id = omp_get_thread_num();
            for (size_t j = i + 1; j < names.size(); j++) {
                detect_conflict_between_groups(
                    group_data_map[names[i]], 
                    group_data_map[names[j]],
                    thread_local_conflicts[thread_id]);
            }
        }

        // Merge results after parallel section
        for (const auto& local_conflicts : thread_local_conflicts) {
            for (auto &conflict : local_conflicts) {
                conflict_set.add_conflict(conflict);
            }
        }

        // ------ No Conflict ------
        if (conflict_set.m_conflict_size == 0)
        {
            return true;
        }

        // ------ Select constraint ------
        ConstraintSelectorType constraint_selector(conflict_set);

        std::map<std::string, MathSpace::NetworkGraph> graph_map;
        for (auto& iter: group_data_map)
        {
            graph_map[iter.first] = iter.second.graph;
        }
        constraint_selector.select_constraint(graph_map, selected_constraint);

        return false;
    }

    // Explicit template instantiations
    template class CbsNode<SearchSpace::NearEndConstraintSelector>;
    template class CbsSearch<SearchSpace::NearEndConstraintSelector>;
}
