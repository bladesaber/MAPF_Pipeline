#include "ConstraintSelector.h"
#include "BaseFunctions.h"

namespace SearchSpace
{
    void NearEndConstraintSelector::select_constraint(
        std::map<std::string, MathSpace::NetworkGraph> &graph_map, Constraint &constraint)
    {
        MathSpace::NetworkGraph graph;
        std::vector<LocXyz> terminals;
        LocXyz start_loc;
        bool is_group0;

        Eigen::Matrix<double, Eigen::Dynamic, 2> degree_mat(m_conflict_set.conflicts.size(), 2);

        for (const auto &cindexs : m_conflict_set.conflict_index_map)
        {
            // ------ access graph ------
            graph = graph_map[cindexs.first];

            // ------ access terminals ------
            terminals.clear();
            graph.get_terminal_nodes(terminals);

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat(terminals.size(), cindexs.second.size());
            for (size_t i = 0; i < cindexs.second.size(); i++)
            {
                if (std::get<1>(cindexs.second[i]))
                {
                    start_loc = m_conflict_set.conflicts[std::get<0>(cindexs.second[i])].loc0;
                }
                else
                {
                    start_loc = m_conflict_set.conflicts[std::get<0>(cindexs.second[i])].loc1;
                }
                for (size_t j = 0; j < terminals.size(); j++)
                {
                    mat(j, i) = graph.get_shortest_path_length(start_loc, terminals[j]);
                }
            }

            Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> indices(mat.rows(), mat.cols());
            MathSpace::row_asscend_rank(mat, indices);
            
            Eigen::Matrix<double, 1, Eigen::Dynamic> min_col_vec;
            MathSpace::min_col(indices, min_col_vec);

            for (size_t i = 0; i < cindexs.second.size(); i++)
            {
                if (std::get<1>(cindexs.second[i]))
                {
                    degree_mat(std::get<0>(cindexs.second[i]), 0) = min_col_vec[0, i];
                }
                else
                {
                    degree_mat(std::get<0>(cindexs.second[i]), 1) = min_col_vec[0, i];
                }
            }
        }

        Eigen::Matrix<double, Eigen::Dynamic, 1> sum_row_vec;
        MathSpace::sum_row(degree_mat, sum_row_vec);

        // ------ find its minimum index ------
        int min_index = MathSpace::min_col_index(sum_row_vec, 0);
        
        // ------ crossing constraint ------
        Conflict min_conflict = m_conflict_set.conflicts[min_index];
        constraint.run_group0 = min_conflict.group1;
        constraint.obs0_loc = min_conflict.loc0;
        constraint.obs0_radius = min_conflict.radius0;
        constraint.run_group1 = min_conflict.group0;
        constraint.obs1_loc = min_conflict.loc1;
        constraint.obs1_radius = min_conflict.radius1;
    }
}
