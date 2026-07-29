#include "AstarSearch.h"

namespace SearchSpace
{
    // Define the direction offset arrays
    const std::array<int16_t, DirectionOffsets::NUM_DIRECTIONS> DirectionOffsets::offsetX = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0, 0, 0, 0,  0,  0,  1,  1,  1,  1, 1, 1,  1, 1, 1
    };
    const std::array<int16_t, DirectionOffsets::NUM_DIRECTIONS> DirectionOffsets::offsetY = {
        -1, -1, -1,  0,  0,  0,  1,  1,  1, -1, -1, -1, 0, 1, 1,  1,  0, -1, -1, -1,  0, 0, 0,  1, 1, 1
    };
    const std::array<int16_t, DirectionOffsets::NUM_DIRECTIONS> DirectionOffsets::offsetZ = {
        -1,  0,  1, -1,  0,  1,  -1, 0,  1, -1,  0,  1, 1, 1, 0, -1, -1, -1,  0,  1, -1, 0, 1, -1, 0, 1
    };

    template <typename HeapType>
    SearchSpace::SearchNode *AstarSearch<HeapType>::generate_next_node(
        GridSpace::SearchGrid *grid,
        SearchSpace::SearchNode *current, int16_t dx, int16_t dy, int16_t dz,
        const std::vector<LocXyz> &goals)
    {
        SearchSpace::SearchNode *successor = new SearchSpace::SearchNode(
            LocXyz{current->m_loc.x + dx, current->m_loc.y + dy, current->m_loc.z + dz}, current);
        successor->g_cost = current->g_cost + this->compute_euclidean_cost(current->m_loc, successor->m_loc);
        successor->f_cost = compute_final_cost(
            successor->g_cost, this->compute_euclidean_heuristic(successor->m_loc, goals)
            );

        if (m_with_constraint_table)
        {
            bool is_constrained;
            double constrained_distance;
            std::tie(is_constrained, constrained_distance) = grid->is_constrained(successor->m_loc, this->m_radius);
            successor->conflict_distance = current->conflict_distance + constrained_distance;
        }

        return successor;
    }

    template <typename HeapType>
    bool AstarSearch<HeapType>::search(
        GridSpace::SearchGrid *grid,
        const std::vector<LocXyz> &starts,
        const std::vector<LocXyz> &goals,
        PathData &path)
    {
        double time_start = omp_get_wtime();

        bool is_found = false;
        size_t iteration = 0;

        // Initialize the start node
        for (const auto &start : starts)
        {
            SearchSpace::SearchNode *node = new SearchSpace::SearchNode(LocXyz{start.x, start.y, start.z}, nullptr);
            node->g_cost = 0;
            node->f_cost = this->compute_euclidean_heuristic(node->m_loc, goals);
            m_open_list.push(node);
        }

        while (!m_open_list.is_empty() && iteration < m_max_iterations)
        {
            iteration++;
            SearchSpace::SearchNode *current = m_open_list.pop();

            // // --------------------------------------------
            // double distance = MathSpace::get_distance(current->m_loc, goals[0]);
            // bool debug = distance < 1.5;
            // if (debug)
            // {
            //     std::cout << "iteration: " << iteration << " goal: " << goals[0] 
            //         << " current: " << current->m_loc << " distance: " << distance << std::endl;
            // }
            // // --------------------------------------------

            if (is_goal(current, goals))
            {
                double time_end = omp_get_wtime();
                this->get_path(current, path, time_end - time_start);
                is_found = true;
                break;
            }

            this->before_expand(grid, current, goals);
            for (size_t i = 0; i < DirectionOffsets::NUM_DIRECTIONS; i++)
            {
                int16_t dx = DirectionOffsets::offsetX[i] * x_step_scale;
                int16_t dy = DirectionOffsets::offsetY[i] * y_step_scale;
                int16_t dz = DirectionOffsets::offsetZ[i] * z_step_scale;

                if (current->m_parent != nullptr && this->is_inverse_direction(
                                                        current->m_loc.x - current->m_parent->m_loc.x,
                                                        current->m_loc.y - current->m_parent->m_loc.y,
                                                        current->m_loc.z - current->m_parent->m_loc.z,
                                                        dx, dy, dz))
                {
                    continue;
                }

                // // --------------------------------------------
                // if (debug){
                //     std::cout << "dx: " << dx << " dy: " << dy << " dz: " << dz << std::endl;
                //     std::cout << "is_inverse_direction pass" << std::endl;
                //     std::cout << "current: " << current->m_loc + LocXyz{dx, dy, dz} << " radius: " << this->m_radius << std::endl;
                //     std::cout << "is_obstacle {dx, dy, dz}: " << std::get<0>(grid->is_obstacle(current->m_loc + LocXyz{dx, dy, dz}, this->m_radius)) << std::endl;
                //     std::cout << "dist2obstacle {dx, dy, dz}: " << std::get<1>(grid->is_obstacle(current->m_loc + LocXyz{dx, dy, dz}, this->m_radius)) << std::endl;
                //     // std::cout << "is_obstacle {dx, 0, 0}: " << std::get<0>(grid->is_obstacle(current->m_loc + LocXyz{dx, 0, 0}, this->m_radius)) << std::endl;
                //     // std::cout << "is_obstacle {0, dy, 0}: " << std::get<0>(grid->is_obstacle(current->m_loc + LocXyz{0, dy, 0}, this->m_radius)) << std::endl;
                //     // std::cout << "is_obstacle {0, 0, dz}: " << std::get<0>(grid->is_obstacle(current->m_loc + LocXyz{0, 0, dz}, this->m_radius)) << std::endl;
                // }
                // // --------------------------------------------

                if (!grid->is_move_valid(current->m_loc, this->m_radius, dx, dy, dz))
                {
                    continue;
                }

                // // --------------------------------------------
                // if (debug){
                //     std::cout << "is_move_valid pass" << std::endl;
                // }
                // // --------------------------------------------

                SearchSpace::SearchNode *successor = this->generate_next_node(grid, current, dx, dy, dz, goals);
                SearchSpace::SearchNode *existing_node = m_open_list.find_node(successor);

                if (existing_node == nullptr)
                {
                    m_open_list.push(successor);
                }
                else
                {
                    if (
                        successor->f_cost < existing_node->f_cost
                        || (m_with_constraint_table 
                            && successor->f_cost == existing_node->f_cost 
                            && successor->conflict_distance < existing_node->conflict_distance)
                        )
                    {
                        if (existing_node->m_status == SearchSpace::SearchNodeStatus::OnOpen)
                        {
                            existing_node->m_parent = current;
                            existing_node->g_cost = successor->g_cost;
                            existing_node->f_cost = successor->f_cost;
                            m_open_list.update(existing_node, successor->f_cost);
                        }
                        else
                        {
                            existing_node->m_parent = current;
                            existing_node->g_cost = successor->g_cost;
                            existing_node->f_cost = successor->f_cost;
                            m_open_list.push(existing_node);
                        }
                    }
                    delete successor;
                }
            }
        }

        reset();
        return is_found;
    }

    template <typename HeapType>
    SearchSpace::SearchNode *ThetaStarSearch<HeapType>::generate_next_node(
        GridSpace::SearchGrid *grid,
        SearchSpace::SearchNode *current, int16_t dx, int16_t dy, int16_t dz,
        const std::vector<LocXyz> &goals)
    {
        // Create the successor node
        LocXyz new_loc = {
            static_cast<int16_t>(current->m_loc.x + dx),
            static_cast<int16_t>(current->m_loc.y + dy),
            static_cast<int16_t>(current->m_loc.z + dz)};

        SearchSpace::SearchNode *successor = new SearchSpace::SearchNode(new_loc, current);

        // If current has a parent and there's line of sight from parent to successor,
        // update successor's parent to be current's parent (path smoothing)
        bool is_light_of_sight = false;
        if (current->m_parent != nullptr)
        {
            is_light_of_sight = grid->is_light_of_sight(current->m_parent->m_loc, new_loc, this->m_radius);
        }

        if (is_light_of_sight)
        {
            successor->m_parent = current->m_parent;
            successor->g_cost = current->m_parent->g_cost + this->compute_euclidean_cost(current->m_parent->m_loc, new_loc);
        }
        else
        {
            successor->m_parent = current;
            successor->g_cost = current->g_cost + this->compute_euclidean_cost(current->m_loc, new_loc);
        }
        successor->f_cost = this->compute_final_cost(successor->g_cost, this->compute_euclidean_heuristic(successor->m_loc, goals));

        if (this->m_with_constraint_table)
        {
            throw std::runtime_error("Theta* search does not support constraint table");
        }

        return successor;
    }

    template <typename HeapType>
    bool BiDirectionalAstarSearch<HeapType>::search(
        GridSpace::SearchGrid *grid,
        const std::vector<LocXyz> &starts,
        const std::vector<LocXyz> &goals,
        PathData &path)
    {
        double time_start = omp_get_wtime();

        bool is_found = false;
        size_t iteration = 0;

        // ------ initialize forward search open list ------
        for (const auto &start : starts)
        {
            SearchSpace::SearchNode *node = new SearchSpace::SearchNode(LocXyz{start.x, start.y, start.z}, nullptr);
            node->g_cost = 0;
            node->f_cost = this->compute_euclidean_heuristic(node->m_loc, goals);
            this->m_open_list.push(node);
        }

        // ------ initialize backward search open list ------
        for (const auto &goal : goals)
        {
            SearchSpace::SearchNode *node = new SearchSpace::SearchNode(LocXyz{goal.x, goal.y, goal.z}, nullptr);
            node->g_cost = 0;
            node->f_cost = this->compute_euclidean_heuristic(node->m_loc, starts);
            m_backward_list.push(node);
        }

        // ------ search ------
        while (!this->m_open_list.is_empty() && !m_backward_list.is_empty() && iteration < this->m_max_iterations)
        {
            iteration++;
            
            // ------ forward search ------
            SearchSpace::SearchNode *current_forward = this->m_open_list.pop();
            if (is_backward_found(current_forward))
            {
                double time_end = omp_get_wtime();
                this->get_path(current_forward, path, time_end - time_start);
                is_found = true;
                break;
            }

            // ------ backward search ------
            SearchSpace::SearchNode *current_backward = m_backward_list.pop();
            if (is_forward_found(current_backward))
            {
                double time_end = omp_get_wtime();
                this->get_path(current_backward, path, time_end - time_start);
                is_found = true;
                break;
            }

            // ------ expand forward search nodes ------
            for (size_t i = 0; i < DirectionOffsets::NUM_DIRECTIONS; i++)
            {
                int16_t dx = DirectionOffsets::offsetX[i] * this->x_step_scale;
                int16_t dy = DirectionOffsets::offsetY[i] * this->y_step_scale;
                int16_t dz = DirectionOffsets::offsetZ[i] * this->z_step_scale;

                // ------ expand forward search nodes ------
                if (current_forward->m_parent != nullptr && this->is_inverse_direction(
                                                                current_forward->m_loc.x - current_forward->m_parent->m_loc.x,
                                                                current_forward->m_loc.y - current_forward->m_parent->m_loc.y,
                                                                current_forward->m_loc.z - current_forward->m_parent->m_loc.z,
                                                                dx, dy, dz))
                {
                    continue;
                }

                if (!grid->is_move_valid(current_forward->m_loc, this->m_radius, dx, dy, dz)){
                    continue;
                }

                SearchSpace::SearchNode *successor_forward = this->generate_next_node(grid, current_forward, dx, dy, dz, goals);
                SearchSpace::SearchNode *existing_forward = this->m_open_list.find_node(successor_forward);
                if (existing_forward == nullptr)
                {
                    this->m_open_list.push(successor_forward);
                }
                else
                {
                    if (successor_forward->g_cost < existing_forward->g_cost)
                    {
                        if (existing_forward->m_status == SearchSpace::SearchNodeStatus::OnOpen)
                        {
                            existing_forward->m_parent = current_forward;
                            existing_forward->g_cost = successor_forward->g_cost;
                            existing_forward->f_cost = successor_forward->f_cost;
                            this->m_open_list.update(existing_forward, successor_forward->f_cost);
                        }
                        else
                        {
                            existing_forward->m_parent = current_forward;
                            existing_forward->g_cost = successor_forward->g_cost;
                            existing_forward->f_cost = successor_forward->f_cost;
                            this->m_open_list.push(existing_forward);
                        }
                    }
                    delete successor_forward;
                }
            }

            for (size_t i = 0; i < DirectionOffsets::NUM_DIRECTIONS; i++)
            {
                int16_t dx = DirectionOffsets::offsetX[i] * this->x_step_scale;
                int16_t dy = DirectionOffsets::offsetY[i] * this->y_step_scale;
                int16_t dz = DirectionOffsets::offsetZ[i] * this->z_step_scale;
                // ------ expand backward search nodes ------
                if (current_backward->m_parent != nullptr && this->is_inverse_direction(
                                                                 current_backward->m_loc.x - current_backward->m_parent->m_loc.x,
                                                                 current_backward->m_loc.y - current_backward->m_parent->m_loc.y,
                                                                 current_backward->m_loc.z - current_backward->m_parent->m_loc.z,
                                                                 dx, dy, dz))
                {
                    continue;
                }

                if (!grid->is_move_valid(current_backward->m_loc, this->m_radius, dx, dy, dz))
                {
                    continue;
                }

                SearchSpace::SearchNode *successor_backward = this->generate_next_node(grid, current_backward, dx, dy, dz, starts);
                SearchSpace::SearchNode *existing_backward = m_backward_list.find_node(successor_backward);
                if (existing_backward == nullptr)
                {
                    m_backward_list.push(successor_backward);
                }
                else
                {
                    if (successor_backward->g_cost < existing_backward->g_cost)
                    {
                        if (existing_backward->m_status == SearchSpace::SearchNodeStatus::OnOpen)
                        {
                            existing_backward->m_parent = current_backward;
                            existing_backward->g_cost = successor_backward->g_cost;
                            existing_backward->f_cost = successor_backward->f_cost;
                            m_backward_list.update(existing_backward, successor_backward->f_cost);
                        }
                        else
                        {
                            existing_backward->m_parent = current_backward;
                            existing_backward->g_cost = successor_backward->g_cost;
                            existing_backward->f_cost = successor_backward->f_cost;
                            m_backward_list.push(existing_backward);
                        }
                    }
                    delete successor_backward;
                }
            }
        }

        reset();
        return is_found;
    }

    template <typename HeapType>
    void BiDirectionalAstarSearch<HeapType>::get_path(SearchSpace::SearchNode *meeting_node, PathData &path_data, double time_cost)
    {
        std::vector<LocXyz> path;

        // Get forward path from start to meeting point
        std::vector<LocXyz> forward_path;
        SearchSpace::SearchNode *current;
        
        SearchSpace::SearchNode *forward_node = this->m_open_list.find_node(meeting_node);
        if (forward_node != nullptr){
            current = forward_node;
            while (current != nullptr)
            {
                forward_path.push_back(current->m_loc);
                current = current->m_parent;
            }
            std::reverse(forward_path.begin(), forward_path.end());
        }

        // Get backward path from goal to meeting point
        std::vector<LocXyz> backward_path;
        SearchSpace::SearchNode *backward_node = m_backward_list.find_node(meeting_node);
        if (backward_node != nullptr) {
            current = backward_node;
            while (current != nullptr)
            {
                backward_path.push_back(current->m_loc);
                current = current->m_parent;
            }
        }

        // Combine paths
        path = forward_path;
        if (!backward_path.empty()) {
            path.insert(path.end(), backward_path.begin() + 1, backward_path.end());
        }

        path_data.set_path(path, this->m_radius, time_cost);
    }

    // Explicit template instantiations
    template class AstarSearch<QueueSpace::BoostHeap>;
    template class AstarSearch<QueueSpace::BucketPriorityQueue>;
    template class WeightedAstarSearch<QueueSpace::BoostHeap>;
    template class WeightedAstarSearch<QueueSpace::BucketPriorityQueue>;
    template class ThetaStarSearch<QueueSpace::BoostHeap>;
    template class ThetaStarSearch<QueueSpace::BucketPriorityQueue>;
    template class BiDirectionalAstarSearch<QueueSpace::BoostHeap>;
    template class BiDirectionalAstarSearch<QueueSpace::BucketPriorityQueue>;
    template class ScaledWeightedAstarSearch<QueueSpace::BoostHeap>;
    template class ScaledWeightedAstarSearch<QueueSpace::BucketPriorityQueue>;
    template class DoubleScaledWeightedAstarSearch<QueueSpace::BoostHeap>;
    template class DoubleScaledWeightedAstarSearch<QueueSpace::BucketPriorityQueue>;
}