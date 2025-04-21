#ifndef ASTAR_SEARCH_H
#define ASTAR_SEARCH_H

#include "BaseHead.h"
#include "SearchGrid.h"
#include "SearchNode.h"
#include "SearchHeap.h"
#include "BaseFunctions.h"
#include <algorithm>
#include <climits>

namespace SearchSpace
{
    struct DirectionOffsets {
        static constexpr size_t NUM_DIRECTIONS = 26;
        static const std::array<int16_t, NUM_DIRECTIONS> offsetX;
        static const std::array<int16_t, NUM_DIRECTIONS> offsetY;
        static const std::array<int16_t, NUM_DIRECTIONS> offsetZ;
    };

    class BaseSearch
    {
    public:
        BaseSearch(double radius, size_t max_iterations = 100000) : m_radius(radius), m_max_iterations(max_iterations) {}
        virtual ~BaseSearch() {}

        virtual void reset() = 0;
        virtual bool search(
            GridSpace::SearchGrid *grid,
            const std::vector<LocXyz> &starts,
            const std::vector<LocXyz> &goals,
            std::vector<LocXyz> &path) = 0;
        virtual double compute_final_cost(double g_cost, double h_cost) = 0;

        void get_path(SearchSpace::SearchNode *node, std::vector<LocXyz> &path)
        {
            SearchSpace::SearchNode *current = node;
            while (current != nullptr)
            {
                path.push_back(LocXyz{current->m_loc});
                current = current->m_parent;
            }
            std::reverse(path.begin(), path.end());
        }

        bool is_goal(SearchSpace::SearchNode *node, const std::vector<LocXyz> &goals)
        {
            for (const auto &goal : goals)
            {
                if (node->m_loc == goal)
                {
                    return true;
                }
            }
            return false;
        }

        static inline double compute_manhattan_heuristic(LocXyz &loc, const std::vector<LocXyz> &goals)
        {
            double min_h = std::numeric_limits<double>::max();
            for (const auto &goal : goals)
            {
                double h = std::abs((loc.x) - goal.x) +
                           std::abs((loc.y) - goal.y) +
                           std::abs((loc.z) - goal.z);
                min_h = std::min(min_h, h);
            }
            return min_h;
        }

        static inline double compute_euclidean_heuristic(LocXyz &loc, const std::vector<LocXyz> &goals)
        {
            double min_h = std::numeric_limits<double>::max();
            for (const auto &goal : goals)
            {
                double h = std::sqrt(
                    std::pow(loc.x - goal.x, 2) +
                    std::pow(loc.y - goal.y, 2) +
                    std::pow(loc.z - goal.z, 2));
                min_h = std::min(min_h, h);
            }
            return min_h;
        }

        static inline double compute_euclidean_cost(LocXyz &loc1, LocXyz &loc2)
        {
            return std::sqrt(
                std::pow(loc1.x - loc2.x, 2) +
                std::pow(loc1.y - loc2.y, 2) +
                std::pow(loc1.z - loc2.z, 2));
        }

        inline void set_radius(double radius)
        {
            m_radius = radius;
        }

        inline bool is_inverse_direction(
            int16_t dx_parent, int16_t dy_parent, int16_t dz_parent, int16_t dx, int16_t dy, int16_t dz)
        {
            return MathSpace::compute_vector_angle(dx_parent, dy_parent, dz_parent, dx, dy, dz) < -1e-2;
            // return MathSpace::compute_vector_angle(dx_parent, dy_parent, dz_parent, dx, dy, dz) > 1e-2;
        }

        inline double get_radius() const
        {
            return m_radius;
        }

    protected:
        double m_radius;
        size_t m_max_iterations;

        virtual SearchSpace::SearchNode *generate_next_node(
            GridSpace::SearchGrid *grid, SearchSpace::SearchNode *current, int16_t dx, int16_t dy, int16_t dz,
            const std::vector<LocXyz> &goals) = 0;
        virtual void before_expand(
            GridSpace::SearchGrid *grid, SearchSpace::SearchNode *current, const std::vector<LocXyz> &goals) = 0;
    };

    template <typename HeapType>
    class AstarSearch : public BaseSearch
    {
        static_assert(std::is_base_of<QueueSpace::SearchHeap, HeapType>::value, "HeapType must inherit from SearchHeap");

    public:
        AstarSearch(double radius, size_t max_iterations = 100000, bool with_constraint_table = false) : 
            BaseSearch(radius, max_iterations), m_with_constraint_table(with_constraint_table) {}

        ~AstarSearch() override {}

        void reset() override { m_open_list.reset(); }
        bool search(
            GridSpace::SearchGrid *grid,
            const std::vector<LocXyz> &starts,
            const std::vector<LocXyz> &goals,
            std::vector<LocXyz> &path) override;
        inline double compute_final_cost(double g_cost, double h_cost) override { return g_cost + h_cost; }

    protected:
        HeapType m_open_list;
        bool m_with_constraint_table = false;

        uint32_t x_step_scale = 1;
        uint32_t y_step_scale = 1;
        uint32_t z_step_scale = 1;

        SearchSpace::SearchNode *generate_next_node(
            GridSpace::SearchGrid *grid, SearchSpace::SearchNode *current, int16_t dx, int16_t dy, int16_t dz,
            const std::vector<LocXyz> &goals) override;
        void before_expand(
            GridSpace::SearchGrid *grid, SearchSpace::SearchNode *current, const std::vector<LocXyz> &goals) override {}
    };

    template <typename HeapType>
    class WeightedAstarSearch : public AstarSearch<HeapType>
    {
    public:
        WeightedAstarSearch(
            double radius, size_t max_iterations, double g_weight, double h_weight, bool with_constraint_table = false)
            : AstarSearch<HeapType>(radius, max_iterations, with_constraint_table), 
            m_g_weight(g_weight), m_h_weight(h_weight) {}

        ~WeightedAstarSearch() = default;

        inline double compute_final_cost(double g_cost, double h_cost) { return g_cost * m_g_weight + h_cost * m_h_weight; }

    protected:
        double m_g_weight;
        double m_h_weight;
    };

    template <typename HeapType>
    class ThetaStarSearch : public WeightedAstarSearch<HeapType>
    {
    public:
        ThetaStarSearch(double radius, size_t max_iterations, double g_weight, double h_weight)
            : WeightedAstarSearch<HeapType>(radius, max_iterations, g_weight, h_weight, false) {}

        ~ThetaStarSearch() = default;

    protected:
        SearchSpace::SearchNode *generate_next_node(
            GridSpace::SearchGrid *grid,
            SearchSpace::SearchNode *current, int16_t dx, int16_t dy, int16_t dz,
            const std::vector<LocXyz> &goals);
    };

    template <typename HeapType>
    class ScaledWeightedAstarSearch : public WeightedAstarSearch<HeapType>
    {
    public:
        ScaledWeightedAstarSearch(
            double radius, size_t max_iterations, double g_weight, double h_weight,
            double threshold1, uint32_t scale_factor1, bool with_constraint_table = false)
            : WeightedAstarSearch<HeapType>(radius, max_iterations, g_weight, h_weight, with_constraint_table),
              m_threshold1(threshold1), m_scale_factor1(scale_factor1) {}

    protected:
        double m_threshold1;
        uint32_t m_scale_factor1;

        void before_expand(
            GridSpace::SearchGrid *grid, SearchSpace::SearchNode *current, const std::vector<LocXyz> &goals) override {
            double distance_to_goal = 0;
            for (const auto &goal : goals) {
                distance_to_goal = MathSpace::get_distance(current->m_loc, goal);
                if (distance_to_goal < m_threshold1) {
                    this->x_step_scale = 1;
                    this->y_step_scale = 1;
                    this->z_step_scale = 1;
                    return;
                }
            }

            double distance_to_obstacle = std::get<1>(grid->is_obstacle(current->m_loc, this->m_radius));
            if (distance_to_obstacle > m_threshold1) {
                this->x_step_scale = m_scale_factor1;
                this->y_step_scale = m_scale_factor1;
                this->z_step_scale = m_scale_factor1;
            } else {
                this->x_step_scale = 1;
                this->y_step_scale = 1;
                this->z_step_scale = 1;
            }
        }
    };

    template <typename HeapType>
    class DoubleScaledWeightedAstarSearch : public ScaledWeightedAstarSearch<HeapType>
    {
    public:
        DoubleScaledWeightedAstarSearch(
            double radius, size_t max_iterations, double g_weight, double h_weight, 
            double threshold1, uint32_t scale_factor1, double threshold2, uint32_t scale_factor2, 
            bool with_constraint_table = false)
            : ScaledWeightedAstarSearch<HeapType>(radius, max_iterations, g_weight, h_weight, 
            threshold1, scale_factor1, with_constraint_table),
              m_threshold2(threshold2), m_scale_factor2(scale_factor2) {
                assert(threshold1 > threshold2 && "threshold1 must be greater than threshold2");
            }

    protected:
        double m_threshold2;
        uint32_t m_scale_factor2;

        void before_expand(
            GridSpace::SearchGrid *grid, SearchSpace::SearchNode *current, const std::vector<LocXyz> &goals) override {
            double distance_to_goal = 0;
            for (const auto &goal : goals) {
                distance_to_goal = MathSpace::get_distance(current->m_loc, goal);
                if (distance_to_goal < m_threshold2) {
                    this->x_step_scale = 1;
                    this->y_step_scale = 1;
                    this->z_step_scale = 1;
                    return;
                }
            }

            double distance_to_obstacle = std::get<1>(grid->is_obstacle(current->m_loc, this->m_radius));
            if (distance_to_obstacle > this->m_threshold1)
            {
                this->x_step_scale = this->m_scale_factor1;
                this->y_step_scale = this->m_scale_factor1;
                this->z_step_scale = this->m_scale_factor1;
            }
            else if (distance_to_obstacle > m_threshold2)
            {
                this->x_step_scale = m_scale_factor2;
                this->y_step_scale = m_scale_factor2;
                this->z_step_scale = m_scale_factor2;
            }
            else
            {
                this->x_step_scale = 1;
                this->y_step_scale = 1;
                this->z_step_scale = 1;
            }
        }
    };

    template <typename HeapType>
    class BiDirectionalAstarSearch : public WeightedAstarSearch<HeapType>
    {
    public:
        BiDirectionalAstarSearch(double radius, size_t max_iterations, double g_weight, double h_weight)
            : WeightedAstarSearch<HeapType>(radius, max_iterations, g_weight, h_weight, false) {}

        ~BiDirectionalAstarSearch() = default;

        void reset() override
        {
            WeightedAstarSearch<HeapType>::reset();
            m_backward_list.reset();
        }

        void get_path(SearchSpace::SearchNode *meeting_node, std::vector<LocXyz> &path);

        bool search(
            GridSpace::SearchGrid *grid,
            const std::vector<LocXyz> &starts,
            const std::vector<LocXyz> &goals,
            std::vector<LocXyz> &path) override;

        inline bool is_forward_found(SearchSpace::SearchNode *node)
        {
            return this->m_open_list.contains(node);
        }

        inline bool is_backward_found(SearchSpace::SearchNode *node)
        {
            return m_backward_list.contains(node);
        }

    protected:
        HeapType m_backward_list;
    };
}

#endif