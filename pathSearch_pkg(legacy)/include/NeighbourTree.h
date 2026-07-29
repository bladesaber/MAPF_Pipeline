#ifndef NEIGHBOUR_TREE_H
#define NEIGHBOUR_TREE_H

#include "BaseHead.h"
#include <flann/flann.hpp>

namespace MathSpace
{
    class KdTree
    {
        public:
        KdTree();
        ~KdTree();

        void init_tree(size_t tree_size);
        void add_leaf(const size_t &index, const LocXyz &loc);
        void build_tree();

        void search_knn(
            const LocXyz &loc, int k, std::vector<int> &indices, std::vector<double> &distances
        ) const;
        void search_knn(
            const std::vector<LocXyz> &locs, int k, 
            std::vector<std::vector<int>> &indices, std::vector<std::vector<double>> &distances
        ) const;

        void search_radius(
            const LocXyz &loc, double radius, 
            std::vector<int> &indices, std::vector<double> &distances
        ) const;
        void search_radius(
            const std::vector<LocXyz> &locs, double radius, 
            std::vector<std::vector<int>> &indices, std::vector<std::vector<double>> &distances
        ) const;

        double* get_row(const size_t &index) const {
            return data_.ptr() ? &data_[index][0] : nullptr;
        }
        
        protected:
        flann::Matrix<double> data_;  // FLANN data matrix
        flann::Index<flann::L2<double>>* index_;  // FLANN index
    };

    class WeightedKdTree : public KdTree
    {
        public:
        WeightedKdTree() = default;
        ~WeightedKdTree() = default;

        void init_tree(size_t tree_size);

        void add_leaf(const size_t &index, const LocXyz &loc, const double &weight);
        void get_weights(const std::vector<int> &indices, std::vector<double> &weights) const;
        void get_weights(const std::vector<std::vector<int>> &indices, std::vector<std::vector<double>> &weights) const;
        double get_weight(const size_t &index) const
        {
            return weights_[index];
        }
        
        protected:
        std::vector<double> weights_;  // Store weights
        flann::Matrix<double> data_;  // FLANN data matrix
        flann::Index<flann::L2<double>> *index_;  // FLANN index
    };
}

#endif
