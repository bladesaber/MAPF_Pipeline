#include "NeighbourTree.h"

namespace MathSpace
{
    KdTree::KdTree()
        : data_(nullptr, 0, 3)  // Initialize empty FLANN matrix
        , index_(nullptr)
    {
    }

    KdTree::~KdTree()
    {
        if (index_) {
            delete index_;
            index_ = nullptr;
        }
        if (data_.ptr() != 0) {
            delete[] data_.ptr();
            data_ = flann::Matrix<double>(nullptr, 0, 3);
        }
    }

    void KdTree::init_tree(size_t tree_size)
    {
        // Clean up existing resources
        if (index_) {
            delete index_;
            index_ = nullptr;
        }
        
        // Allocate new data
        double* data_ptr = new double[tree_size * 3];
        
        // If we have existing data, delete it after allocating new data
        // This ensures we don't leave the object in an invalid state if allocation fails
        if (data_.ptr() != 0) {
            delete[] data_.ptr();
        }
        
        // Assign new data
        data_ = flann::Matrix<double>(data_ptr, tree_size, 3);
    }
    
    void KdTree::add_leaf(const size_t &index, const LocXyz &loc)
    {
        data_[index][0] = loc.x;
        data_[index][1] = loc.y;
        data_[index][2] = loc.z;
    }

    void KdTree::build_tree()
    {
        index_ = new flann::Index<flann::L2<double>>(data_, flann::KDTreeSingleIndexParams());
        index_->buildIndex();
    }
    
    void KdTree::search_knn(const LocXyz &loc, int k, std::vector<int> &indices, std::vector<double> &distances) const
    {
        if (!index_) {
            throw std::runtime_error("Index not built");
        }
        double query_data[3] = {
            static_cast<double>(loc.x),
            static_cast<double>(loc.y), 
            static_cast<double>(loc.z)
        };
        flann::Matrix<double> query(query_data, 1, 3); // 1x3 matrix

        std::vector<std::vector<int>> indices_vec(1);
        std::vector<std::vector<double>> dists_vec(1);
        
        index_->knnSearch(query, indices_vec, dists_vec, k, flann::SearchParams(128));
        indices = indices_vec[0];
        distances = dists_vec[0];
        
        // Convert squared distances to actual Euclidean distances
        for (auto& dist : distances) {
            dist = std::sqrt(dist);
        }
    }
    
    void KdTree::search_knn(
        const std::vector<LocXyz> &locs, int k, 
        std::vector<std::vector<int>> &indices, std::vector<std::vector<double>> &distances
        ) const
    {
        if (!index_) {
            throw std::runtime_error("Index not built");
        }
        size_t n_points = locs.size();
        std::vector<double> query_data(n_points * 3);
        
        for (size_t i = 0; i < n_points; ++i) {
            query_data[i*3] = static_cast<double>(locs[i].x);
            query_data[i*3+1] = static_cast<double>(locs[i].y);
            query_data[i*3+2] = static_cast<double>(locs[i].z);
        }
        flann::Matrix<double> query(query_data.data(), n_points, 3);

        std::vector<std::vector<int>> indices_vec(n_points);
        std::vector<std::vector<double>> dists_vec(n_points);
        index_->knnSearch(query, indices_vec, dists_vec, k, flann::SearchParams(128));

        indices = indices_vec;
        distances = dists_vec;
        
        // Convert squared distances to actual Euclidean distances
        for (auto& dist_vec : distances) {
            for (auto& dist : dist_vec) {
                dist = std::sqrt(dist);
            }
        }
    }
    
    void KdTree::search_radius(
        const LocXyz &loc, double radius, 
        std::vector<int> &indices, std::vector<double> &distances
        ) const
    {
        if (!index_) {
            throw std::runtime_error("Index not built");
        }
        // search objects which distance is less than or equal to radius
        radius += 1e-2;

        double query_data[3] = {static_cast<double>(loc.x), 
                               static_cast<double>(loc.y), 
                               static_cast<double>(loc.z)};
        flann::Matrix<double> query(query_data, 1, 3);

        std::vector<std::vector<int>> indices_vec(1);
        std::vector<std::vector<double>> dists_vec(1);
        index_->radiusSearch(query, indices_vec, dists_vec, radius * radius, flann::SearchParams(128));

        indices = indices_vec[0];
        distances = dists_vec[0];

        // Convert squared distances to actual Euclidean distances
        for (auto& dist : distances) {
            dist = std::sqrt(dist);
        }
    }

    void KdTree::search_radius(
        const std::vector<LocXyz> &locs, double radius, 
        std::vector<std::vector<int>> &indices, std::vector<std::vector<double>> &distances
        ) const
    {
        if (!index_) {
            throw std::runtime_error("Index not built");
        }
        // search objects which distance is less than or equal to radius
        radius += 1e-2;
        
        size_t n_points = locs.size();
        std::vector<double> query_data(n_points * 3);
        
        for (size_t i = 0; i < n_points; ++i) {
            query_data[i*3] = static_cast<double>(locs[i].x);
            query_data[i*3+1] = static_cast<double>(locs[i].y);
            query_data[i*3+2] = static_cast<double>(locs[i].z);
        }
        flann::Matrix<double> query(query_data.data(), n_points, 3);

        std::vector<std::vector<int>> indices_vec(n_points);
        std::vector<std::vector<double>> dists_vec(n_points);
        
        index_->radiusSearch(query, indices_vec, dists_vec, radius * radius, flann::SearchParams(128));

        indices = indices_vec;
        distances = dists_vec;

        // Convert squared distances to actual Euclidean distances
        for (auto& dist_vec : distances) {
            for (auto& dist : dist_vec) {
                dist = std::sqrt(dist);
            }
        }
    }

    void WeightedKdTree::get_weights(const std::vector<int> &indices, std::vector<double> &weights) const
    {
        weights.resize(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            weights[i] = weights_[indices[i]];
        }
    }

    void WeightedKdTree::get_weights(const std::vector<std::vector<int>> &indices, std::vector<std::vector<double>> &weights) const
    {
        weights.resize(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            get_weights(indices[i], weights[i]);
        }
    }

    void WeightedKdTree::add_leaf(const size_t &index, const LocXyz &loc, const double &weight)
    {
        weights_[index] = weight;
        KdTree::add_leaf(index, loc);
    }

    void WeightedKdTree::init_tree(size_t tree_size)
    {
        KdTree::init_tree(tree_size);
        weights_.clear();
        weights_.resize(tree_size);
    }
}
