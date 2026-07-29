#pragma once

#include "BaseHead.h"
#include <omp.h>

namespace MathSpace
{
    static double get_distance(const LocXyz &loc1, const LocXyz &loc2)
    {
        return std::sqrt(std::pow(loc1.x - loc2.x, 2) + std::pow(loc1.y - loc2.y, 2) + std::pow(loc1.z - loc2.z, 2));
    }

    static void argsort_row(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &indices)
    {
        // Pre-allocate indices for each row
        #pragma omp parallel for
        for (int i = 0; i < mat.rows(); i++) {
            // Create index array
            std::vector<int> idx(mat.cols());
            std::iota(idx.begin(), idx.end(), 0);  // Fill with 0,1,2,...

            // Sort indices based on matrix values using lambda
            std::sort(idx.begin(), idx.end(), [&mat, i](int j1, int j2) { return mat(i,j1) < mat(i,j2); });

            // Copy sorted indices to result matrix
            for (int j = 0; j < mat.cols(); j++) {
                indices(i,j) = idx[j];
            }
        }
    }

    static void sort_row(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &sorted_mat)
    {
        sorted_mat.resize(mat.rows(), mat.cols());
        #pragma omp parallel for
        for (int i = 0; i < mat.rows(); i++) {
            Eigen::VectorXd row = mat.row(i);
            std::sort(row.data(), row.data() + row.size());
            sorted_mat.row(i) = row;
        }
    }

    static void argsort_col(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &indices)
    {
        indices.resize(mat.rows(), mat.cols());
        
        #pragma omp parallel for
        for (int j = 0; j < mat.cols(); j++) {
            std::vector<int> idx(mat.rows());
            std::iota(idx.begin(), idx.end(), 0);  // Fill with 0,1,2,...
            
            // Sort indices based on matrix values using lambda
            std::sort(idx.begin(), idx.end(),
                [&mat, j](int i1, int i2) { return mat(i1,j) < mat(i2,j); });
            
            // Store sorted indices in column
            for (int i = 0; i < mat.rows(); i++) {
                indices(i,j) = idx[i];  // Store indices in order
            }
        }
    }

    static void sort_col(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &sorted_mat)
    {
        sorted_mat.resize(mat.rows(), mat.cols());
        #pragma omp parallel for
        for (int j = 0; j < mat.cols(); j++) {
            Eigen::VectorXd col = mat.col(j);
            std::sort(col.data(), col.data() + col.size());
            sorted_mat.col(j) = col;
        }
    }

    static void row_asscend_rank(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &indices)
    {
        indices.resize(mat.rows(), mat.cols());   
        #pragma omp parallel for
        for (int i = 0; i < mat.rows(); i++) {
            // Create index array
            std::vector<int> idx(mat.cols());
            std::iota(idx.begin(), idx.end(), 0);  // Fill with 0,1,2,...

            // Sort indices based on matrix values
            std::sort(idx.begin(), idx.end(),
                     [&mat, i](int j1, int j2) { return mat(i,j1) < mat(i,j2); });

            // Create rank array - the rank of each index
            std::vector<int> rank(mat.cols());
            for (int j = 0; j < mat.cols(); j++) {
                rank[idx[j]] = j;  // Value at position idx[j] gets rank j
            }
            
            // Store ranks in output matrix
            for (int j = 0; j < mat.cols(); j++) {
                indices(i,j) = rank[j];
            }
        }
    }

    static void col_asscend_rank(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &indices)
    {
        indices.resize(mat.rows(), mat.cols());
        #pragma omp parallel for
        for (int j = 0; j < mat.cols(); j++) {
            std::vector<int> idx(mat.rows());
            std::iota(idx.begin(), idx.end(), 0);  // Fill with 0,1,2,...
            
            // Sort indices based on matrix values
            std::sort(idx.begin(), idx.end(),
                [&mat, j](int i1, int i2) { return mat(i1,j) < mat(i2,j); });
            
            // Create rank array - the rank of each index
            std::vector<int> rank(mat.rows());
            for (int i = 0; i < mat.rows(); i++) {
                rank[idx[i]] = i;  // Value at position idx[i] gets rank i
            }
            
            // Store ranks in output matrix
            for (int i = 0; i < mat.rows(); i++) {
                indices(i,j) = rank[i];
            }
        }   
    }

    static void min_row(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
        Eigen::Matrix<double, Eigen::Dynamic, 1> &min_vec)
    {
        // Use Eigen's optimized reduction operations
        min_vec = mat.rowwise().minCoeff();
    }

    static void max_row(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
        Eigen::Matrix<double, Eigen::Dynamic, 1> &max_vec)
    {
        // Use Eigen's optimized reduction operations
        max_vec = mat.rowwise().maxCoeff();
    }

    static void min_col(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
        Eigen::Matrix<double, 1, Eigen::Dynamic> &min_vec)
    {
        // Use Eigen's optimized reduction operations
        min_vec = mat.colwise().minCoeff();
    }

    static void min_col(
        const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &mat,
        Eigen::Matrix<double, 1, Eigen::Dynamic> &min_vec)
    {
        // Convert integer matrix to double matrix and use existing implementation
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> double_mat = mat.cast<double>();
        min_vec = double_mat.colwise().minCoeff();
    }

    static void max_col(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
        Eigen::Matrix<double, 1, Eigen::Dynamic> &max_vec)
    {
        // Use Eigen's optimized reduction operations
        max_vec = mat.colwise().maxCoeff();
    }

    static void sum_row(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
        Eigen::Matrix<double, Eigen::Dynamic, 1> &sum_vec)
    {
        sum_vec = mat.rowwise().sum();
    }

    static void sum_col(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat,
        Eigen::Matrix<double, 1, Eigen::Dynamic> &sum_vec)
    {
        sum_vec = mat.colwise().sum();
    }

    static double compute_vector_angle(double x1, double y1, double z1, double x2, double y2, double z2)
    {
        Eigen::Vector3d vec1(x1, y1, z1);
        Eigen::Vector3d vec2(x2, y2, z2);
        double dot_product = vec1.dot(vec2);
        double magnitude1 = vec1.norm();
        double magnitude2 = vec2.norm();
        return dot_product / (magnitude1 * magnitude2);
    }

    static int min_col_index(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat, size_t col_index)
    {
        auto min_iter = std::min_element(mat.col(col_index).begin(), mat.col(col_index).end());
        return std::distance(mat.col(col_index).begin(), min_iter);
    }

    static int max_col_index(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat, size_t col_index)
    {
        auto max_iter = std::max_element(mat.col(col_index).begin(), mat.col(col_index).end());
        return std::distance(mat.col(col_index).begin(), max_iter);
    }

    static int min_row_index(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat, size_t row_index)
    {
        auto min_iter = std::min_element(mat.row(row_index).begin(), mat.row(row_index).end());
        return std::distance(mat.row(row_index).begin(), min_iter);
    }

    static int max_row_index(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &mat, size_t row_index)
    {
        auto max_iter = std::max_element(mat.row(row_index).begin(), mat.row(row_index).end());
        return std::distance(mat.row(row_index).begin(), max_iter);
    }
    
}

