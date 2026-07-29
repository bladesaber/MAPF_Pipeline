#include <gtest/gtest.h>
#include "../include/NeighbourTree.h"

TEST(KdTreeTest, BasicOperations) {
    MathSpace::KdTree tree;
    
    // Initialize tree with 3 points
    tree.init_tree(3);
    
    // Add points in a triangle formation
    tree.add_leaf(0, LocXyz(0, 0, 0));
    tree.add_leaf(1, LocXyz(10, 0, 0));
    tree.add_leaf(2, LocXyz(0, 10, 0));
    
    // Build the tree
    tree.build_tree();
    
    // Test k-nearest neighbor search
    std::vector<int> indices;
    std::vector<double> distances;
    tree.search_knn(LocXyz(5, 5, 0), 2, indices, distances);
    
    // Should find 2 nearest points
    ASSERT_EQ(indices.size(), 2);
    ASSERT_EQ(distances.size(), 2);

    // Test distances are correct (approximately)
    EXPECT_NEAR(distances[0], 7.07, 0.01); // ~sqrt(0.5^2 + 0.5^2)
    EXPECT_NEAR(distances[1], 7.07, 0.01);
}

TEST(KdTreeTest, RadiusSearch) {
    MathSpace::KdTree tree;
    
    // Initialize and build tree with 4 points
    tree.init_tree(4);
    tree.add_leaf(0, LocXyz(0, 0, 0));
    tree.add_leaf(1, LocXyz(1, 0, 0));
    tree.add_leaf(2, LocXyz(0, 1, 0));
    tree.add_leaf(3, LocXyz(2, 2, 0));
    tree.build_tree();
    
    // Test radius search
    std::vector<int> indices;
    std::vector<double> distances;
    tree.search_radius(LocXyz(0, 0, 0), 1.5, indices, distances);
    
    // Should find points within radius 1.5
    ASSERT_EQ(indices.size(), 3); // Should find (0,0,0), (1,0,0), and (0,1,0)
    std::sort(indices.begin(), indices.end());
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 1);
    EXPECT_EQ(indices[2], 2);
    
    // Test batch kNN search
    std::vector<LocXyz> queries = {LocXyz(0, 0, 0), LocXyz(2.5, 2.5, 0)};
    std::vector<std::vector<int>> batch_indices;
    std::vector<std::vector<double>> batch_distances;
    tree.search_knn(queries, 1, batch_indices, batch_distances);
    
    ASSERT_EQ(batch_indices.size(), 2);
    ASSERT_EQ(batch_indices[0].size(), 1);
    ASSERT_EQ(batch_indices[1].size(), 1);
    EXPECT_EQ(batch_indices[0][0], 0);
    EXPECT_EQ(batch_indices[1][0], 3);
}

TEST(WeightedKdTreeTest, BasicOperations) {
    MathSpace::WeightedKdTree tree;
    
    // Initialize tree with 3 weighted points
    tree.init_tree(3);
    
    // Add points with weights
    tree.add_leaf(0, LocXyz(0, 0, 0), 1.0);
    tree.add_leaf(1, LocXyz(1, 0, 0), 2.0);
    tree.add_leaf(2, LocXyz(0, 1, 0), 3.0);
    
    // Test weight retrieval for single point
    EXPECT_DOUBLE_EQ(tree.get_weight(1), 2.0);
    
    // Test weight retrieval for multiple points
    std::vector<int> indices = {0, 2};
    std::vector<double> weights;
    tree.get_weights(indices, weights);
    
    ASSERT_EQ(weights.size(), 2);
    EXPECT_DOUBLE_EQ(weights[0], 1.0);
    EXPECT_DOUBLE_EQ(weights[1], 3.0);
}

TEST(WeightedKdTreeTest, BatchOperations) {
    MathSpace::WeightedKdTree tree;
    
    // Initialize and build tree
    tree.init_tree(4);
    tree.add_leaf(0, LocXyz(0, 0, 0), 1.0);
    tree.add_leaf(1, LocXyz(1, 0, 0), 2.0);
    tree.add_leaf(2, LocXyz(0, 1, 0), 3.0);
    tree.add_leaf(3, LocXyz(1, 1, 0), 4.0);
    tree.build_tree();
    
    // Test batch weight retrieval
    std::vector<std::vector<int>> batch_indices = {{0, 1}, {2, 3}};
    std::vector<std::vector<double>> batch_weights;
    tree.get_weights(batch_indices, batch_weights);
    
    ASSERT_EQ(batch_weights.size(), 2);
    ASSERT_EQ(batch_weights[0].size(), 2);
    ASSERT_EQ(batch_weights[1].size(), 2);
    
    EXPECT_DOUBLE_EQ(batch_weights[0][0], 1.0);
    EXPECT_DOUBLE_EQ(batch_weights[0][1], 2.0);
    EXPECT_DOUBLE_EQ(batch_weights[1][0], 3.0);
    EXPECT_DOUBLE_EQ(batch_weights[1][1], 4.0);
    
    // Test combined search and weight retrieval
    std::vector<int> search_indices;
    std::vector<double> search_distances;
    tree.search_knn(LocXyz(0.5, 0.5, 0), 2, search_indices, search_distances);
    
    std::vector<double> point_weights;
    tree.get_weights(search_indices, point_weights);
    ASSERT_EQ(point_weights.size(), 2);
}
        