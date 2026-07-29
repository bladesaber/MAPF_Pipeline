#include <gtest/gtest.h>
#include "../include/NetworkUtils.h"

TEST(NetworkGraphTest, BasicOperations) {
    MathSpace::NetworkGraph graph;
    
    // Test adding edges
    LocXyz loc1(0, 0, 0);
    LocXyz loc2(1, 1, 1);
    LocXyz loc3(2, 2, 2);
    
    graph.add_edge(loc1, loc2, 1.732); // sqrt(3)
    graph.add_edge(loc2, loc3, 1.732);
    
    // Test node count and edge count
    EXPECT_EQ(graph.get_node_count(), 3);
    EXPECT_EQ(graph.get_edge_count(), 2);
    
    // Test edge weight retrieval
    EXPECT_DOUBLE_EQ(graph.get_edge_weight(loc1, loc2), 1.732);
    EXPECT_DOUBLE_EQ(graph.get_edge_weight(loc2, loc3), 1.732);
}

TEST(NetworkGraphTest, ShortestPath) {
    MathSpace::NetworkGraph graph;
    
    // Create a simple graph
    LocXyz start(0, 0, 0);
    LocXyz middle(1, 1, 1);
    LocXyz end(2, 2, 2);
    LocXyz alt_point(2, 0, 0);
    
    // Add edges forming two possible paths
    graph.add_edge(start, middle, 1.732);  // Direct path
    graph.add_edge(middle, end, 1.732);
    graph.add_edge(start, alt_point, 2.0); // Alternative path
    graph.add_edge(alt_point, end, 2.828); // sqrt(8)
    
    // Test shortest path length
    double shortest_length = graph.get_shortest_path_length(start, end);
    EXPECT_NEAR(shortest_length, 3.464, 0.001); // ~2*sqrt(3)
    
    // Test shortest path vertices
    std::vector<LocXyz> path;
    graph.get_shortest_path(start, end, path);
    ASSERT_EQ(path.size(), 3);
    EXPECT_EQ(path[0], start);
    EXPECT_EQ(path[1], middle);
    EXPECT_EQ(path[2], end);
}

TEST(NetworkGraphTest, Neighbors) {
    MathSpace::NetworkGraph graph;
    
    LocXyz center(0, 0, 0);
    LocXyz n1(1, 0, 0);
    LocXyz n2(0, 1, 0);
    LocXyz n3(0, 0, 1);
    
    graph.add_edge(center, n1, 1.0);
    graph.add_edge(center, n2, 1.0);
    graph.add_edge(center, n3, 1.0);
    
    // Test degree
    EXPECT_EQ(graph.get_degree(center), 3);
    EXPECT_EQ(graph.get_degree(n1), 1);
    
    // Test neighbors
    std::vector<LocXyz> neighbors;
    graph.get_neighbours(center, neighbors);
    ASSERT_EQ(neighbors.size(), 3);
    
    // Sort neighbors to ensure consistent testing
    std::sort(neighbors.begin(), neighbors.end());

    EXPECT_EQ(neighbors[0], n3);
    EXPECT_EQ(neighbors[1], n2);
    EXPECT_EQ(neighbors[2], n1);
}

TEST(NetworkGraphTest, TerminalNodes) {
    MathSpace::NetworkGraph graph;
    
    LocXyz n1(0, 0, 0);
    LocXyz n2(1, 0, 0);
    LocXyz n3(2, 0, 0);
    LocXyz n4(1, 1, 0);
    
    // Create a T-shaped graph
    graph.add_edge(n1, n2, 1.0);
    graph.add_edge(n2, n3, 1.0);
    graph.add_edge(n2, n4, 1.0);
    
    // Test terminal nodes (nodes with degree 1)
    std::vector<LocXyz> terminals;
    graph.get_terminal_nodes(terminals);
    ASSERT_EQ(terminals.size(), 3);
    
    // Sort terminals to ensure consistent testing
    std::sort(terminals.begin(), terminals.end());
    EXPECT_EQ(terminals[0], n1);
    EXPECT_EQ(terminals[1], n4);
    EXPECT_EQ(terminals[2], n3);
}
