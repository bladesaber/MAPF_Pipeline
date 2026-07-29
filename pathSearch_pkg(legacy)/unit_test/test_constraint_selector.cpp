#include <gtest/gtest.h>
#include "../include/ConstraintSelector.h"
#include "../include/NetworkUtils.h"

class ConflictSetTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize ConflictSet with group names
        group_names = {"group1", "group2", "group3"};
        conflict_set = ConflictSet(group_names);
        
        // Create sample conflicts
        conflict1.group0 = "group1";
        conflict1.group1 = "group2";
        conflict1.loc0 = LocXyz{1, 1, 0};
        conflict1.loc1 = LocXyz{1, 2, 0};
        conflict1.radius0 = 1.0;
        conflict1.radius1 = 1.5;
        conflict1.conflict_cost = 1.5;

        conflict2.group0 = "group2";
        conflict2.group1 = "group3";
        conflict2.loc0 = LocXyz{1, 2, 0};
        conflict2.loc1 = LocXyz{1, 3, 0};
        conflict2.radius0 = 1.0;
        conflict2.radius1 = 1.0;
        conflict2.conflict_cost = 1.0;

        conflict3.group0 = "group1";
        conflict3.group1 = "group3";
        conflict3.loc0 = LocXyz{2, 3, 0};
        conflict3.loc1 = LocXyz{1, 3, 0};
        conflict3.radius0 = 1.0;
        conflict3.radius1 = 1.0;
        conflict3.conflict_cost = 1.0;
    }

    std::vector<std::string> group_names;
    ConflictSet conflict_set;
    Conflict conflict1, conflict2, conflict3;
};

TEST_F(ConflictSetTest, InitialState) {
    // Test initial state of ConflictSet
    EXPECT_EQ(conflict_set.conflicts.size(), 0);
    EXPECT_EQ(conflict_set.m_conflict_size, 0);
    
    // Check that group costs are initialized to 0
    for (const auto& group : group_names) {
        EXPECT_EQ(conflict_set.m_group_cost[group], 0.0);
    }
}

TEST_F(ConflictSetTest, AddConflict) {
    // Add first conflict
    conflict_set.add_conflict(conflict1);
    
    // Check conflict was added
    EXPECT_EQ(conflict_set.conflicts.size(), 1);
    EXPECT_EQ(conflict_set.m_conflict_size, 1);
    
    // Check conflict index map
    EXPECT_EQ(conflict_set.conflict_index_map["group1"].size(), 1);
    EXPECT_EQ(conflict_set.conflict_index_map["group2"].size(), 1);
    
    // Check group costs were updated
    EXPECT_DOUBLE_EQ(conflict_set.m_group_cost["group1"], 1.5);
    EXPECT_DOUBLE_EQ(conflict_set.m_group_cost["group2"], 1.5);
    EXPECT_DOUBLE_EQ(conflict_set.m_group_cost["group3"], 0.0);
}

TEST_F(ConflictSetTest, MultipleConflicts) {
    // Add both conflicts
    conflict_set.add_conflict(conflict1);
    conflict_set.add_conflict(conflict2);
    
    // Check conflicts were added
    EXPECT_EQ(conflict_set.conflicts.size(), 2);
    EXPECT_EQ(conflict_set.m_conflict_size, 2);
    
    // Check conflict index map sizes
    EXPECT_EQ(conflict_set.conflict_index_map["group1"].size(), 1);
    EXPECT_EQ(conflict_set.conflict_index_map["group2"].size(), 2);
    EXPECT_EQ(conflict_set.conflict_index_map["group3"].size(), 1);
    
    // Check accumulated group costs
    EXPECT_DOUBLE_EQ(conflict_set.m_group_cost["group1"], 1.5);
    EXPECT_DOUBLE_EQ(conflict_set.m_group_cost["group2"], 2.5);  // Involved in both conflicts
    EXPECT_DOUBLE_EQ(conflict_set.m_group_cost["group3"], 1.0);

    conflict_set.add_conflict(conflict3);

    // Check conflict index map sizes
    EXPECT_EQ(conflict_set.conflict_index_map["group1"].size(), 2);
    EXPECT_EQ(conflict_set.conflict_index_map["group2"].size(), 2);
    EXPECT_EQ(conflict_set.conflict_index_map["group3"].size(), 2);

    EXPECT_DOUBLE_EQ(conflict_set.m_group_cost["group1"], 2.5);
    EXPECT_DOUBLE_EQ(conflict_set.m_group_cost["group2"], 2.5);
    EXPECT_DOUBLE_EQ(conflict_set.m_group_cost["group3"], 2.0);
}

TEST_F(ConflictSetTest, Clear) {
    // Add conflicts and then clear
    conflict_set.add_conflict(conflict1);
    conflict_set.add_conflict(conflict2);
    conflict_set.clear();
    
    // Check everything was cleared
    EXPECT_EQ(conflict_set.conflicts.size(), 0);
    EXPECT_EQ(conflict_set.conflict_index_map.size(), 0);
    EXPECT_EQ(conflict_set.m_conflict_size, 2);
    
    // Group costs should remain initialized but unchanged
    for (const auto& group : group_names) {
        EXPECT_GT(conflict_set.m_group_cost[group], 0.0);
    }
}

TEST_F(ConflictSetTest, ConflictIndexMapCorrectness) {
    conflict_set.add_conflict(conflict1);
    
    // Check index and boolean flag for first conflict
    auto& group1_indices = conflict_set.conflict_index_map["group1"];
    auto& group2_indices = conflict_set.conflict_index_map["group2"];
    auto& group3_indices = conflict_set.conflict_index_map["group3"];
    
    ASSERT_EQ(group1_indices.size(), 1);
    ASSERT_EQ(group2_indices.size(), 1);
    ASSERT_EQ(group3_indices.size(), 0);
    
    // Check that indices and flags are correct
    EXPECT_EQ(std::get<0>(group1_indices[0]), 0);  // First conflict index
    EXPECT_TRUE(std::get<1>(group1_indices[0]));   // group0 flag
    
    EXPECT_EQ(std::get<0>(group2_indices[0]), 0);  // First conflict index
    EXPECT_FALSE(std::get<1>(group2_indices[0]));  // group1 flag

    conflict_set.add_conflict(conflict3);

    // Check conflict index map sizes
    EXPECT_EQ(conflict_set.conflict_index_map["group1"].size(), 2);
    EXPECT_EQ(conflict_set.conflict_index_map["group2"].size(), 1);
    EXPECT_EQ(conflict_set.conflict_index_map["group3"].size(), 1);
    
    // Check that indices and flags are correct
    EXPECT_EQ(std::get<0>(group1_indices[1]), 1);  // Second conflict index
    EXPECT_TRUE(std::get<1>(group1_indices[1]));   // group0 flag

    EXPECT_EQ(std::get<0>(group3_indices[0]), 1);  // Second conflict index
    EXPECT_FALSE(std::get<1>(group3_indices[0]));   // group1 flag
}

class ConstraintSelectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create sample network graphs
        MathSpace::NetworkGraph graph1, graph2;
        // Add some nodes and edges to the graphs
        graph1.add_edge(LocXyz{0, 0, 0}, LocXyz{1, 1, 0}, 1.414);
        graph1.add_edge(LocXyz{1, 1, 0}, LocXyz{2, 2, 0}, 1.414);
        graph1.add_edge(LocXyz{2, 2, 0}, LocXyz{3, 3, 0}, 1.414);
        graph1.add_edge(LocXyz{3, 3, 0}, LocXyz{4, 4, 0}, 1.414);
        graph1.add_edge(LocXyz{1, 1, 0}, LocXyz{1, 2, 0}, 1.0);
        graph1.add_edge(LocXyz{1, 2, 0}, LocXyz{1, 3, 0}, 1.0);

        graph2.add_edge(LocXyz{0, 0, 0}, LocXyz{1, 1, 0}, 1.414);
        graph2.add_edge(LocXyz{1, 1, 0}, LocXyz{2, 2, 0}, 1.414);
        graph2.add_edge(LocXyz{1, 1, 0}, LocXyz{1, 2, 0}, 1.0);
        graph2.add_edge(LocXyz{1, 2, 0}, LocXyz{1, 3, 0}, 1.0);

        graph_map["group1"] = graph1;
        graph_map["group2"] = graph2;

        // Create a sample conflict set
        conflict1.group0 = "group1";
        conflict1.group1 = "group2";
        conflict1.loc0 = LocXyz{1, 1, 0};
        conflict1.loc1 = LocXyz{1, 1, 0};
        conflict1.radius0 = 0.5;
        conflict1.radius1 = 0.5;
        conflict1.conflict_cost = 1.0;
        conflict_set.add_conflict(conflict1);

        conflict2.group0 = "group1";
        conflict2.group1 = "group2";
        conflict2.loc0 = LocXyz{1, 2, 0};
        conflict2.loc1 = LocXyz{1, 2, 0};
        conflict2.radius0 = 0.5;
        conflict2.radius1 = 0.5;
        conflict2.conflict_cost = 1.0;
        conflict_set.add_conflict(conflict2);
        selector = new SearchSpace::NearEndConstraintSelector(conflict_set);
    }

    void TearDown() override {
        delete selector;
    }

    Conflict conflict1, conflict2;
    ConflictSet conflict_set;
    std::map<std::string, MathSpace::NetworkGraph> graph_map;
    SearchSpace::NearEndConstraintSelector* selector;
};

TEST_F(ConstraintSelectorTest, SelectConstraint) {
    Constraint constraint;
    selector->select_constraint(graph_map, constraint);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
