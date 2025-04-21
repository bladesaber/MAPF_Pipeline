#include <gtest/gtest.h>
#include "../include/SearchNode.h"

class NodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        node = new SearchSpace::SearchNode(LocXyz{1, 2, 3});
    }

    void TearDown() override {
        delete node;
    }

    SearchSpace::SearchNode* node;
};

TEST_F(NodeTest, Construction) {
    EXPECT_EQ(node->m_loc.x, 1);
    EXPECT_EQ(node->m_loc.y, 2);
    EXPECT_EQ(node->m_loc.z, 3);
    EXPECT_EQ(node->m_parent, nullptr);
    EXPECT_EQ(node->m_status, SearchSpace::SearchNodeStatus::OnNone);
}

TEST_F(NodeTest, NodeHasher) {
    SearchSpace::SearchNode::NodeHasher hasher;
    SearchSpace::SearchNode node1(LocXyz{1, 2, 3});
    SearchSpace::SearchNode node2(LocXyz{1, 2, 3});
    SearchSpace::SearchNode node3(LocXyz{3, 2, 1});
    
    EXPECT_EQ(hasher(&node1), hasher(&node2));
    EXPECT_NE(hasher(&node1), hasher(&node3));
}

TEST_F(NodeTest, NodeEqual) {
    SearchSpace::SearchNode::NodeEqual equal;
    SearchSpace::SearchNode node1(LocXyz{1, 2, 3});
    SearchSpace::SearchNode node2(LocXyz{1, 2, 3});
    SearchSpace::SearchNode node3(LocXyz{3, 2, 1});
    
    EXPECT_TRUE(equal(&node1, &node2));
    EXPECT_FALSE(equal(&node1, &node3));
}

TEST_F(NodeTest, NodeCompare) {
    SearchSpace::SearchNode::NodeCompare compare;
    SearchSpace::SearchNode node1(LocXyz{0, 0, 0});
    SearchSpace::SearchNode node2(LocXyz{1, 1, 1});
    
    node1.f_cost = 1.0;
    node2.f_cost = 2.0;
    
    EXPECT_FALSE(compare(&node1, &node2));  // node1 < node2
    EXPECT_TRUE(compare(&node2, &node1));   // node2 > node1
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
