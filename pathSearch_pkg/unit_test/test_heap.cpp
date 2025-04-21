#include <gtest/gtest.h>
#include "../include/SearchHeap.h"

class BoostHeapTest : public ::testing::Test {
protected:
    void SetUp() override {
        heap = new QueueSpace::BoostHeap();
    }

    void TearDown() override {
        delete heap;
    }

    QueueSpace::BoostHeap* heap;
};

TEST_F(BoostHeapTest, PushAndPop) {
    auto* node1 = new SearchSpace::SearchNode(LocXyz{0, 0, 0});
    auto* node2 = new SearchSpace::SearchNode(LocXyz{1, 1, 1});
    auto* node3 = new SearchSpace::SearchNode(LocXyz{2, 2, 2});
    node1->f_cost = 1.0;
    node2->f_cost = 2.0;
    node3->f_cost = 3.0;

    heap->push(node3);
    heap->push(node2);
    heap->push(node1);

    EXPECT_EQ(heap->pop(), node1);
    EXPECT_EQ(heap->pop(), node2);
    EXPECT_EQ(heap->pop(), node3);
    EXPECT_TRUE(heap->is_empty());

    delete node1;
    delete node2;
    delete node3;
}

TEST_F(BoostHeapTest, Contains) {
    auto* node = new SearchSpace::SearchNode(LocXyz{0, 0, 0});
    heap->push(node);
    EXPECT_TRUE(heap->contains(node));
    heap->pop();
    delete node;
}

TEST_F(BoostHeapTest, Update) {
    auto* node1 = new SearchSpace::SearchNode(LocXyz{0, 0, 0});
    auto* node2 = new SearchSpace::SearchNode(LocXyz{1, 1, 1});
    auto* node3 = new SearchSpace::SearchNode(LocXyz{2, 2, 2});

    node1->f_cost = 2.0;
    node2->f_cost = 1.0;
    node3->f_cost = 3.0;

    heap->push(node1);
    heap->push(node2);
    heap->push(node3);
    
    double oldCost = node1->f_cost;
    node1->f_cost = 0.5;
    heap->update(node1, oldCost);

    EXPECT_EQ(heap->pop()->f_cost, 0.5);
    EXPECT_EQ(heap->pop(), node2);
    EXPECT_EQ(heap->pop(), node3);

    delete node1;
    delete node2;
    delete node3;
}

TEST_F(BoostHeapTest, Remove) {
    auto* node = new SearchSpace::SearchNode(LocXyz{0, 0, 0});
    heap->push(node);
    heap->remove(node);
    EXPECT_FALSE(heap->contains(node));
    delete node;
}

class BucketPriorityQueueTest : public ::testing::Test {
protected:
    void SetUp() override {
        heap = new QueueSpace::BucketPriorityQueue();
    }

    void TearDown() override {
        delete heap;
    }

    QueueSpace::BucketPriorityQueue *heap;
};

TEST_F(BucketPriorityQueueTest, PushAndPop)
{
    auto *node1 = new SearchSpace::SearchNode(LocXyz{0, 0, 0});
    auto *node2 = new SearchSpace::SearchNode(LocXyz{1, 1, 1});
    auto *node3 = new SearchSpace::SearchNode(LocXyz{2, 2, 2});
    node1->f_cost = 1.0;
    node2->f_cost = 2.0;
    node3->f_cost = 3.0;

    heap->push(node3);
    heap->push(node2);
    heap->push(node1);

    EXPECT_EQ(heap->pop(), node1);
    EXPECT_EQ(heap->pop(), node2);
    EXPECT_EQ(heap->pop(), node3);
    EXPECT_TRUE(heap->is_empty());

    delete node1;
    delete node2;
    delete node3;
}

TEST_F(BucketPriorityQueueTest, Contains) {
    auto *node = new SearchSpace::SearchNode(LocXyz{0, 0, 0});
    node->f_cost = 1.0;
    heap->push(node);
    EXPECT_TRUE(heap->contains(node));
    heap->pop();
    delete node;
}

TEST_F(BucketPriorityQueueTest, Update) {
    auto *node1 = new SearchSpace::SearchNode(LocXyz{0, 0, 0});
    auto *node2 = new SearchSpace::SearchNode(LocXyz{1, 1, 1});
    auto *node3 = new SearchSpace::SearchNode(LocXyz{2, 2, 2});

    node1->f_cost = 2.0;
    node2->f_cost = 1.0;
    node3->f_cost = 3.0;

    heap->push(node1);
    heap->push(node2);
    heap->push(node3);
    
    double oldCost = node1->f_cost;
    node1->f_cost = 0.5;
    heap->update(node1, oldCost);

    EXPECT_EQ(heap->pop()->f_cost, 0.5);
    EXPECT_EQ(heap->pop(), node2);
    EXPECT_EQ(heap->pop(), node3);

    delete node1;
    delete node2;
    delete node3;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
