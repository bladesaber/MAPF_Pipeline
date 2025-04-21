#include <gtest/gtest.h>
#include "../include/TaskManager.h"

class TaskManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        pair_task_set = new TaskSpace::PairNodeTaskSet("test_pair");
        node_task_set = new TaskSpace::NodeTaskSet("test_node");
    }

    void TearDown() override {
        delete pair_task_set;
        delete node_task_set;
    }

    TaskSpace::PairNodeTaskSet* pair_task_set;
    TaskSpace::NodeTaskSet* node_task_set;
};

TEST_F(TaskManagerTest, PairNodeTaskAddition) {
    pair_task_set->add_task(
        "task1",
        "start1",
        "end1",
        LocXyz{0, 0, 0},
        LocXyz{1, 1, 1},
        0.5
    );

    EXPECT_EQ(pair_task_set->get_task_size(), 1);
    const auto& task = pair_task_set->get_task(0);
    EXPECT_EQ(task.name, "task1");
    EXPECT_EQ(task.start_name, "start1");
    EXPECT_EQ(task.end_name, "end1");
}

TEST_F(TaskManagerTest, NodeTaskAddition) {
    node_task_set->add_task(
        "task1",
        LocXyz{0, 0, 0},
        0.5
    );

    EXPECT_EQ(node_task_set->get_task_size(), 1);
    const auto& task = node_task_set->get_task(0);
    EXPECT_EQ(task.name, "task1");
    EXPECT_EQ(task.loc.x, 0);
    EXPECT_EQ(task.loc.y, 0);
    EXPECT_EQ(task.loc.z, 0);
}

TEST_F(TaskManagerTest, AddLandmarkToTask) {
    pair_task_set->add_task(
        "task1",
        "start1",
        "end1",
        LocXyz{0, 0, 0},
        LocXyz{2, 2, 2},
        0.5
    );

    pair_task_set->add_landmark_to_task(
        0,
        "landmark1",
        LocXyz{1, 1, 1}
    );

    EXPECT_EQ(pair_task_set->get_task_size(), 2);
    const auto& task1 = pair_task_set->get_task(0);
    const auto& task2 = pair_task_set->get_task(1);
    
    EXPECT_EQ(task1.start_name, "start1");
    EXPECT_EQ(task1.end_name, "landmark1");
    EXPECT_EQ(task2.start_name, "landmark1");
    EXPECT_EQ(task2.end_name, "end1");
}

TEST_F(TaskManagerTest, TaskPriorities) {
    pair_task_set->add_task(
        "task1",
        "start1",
        "end1",
        LocXyz{0, 0, 0},
        LocXyz{1, 1, 1},
        0.5
    );

    pair_task_set->add_task(
        "task2",
        "start2",
        "end2",
        LocXyz{2, 2, 2},
        LocXyz{3, 3, 3},
        0.5
    );

    const auto& task1 = pair_task_set->get_task(0);
    const auto& task2 = pair_task_set->get_task(1);
    
    EXPECT_EQ(task1.priority, 0);
    EXPECT_EQ(task2.priority, 1);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
