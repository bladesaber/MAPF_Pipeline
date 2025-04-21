#include <gtest/gtest.h>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include "../include/AstarSearch.h"
#include "../include/TaskManager.h"
#include "../include/SetPlanner.h"

namespace
{

    void visualize_result(
        GridSpace::SearchGrid *grid,
        double threshold,
        const PathSetData &result_sets,
        const std::vector<LocXyz> &terminals,
        int padding = 2)
    {
        // Get grid dimensions and calculate bounds
        int width = grid->get_x_size();
        int height = grid->get_y_size();

        // Find bounds including path, starts, and goals
        int min_x = width;
        int max_x = 0;
        int min_y = height;
        int max_y = 0;

        // Update bounds based on all paths in result_sets
        for (const auto &path_data : result_sets.m_paths)
        {
            for (const auto &point : path_data.m_path)
            {
                min_x = std::min(min_x, static_cast<int>(point.x) - padding);
                max_x = std::max(max_x, static_cast<int>(point.x) + padding);
                min_y = std::min(min_y, static_cast<int>(point.y) - padding);
                max_y = std::max(max_y, static_cast<int>(point.y) + padding);
            }
        }

        // Update bounds based on starts and goals
        for (const auto &terminal : terminals)
        {
            min_x = std::min(min_x, static_cast<int>(terminal.x) - padding);
            max_x = std::max(max_x, static_cast<int>(terminal.x) + padding);
            min_y = std::min(min_y, static_cast<int>(terminal.y) - padding);
            max_y = std::max(max_y, static_cast<int>(terminal.y) + padding);
        }

        // Ensure bounds are within grid limits
        min_x = std::max(0, min_x);
        max_x = std::min(width - 1, max_x);
        min_y = std::max(0, min_y);
        max_y = std::min(height - 1, max_y);

        // Create visualization grid
        int vis_width = max_x - min_x + 1;
        int vis_height = max_y - min_y + 1;
        std::vector<std::vector<char>> vis_grid(vis_height, std::vector<char>(vis_width, '.'));

        // Mark obstacles
        for (int y = min_y; y <= max_y; ++y)
        {
            for (int x = min_x; x <= max_x; ++x)
            {
                bool is_obs;
                double dist;
                std::tie(is_obs, dist) = grid->is_obstacle(LocXyz{static_cast<int32_t>(x),
                                                                  static_cast<int32_t>(y),
                                                                  0},
                                                           threshold);
                if (is_obs)
                {
                    vis_grid[y - min_y][x - min_x] = '#';
                }
            }
        }

        // Mark all paths with different symbols for each path
        size_t symbol_idx = 0;
        for (const auto &path_data : result_sets.m_paths)
        {
            for (size_t i = 0; i < path_data.m_path.size(); ++i)
            {
                int x = static_cast<int>(path_data.m_path[i].x) - min_x;
                int y = static_cast<int>(path_data.m_path[i].y) - min_y;
                vis_grid[y][x] = 'o';
            }
        }

        // Mark starts with 'S'
        for (const auto &terminal : terminals)
        {
            int x = static_cast<int>(terminal.x) - min_x;
            int y = static_cast<int>(terminal.y) - min_y;
            if (x >= 0 && x < vis_width && y >= 0 && y < vis_height)
            {
                vis_grid[y][x] = 'S';
            }
        }

        // Print the visualization
        std::cout << "\nPath Planning Visualization:\n";
        std::cout << "Legend: #=Obstacle, S=Start\n";
        std::cout << "\n";

        // Print column numbers
        std::cout << "   ";
        for (int x = min_x; x <= max_x; ++x)
        {
            std::cout << std::abs(x % 10);
        }
        std::cout << "\n   ";
        for (int x = min_x; x <= max_x; ++x)
        {
            std::cout << "-";
        }
        std::cout << "\n";

        // Print grid with row numbers
        for (int y = vis_height - 1; y >= 0; --y)
        {
            std::cout << std::setw(2) << (y + min_y) << "|";
            for (int x = 0; x < vis_width; ++x)
            {
                std::cout << vis_grid[y][x];
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

} // namespace

// Base test class with common functionality
class SetPlannerTestBase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed the random number generator with current time
        std::srand(static_cast<unsigned int>(std::time(nullptr)));

        grid = new GridSpace::StandardGrid(10, 10, 1, 0.1); // 2D grid for simplicity
        astar = new SearchSpace::AstarSearch<QueueSpace::BoostHeap>(0.5);
    }

    void TearDown() override
    {
        delete grid;
        delete astar;
    }

    // Helper function to find path by name
    const std::vector<LocXyz> *findPath(const PathSetData &result_sets, const std::string &name)
    {
        for (const auto &path_data : result_sets.m_paths)
        {
            if (path_data.m_name == name)
            {
                return &path_data.m_path;
            }
        }
        return nullptr;
    }

    // Helper function to check if a location is an obstacle
    bool isObstacle(const LocXyz &loc, double threshold)
    {
        bool is_obstacle;
        double distance;
        std::tie(is_obstacle, distance) = grid->is_obstacle(loc, threshold);
        return is_obstacle;
    }

    // Helper function to get random grid position
    LocXyz getRandomGridPosition()
    {
        return LocXyz{
            static_cast<int32_t>(std::rand() % grid->get_x_size()),
            static_cast<int32_t>(std::rand() % grid->get_y_size()),
            0};
    }

    GridSpace::StandardGrid *grid;
    SearchSpace::AstarSearch<QueueSpace::BoostHeap> *astar;
};

// Test class specifically for SetPairNodePlanner
class SetPairNodePlannerTest : public SetPlannerTestBase
{
protected:
    void SetUp() override
    {
        SetPlannerTestBase::SetUp();
        pair_planner = new SearchSpace::SetPairNodePlanner(*astar);
    }

    void TearDown() override
    {
        delete pair_planner;
        SetPlannerTestBase::TearDown();
    }

    SearchSpace::SetPairNodePlanner *pair_planner;
};

TEST_F(SetPairNodePlannerTest, SimplePathfinding) {
    // Create a simple pair node task set
    TaskSpace::PairNodeTaskSet task_set("test_set");

    task_set.add_task(
        "task1",
        "start1",
        "end1",
        LocXyz{1, 1, 0},
        LocXyz{2, 6, 0},
        0.5
    );
    task_set.add_task(
        "task2",
        "start2",
        "end2",
        LocXyz{8, 8, 0},
        LocXyz{7, 4, 0},
        0.5
    );
    task_set.add_task(
        "task3",
        "start1",
        "start2",
        LocXyz{1, 1, 0},
        LocXyz{8, 8, 0},
        0.5
    );

    pair_planner->set_tasks(task_set);
    PathSetData result_sets;

    EXPECT_TRUE(pair_planner->search(grid, result_sets));

    // Verify the path exists and connects start to goal
    const std::vector<LocXyz>* path = findPath(result_sets, "task1");
    ASSERT_NE(path, nullptr) << "Path not found for task: " << "task1";
    EXPECT_GT(path->size(), 0);

    // Verify path is within grid bounds
    for (const auto& loc : *path) {
        EXPECT_TRUE(grid->is_in_grid(loc)) << "Path point outside grid at " << loc;
    }

    visualize_result(grid, 0.5, result_sets,
        {LocXyz{1, 1, 0}, LocXyz{8, 8, 0}, LocXyz{2, 6, 0}, LocXyz{7, 4, 0}}, 2);
}

TEST_F(SetPairNodePlannerTest, PathfindingWithObstacles) {
    LocXyz start1 = LocXyz{1, 1, 0};
    LocXyz end1 = LocXyz{8, 8, 0};
    LocXyz start2 = LocXyz{1, 8, 0};
    LocXyz end2 = LocXyz{8, 1, 0};

    // Add random obstacles
    for (int i = 0; i < 10; i++) {
        LocXyz obs_loc = getRandomGridPosition();
        // Skip if obstacle would be placed on start/end positions
        if (obs_loc == start1 || obs_loc == end1 || obs_loc == start2 || obs_loc == end2) {
            continue;
        }
        grid->set_obstacle(obs_loc, 1.0);
    }

    TaskSpace::PairNodeTaskSet task_set("test_set");
    task_set.add_task(
        "task1",
        "start1",
        "end1",
        LocXyz{1, 1, 0},
        LocXyz{8, 8, 0},
        0.5
    );
    task_set.add_task(
        "task2",
        "start2",
        "end1",
        LocXyz{1, 8, 0},
        LocXyz{8, 8, 0},
        0.5
    );
    task_set.add_task(
        "task3",
        "start3",
        "end1",
        LocXyz{8, 1, 0},
        LocXyz{8, 8, 0},
        0.5
    );

    pair_planner->set_tasks(task_set);
    PathSetData result_sets;

    EXPECT_TRUE(pair_planner->search(grid, result_sets));

    visualize_result(grid, 0.5, result_sets,
        {LocXyz{1, 1, 0}, LocXyz{8, 8, 0}, LocXyz{1, 8, 0}, LocXyz{8, 1, 0}}, 2);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
