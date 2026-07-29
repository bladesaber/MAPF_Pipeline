#include <gtest/gtest.h>
#include "../include/AstarSearch.h"
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

    void visualize_result(
        GridSpace::SearchGrid *grid,
        double threshold,
        const std::vector<LocXyz> &path,
        const std::vector<LocXyz> &starts,
        const std::vector<LocXyz> &goals,
        int padding = 2)
    {
        // Get grid dimensions and calculate bounds
        int width = grid->get_x_size();
        int height = grid->get_y_size();

        // Find bounds including path, starts, and goals
        int min_x = 0;
        int max_x = width - 1;
        int min_y = 0;
        int max_y = height - 1;

        // Update bounds based on path points
        for (const auto &point : path)
        {
            min_x = std::min(min_x, static_cast<int>(point.x) - padding);
            max_x = std::max(max_x, static_cast<int>(point.x) + padding);
            min_y = std::min(min_y, static_cast<int>(point.y) - padding);
            max_y = std::max(max_y, static_cast<int>(point.y) + padding);
        }

        // Update bounds based on starts and goals
        for (const auto &start : starts)
        {
            min_x = std::min(min_x, static_cast<int>(start.x) - padding);
            max_x = std::max(max_x, static_cast<int>(start.x) + padding);
            min_y = std::min(min_y, static_cast<int>(start.y) - padding);
            max_y = std::max(max_y, static_cast<int>(start.y) + padding);
        }
        for (const auto &goal : goals)
        {
            min_x = std::min(min_x, static_cast<int>(goal.x) - padding);
            max_x = std::max(max_x, static_cast<int>(goal.x) + padding);
            min_y = std::min(min_y, static_cast<int>(goal.y) - padding);
            max_y = std::max(max_y, static_cast<int>(goal.y) + padding);
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
                std::tie(is_obs, dist) = grid->is_obstacle(LocXyz{x, y, 0}, threshold);
                if (is_obs)
                {
                    vis_grid[y - min_y][x - min_x] = '#';
                }
            }
        }

        // Mark path
        for (size_t i = 0; i < path.size(); ++i)
        {
            int x = static_cast<int>(path[i].x) - min_x;
            int y = static_cast<int>(path[i].y) - min_y;
            if (x >= 0 && x < vis_width && y >= 0 && y < vis_height)
            {
                if (i == 0 || i == path.size() - 1)
                {
                    vis_grid[y][x] = '*'; // Path start/end points
                }
                else
                {
                    vis_grid[y][x] = 'o'; // Path points
                }
            }
        }

        // Mark starts with 'S'
        for (const auto &start : starts)
        {
            int x = static_cast<int>(start.x) - min_x;
            int y = static_cast<int>(start.y) - min_y;
            if (x >= 0 && x < vis_width && y >= 0 && y < vis_height)
            {
                vis_grid[y][x] = 'S';
            }
        }

        // Mark goals with 'G'
        for (const auto &goal : goals)
        {
            int x = static_cast<int>(goal.x) - min_x;
            int y = static_cast<int>(goal.y) - min_y;
            if (x >= 0 && x < vis_width && y >= 0 && y < vis_height)
            {
                vis_grid[y][x] = 'G';
            }
        }

        // Print the visualization
        std::cout << "\nPath Planning Visualization:\n";
        std::cout << "Legend: #=Obstacle, S=Start, G=Goal, o=Path, *=Path Start/End, .=Empty\n\n";

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

class AstarTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        grid_3d = new GridSpace::StandardGrid(10, 10, 10, 0.1);
        grid_2d = new GridSpace::StandardGrid(10, 10, 1, 0.1);
        astar = new SearchSpace::AstarSearch<QueueSpace::BoostHeap>(0.5);
    }

    void TearDown() override
    {
        delete grid_3d;
        delete grid_2d;
        delete astar;
        grid_3d = nullptr;
        grid_2d = nullptr;
        astar = nullptr;
    }

    GridSpace::StandardGrid *grid_3d;
    GridSpace::StandardGrid *grid_2d;
    SearchSpace::AstarSearch<QueueSpace::BoostHeap> *astar;
};

TEST_F(AstarTest, SimplePathfinding_2D)
{
    std::vector<LocXyz> starts = {LocXyz{6, 8, 0}};
    std::vector<LocXyz> goals = {LocXyz{1, 1, 0}};
    std::vector<LocXyz> path;

    EXPECT_TRUE(astar->search(grid_2d, starts, goals, path));
    EXPECT_GT(path.size(), 0);
    EXPECT_EQ(path.front(), starts[0]);
    EXPECT_EQ(path.back(), goals[0]);

    visualize_result(grid_2d, astar->get_radius(), path, starts, goals);
}

TEST_F(AstarTest, SimplePathfinding_3D)
{
    std::vector<LocXyz> starts = {LocXyz{1, 1, 1}};
    std::vector<LocXyz> goals = {LocXyz{8, 8, 8}};
    std::vector<LocXyz> path;

    EXPECT_TRUE(astar->search(grid_3d, starts, goals, path));
    EXPECT_GT(path.size(), 0);
    EXPECT_EQ(path.front(), starts[0]);
    EXPECT_EQ(path.back(), goals[0]);
}

TEST_F(AstarTest, BlockedPath_3D)
{
    // Create a wall of obstacles
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            grid_3d->set_obstacle(LocXyz{5, i, j}, 1.0);
        }
    }

    std::vector<LocXyz> starts = {LocXyz{1, 5, 5}};
    std::vector<LocXyz> goals = {LocXyz{8, 5, 5}};
    std::vector<LocXyz> path;

    EXPECT_FALSE(astar->search(grid_3d, starts, goals, path));
}

TEST_F(AstarTest, BlockedPath_2D)
{
    grid_2d->set_obstacle(LocXyz{4, 5, 0}, 1.0);
    grid_2d->set_obstacle(LocXyz{5, 5, 0}, 1.0);
    grid_2d->set_obstacle(LocXyz{6, 5, 0}, 1.0);

    std::vector<LocXyz> starts = {LocXyz{1, 1, 0}};
    std::vector<LocXyz> goals = {LocXyz{9, 9, 0}};
    std::vector<LocXyz> path;
    EXPECT_TRUE(astar->search(grid_2d, starts, goals, path));
    EXPECT_GT(path.size(), 0);
    EXPECT_EQ(path.front(), starts[0]);
    EXPECT_EQ(path.back(), goals[0]);

    visualize_result(grid_2d, astar->get_radius(), path, starts, goals);

    // ------------------------------------------------------------
    grid_2d->set_obstacle(LocXyz{2, 6, 0}, 1.0);
    grid_2d->set_obstacle(LocXyz{3, 6, 0}, 1.0);
    grid_2d->set_obstacle(LocXyz{7, 6, 0}, 1.0);
    grid_2d->set_obstacle(LocXyz{8, 6, 0}, 1.0);
    grid_2d->set_obstacle(LocXyz{9, 6, 0}, 1.0);

    path = std::vector<LocXyz>();
    EXPECT_TRUE(astar->search(grid_2d, starts, goals, path));
    EXPECT_GT(path.size(), 0);
    EXPECT_EQ(path.front(), starts[0]);
    EXPECT_EQ(path.back(), goals[0]);

    visualize_result(grid_2d, astar->get_radius(), path, starts, goals);

    // ------------------------------------------------------------
    grid_2d->set_obstacle(LocXyz{0, 4, 0}, 1.0);
    grid_2d->set_obstacle(LocXyz{1, 4, 0}, 1.0);

    path = std::vector<LocXyz>();
    EXPECT_TRUE(astar->search(grid_2d, starts, goals, path));
    visualize_result(grid_2d, astar->get_radius(), path, starts, goals);
}

class ThetaStarTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        grid_2d = new GridSpace::DistanceGrid(10, 10, 1, 0.1);
        theta_star = new SearchSpace::ThetaStarSearch<QueueSpace::BoostHeap>(1.0, 100000, 1.0, 1.0);
    }

    void TearDown() override
    {
        delete grid_2d;
        delete theta_star;
        grid_2d = nullptr;
        theta_star = nullptr;
    }

    GridSpace::DistanceGrid *grid_2d;
    SearchSpace::ThetaStarSearch<QueueSpace::BoostHeap> *theta_star;
};

TEST_F(ThetaStarTest, SimplePathfinding)
{
    std::vector<LocXyz> starts = {LocXyz{1, 1, 0}};
    std::vector<LocXyz> goals = {LocXyz{6, 8, 0}};
    std::vector<LocXyz> path;

    EXPECT_TRUE(theta_star->search(grid_2d, starts, goals, path));
    EXPECT_GT(path.size(), 0);
    EXPECT_EQ(path.front(), starts[0]);
    EXPECT_EQ(path.back(), goals[0]);

    visualize_result(grid_2d, theta_star->get_radius(), path, starts, goals);
}

TEST_F(ThetaStarTest, PathWithObstacles)
{
    std::vector<LocXyz> obstacle_array = {
        LocXyz{4, 4, 0},
        LocXyz{4, 5, 0},
        LocXyz{5, 5, 0},
        LocXyz{5, 4, 0},
    };

    for (int i = 0; i < grid_2d->get_x_size(); i++)
    {
        for (int j = 0; j < grid_2d->get_y_size(); j++)
        {
            double dist = 1000000.0;
            for (const auto &obs : obstacle_array)
            {
                dist = std::min(dist, MathSpace::get_distance(LocXyz{i, j, 0}, obs));
            }
            grid_2d->set_obstacle(LocXyz{i, j, 0}, dist);
        }
    }

    std::vector<LocXyz> starts = {LocXyz{1, 1, 0}};
    std::vector<LocXyz> goals = {LocXyz{8, 8, 0}};
    std::vector<LocXyz> path;

    EXPECT_TRUE(theta_star->search(grid_2d, starts, goals, path));
    EXPECT_GT(path.size(), 0);
    EXPECT_EQ(path.front(), starts[0]);
    EXPECT_EQ(path.back(), goals[0]);

    visualize_result(grid_2d, theta_star->get_radius(), path, starts, goals);
}

class ScaledWeightedAstarTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        grid_2d = new GridSpace::DistanceGrid(10, 10, 1, 0.1);
        scaled_astar = new SearchSpace::ScaledWeightedAstarSearch<QueueSpace::BoostHeap>(
            1.0, 100000, 1.0, 1.0, 2.0, 2);
    }

    void TearDown() override
    {
        delete grid_2d;
        delete scaled_astar;
        grid_2d = nullptr;
        scaled_astar = nullptr;
    }

    GridSpace::DistanceGrid *grid_2d;
    SearchSpace::ScaledWeightedAstarSearch<QueueSpace::BoostHeap> *scaled_astar;
};

TEST_F(ScaledWeightedAstarTest, SimplePathfinding)
{
    std::vector<LocXyz> starts = {LocXyz{0, 0, 0}};
    std::vector<LocXyz> goals = {LocXyz{9, 9, 0}};
    std::vector<LocXyz> path;

    EXPECT_TRUE(scaled_astar->search(grid_2d, starts, goals, path));
    EXPECT_GT(path.size(), 0);
    EXPECT_EQ(path.front(), starts[0]);
    EXPECT_EQ(path.back(), goals[0]);

    visualize_result(grid_2d, scaled_astar->get_radius(), path, starts, goals);
}

TEST_F(ScaledWeightedAstarTest, PathWithObstacles)
{
    std::vector<LocXyz> obstacle_array = {
        LocXyz{3, 3, 0},
        LocXyz{3, 4, 0},
        LocXyz{4, 4, 0},
        LocXyz{5, 5, 0},
        LocXyz{6, 6, 0},
        LocXyz{7, 6, 0},
    };

    for (int i = 0; i < grid_2d->get_x_size(); i++)
    {
        for (int j = 0; j < grid_2d->get_y_size(); j++)
        {
            double dist = 1000000.0;
            for (const auto &obs : obstacle_array)
            {
                dist = std::min(dist, MathSpace::get_distance(LocXyz{i, j, 0}, obs));
            }
            grid_2d->set_obstacle(LocXyz{i, j, 0}, dist);
        }
    }

    std::vector<LocXyz> starts = {LocXyz{1, 1, 0}};
    std::vector<LocXyz> goals = {LocXyz{8, 8, 0}};
    std::vector<LocXyz> path;

    EXPECT_TRUE(scaled_astar->search(grid_2d, starts, goals, path));
    EXPECT_GT(path.size(), 0);
    EXPECT_EQ(path.front(), starts[0]);
    EXPECT_EQ(path.back(), goals[0]);

    visualize_result(grid_2d, scaled_astar->get_radius(), path, starts, goals);
}

class BiDirectionalAstarSearchTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        grid_2d = new GridSpace::StandardGrid(10, 10, 1, 0.1);
        bidirectional_astar = new SearchSpace::BiDirectionalAstarSearch<QueueSpace::BoostHeap>(0.5, 100000, 1.0, 1.0);
    }

    void TearDown() override
    {
        delete grid_2d;
        delete bidirectional_astar;
        grid_2d = nullptr;
        bidirectional_astar = nullptr;
    }

    GridSpace::StandardGrid *grid_2d;
    SearchSpace::BiDirectionalAstarSearch<QueueSpace::BoostHeap> *bidirectional_astar;
};

TEST_F(BiDirectionalAstarSearchTest, SimplePathfinding)
{
    std::vector<LocXyz> starts = {LocXyz{1, 1, 0}};
    std::vector<LocXyz> goals = {LocXyz{6, 8, 0}};
    std::vector<LocXyz> path;

    EXPECT_TRUE(bidirectional_astar->search(grid_2d, starts, goals, path));
    EXPECT_GT(path.size(), 0);

    visualize_result(grid_2d, bidirectional_astar->get_radius(), path, starts, goals);
}

TEST_F(BiDirectionalAstarSearchTest, PathWithObstacles)
{
    // Create a maze-like environment
    for (int i = 2; i < 8; i += 2)
    {
        for (int j = 0; j < 8; j++)
        {
            if (j != 4)
            { // Leave a gap in the middle
                grid_2d->set_obstacle(LocXyz{i, j, 0}, 1.0);
            }
        }
    }

    std::vector<LocXyz> starts = {LocXyz{0, 0, 0}};
    std::vector<LocXyz> goals = {LocXyz{9, 9, 0}};
    std::vector<LocXyz> path;

    EXPECT_TRUE(bidirectional_astar->search(grid_2d, starts, goals, path));
    EXPECT_GT(path.size(), 0);
    EXPECT_EQ(path.front(), starts[0]);
    EXPECT_EQ(path.back(), goals[0]);

    visualize_result(grid_2d, bidirectional_astar->get_radius(), path, starts, goals);
}

class AstarSearchBucketPriorityQueueTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        grid_2d = new GridSpace::StandardGrid(10, 10, 1, 0.1);
        astar = new SearchSpace::AstarSearch<QueueSpace::BucketPriorityQueue>(0.5);
    }

    void TearDown() override
    {
        delete grid_2d;
        delete astar;
        grid_2d = nullptr;
        astar = nullptr;
    }

    GridSpace::StandardGrid *grid_2d;
    SearchSpace::AstarSearch<QueueSpace::BucketPriorityQueue> *astar;
};

TEST_F(AstarSearchBucketPriorityQueueTest, SimplePathfinding_2D)
{
    std::vector<LocXyz> starts = {LocXyz{6, 8, 0}};
    std::vector<LocXyz> goals = {LocXyz{1, 1, 0}};
    std::vector<LocXyz> path;

    EXPECT_TRUE(astar->search(grid_2d, starts, goals, path));
    EXPECT_GT(path.size(), 0);
    EXPECT_EQ(path.front(), starts[0]);
    EXPECT_EQ(path.back(), goals[0]);

    visualize_result(grid_2d, astar->get_radius(), path, starts, goals);
}


int
main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
