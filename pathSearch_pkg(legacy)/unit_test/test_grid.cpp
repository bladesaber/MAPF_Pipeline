#include <gtest/gtest.h>
#include "../include/SearchGrid.h"
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
    void visualize_grid_2d(GridSpace::SearchGrid* grid, double threshold, int padding = 2) {
        // Get grid dimensions and calculate bounds
        int width = grid->get_x_size();
        int height = grid->get_y_size();
        
        // Find bounds
        int min_x = 0;
        int max_x = width - 1;
        int min_y = 0;
        int max_y = height - 1;

        // Add padding
        min_x = std::max(0, min_x - padding);
        max_x = std::min(width - 1, max_x + padding);
        min_y = std::max(0, min_y - padding);
        max_y = std::min(height - 1, max_y + padding);

        // Create visualization grid
        int vis_width = max_x - min_x + 1;
        int vis_height = max_y - min_y + 1;
        std::vector<std::vector<char>> vis_grid(vis_height, std::vector<char>(vis_width, '.'));

        // Mark obstacles
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                bool is_obs;
                double dist;
                std::tie(is_obs, dist) = grid->is_obstacle(LocXyz{x, y, 0}, threshold);
                if (is_obs) {
                    vis_grid[y - min_y][x - min_x] = '#';
                }
            }
        }

        // Print the visualization
        std::cout << "\nGrid Visualization (threshold = " << threshold << "):\n";
        std::cout << "Legend: #=Obstacle, .=Empty\n\n";
        
        // Print column numbers
        std::cout << "   ";
        for (int x = min_x; x <= max_x; ++x) {
            std::cout << std::abs(x % 10);
        }
        std::cout << "\n   ";
        for (int x = min_x; x <= max_x; ++x) {
            std::cout << "-";
        }
        std::cout << "\n";

        // Print grid with row numbers
        for (int y = vis_height - 1; y >= 0; --y) {
            std::cout << std::setw(2) << (y + min_y) << "|";
            for (int x = 0; x < vis_width; ++x) {
                std::cout << vis_grid[y][x];
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    void visualize_grid_value_2d(GridSpace::SearchGrid* grid, double threshold, int padding = 2) {
        // Get grid dimensions and calculate bounds
        int width = grid->get_x_size();
        int height = grid->get_y_size();
        
        // Find bounds
        int min_x = 0;
        int max_x = width - 1;
        int min_y = 0;
        int max_y = height - 1;

        // Add padding
        min_x = std::max(0, min_x - padding);
        max_x = std::min(width - 1, max_x + padding);
        min_y = std::max(0, min_y - padding);
        max_y = std::min(height - 1, max_y + padding);

        // Create visualization grid with distance values
        int vis_width = max_x - min_x + 1;
        int vis_height = max_y - min_y + 1;
        std::vector<std::vector<double>> vis_grid(vis_height, std::vector<double>(vis_width));

        // Get distance values
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                bool is_obs;
                double dist;
                std::tie(is_obs, dist) = grid->is_obstacle(LocXyz{x, y, 0}, threshold);
                vis_grid[y - min_y][x - min_x] = dist;
            }
        }

        // Print the visualization
        std::cout << "\nDistance Grid Visualization (threshold = " << threshold << "):\n";
        std::cout << "Values show distance to nearest obstacle\n\n";
        
        // Calculate cell width based on maximum value length
        int cell_width = 6;  // Width for format: " 0.00"

        // Print column numbers
        std::cout << std::string(6, ' ');  // Align with row numbers
        for (int x = min_x; x <= max_x; ++x) {
            std::cout << std::setw(cell_width) << x;
        }
        std::cout << "\n" << std::string(6, ' ');
        for (int x = min_x; x <= max_x; ++x) {
            std::cout << std::string(cell_width, '-');
        }
        std::cout << "\n";

        // Print grid with row numbers and distance values
        for (int y = vis_height - 1; y >= 0; --y) {
            std::cout << std::setw(4) << (y + min_y) << " |";
            for (int x = 0; x < vis_width; ++x) {
                std::cout << std::fixed << std::setprecision(2) << std::setw(cell_width) << vis_grid[y][x];
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

class GridTest : public ::testing::Test {
protected:
    void SetUp() override {
        grid_2d = new GridSpace::StandardGrid(15, 10, 1, 0.1);
        grid_3d = new GridSpace::StandardGrid(10, 10, 10, 0.1);
    }

    void TearDown() override {
        delete grid_2d;
        delete grid_3d;
        grid_2d = nullptr;
        grid_3d = nullptr;
    }

    GridSpace::StandardGrid* grid_2d;
    GridSpace::StandardGrid* grid_3d;
};

TEST_F(GridTest, SetAndCheckObstacle) {
    LocXyz loc{5, 5, 5};
    grid_3d->set_obstacle(loc, 1.0);
    bool is_obs;
    double dist;
    std::tie(is_obs, dist) = grid_3d->is_obstacle(loc, 0.5);
    EXPECT_TRUE(is_obs);

    LocXyz loc2{8, 8, 8};
    std::tie(is_obs, dist) = grid_3d->is_obstacle(loc2, 0.5);
    EXPECT_FALSE(is_obs);

    LocXyz loc3{5, 5, 0};
    EXPECT_TRUE(grid_3d->is_in_grid(loc3));
    LocXyz loc4{5, 5, 10};
    EXPECT_FALSE(grid_3d->is_in_grid(loc4));
}

TEST_F(GridTest, ClearObstacle) {
    LocXyz loc{5, 5, 5};
    grid_3d->set_obstacle(loc, 1.0);
    grid_3d->clear_obstacle(loc);
    bool is_obs;
    double dist;
    std::tie(is_obs, dist) = grid_3d->is_obstacle(loc, 0.5);
    EXPECT_FALSE(is_obs);
}

TEST_F(GridTest, ValidMove) {
    grid_3d->set_obstacle(LocXyz{4, 5, 5}, 0.0);
    grid_3d->set_obstacle(LocXyz{5, 4, 5}, 0.0);

    EXPECT_FALSE(grid_3d->is_move_valid(LocXyz{4, 4, 5}, 0.0, 1, 0, 0));
    EXPECT_FALSE(grid_3d->is_move_valid(LocXyz{4, 4, 5}, 0.0, 0, 1, 0));
    EXPECT_TRUE(grid_3d->is_move_valid(LocXyz{4, 4, 5}, 0.0, 0, 0, 1));
    EXPECT_FALSE(grid_3d->is_move_valid(LocXyz{4, 4, 5}, 0.0, 1, 1, 0));

    grid_3d->set_obstacle(LocXyz{4, 4, 6}, 0.0);

    EXPECT_FALSE(grid_3d->is_move_valid(LocXyz{4, 4, 5}, 0.0, 0, 0, 1));
    EXPECT_FALSE(grid_3d->is_move_valid(LocXyz{4, 4, 5}, 0.0, 1, 1, 1));
    EXPECT_FALSE(grid_3d->is_move_valid(LocXyz{4, 4, 5}, 0.0, 1, 0, 1));
    EXPECT_FALSE(grid_3d->is_move_valid(LocXyz{4, 4, 5}, 0.0, 0, 1, 1));
    EXPECT_FALSE(grid_3d->is_move_valid(LocXyz{4, 4, 5}, 0.0, -1, 0, 1));
}

TEST_F(GridTest, Visualize2D) {
    visualize_grid_2d(grid_2d, 0.0);
    grid_2d->set_obstacle(LocXyz{5, 5, 0});
    visualize_grid_2d(grid_2d, 0.0);
    grid_2d->set_obstacle(LocXyz{6, 6, 0});
    visualize_grid_2d(grid_2d, 0.0);
}

class DistanceGridTest : public ::testing::Test {
protected:
    void SetUp() override {
        grid_3d = new GridSpace::DistanceGrid(10, 10, 10, 0.1);
        grid_2d = new GridSpace::DistanceGrid(10, 10, 1, 0.1);
    }

    void TearDown() override {
        delete grid_3d;
        delete grid_2d;
        grid_3d = nullptr;
        grid_2d = nullptr;
    }

    GridSpace::DistanceGrid* grid_3d;
    GridSpace::DistanceGrid* grid_2d;
};

TEST_F(DistanceGridTest, Visualize2D) {
    LocXyz obs_loc{5, 5, 0};
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            double dist = std::sqrt(std::pow(i - obs_loc.x, 2) + std::pow(j - obs_loc.y, 2)) * grid_2d->get_grid_resolution();
            grid_2d->set_obstacle(LocXyz{i, j, 0}, dist);
        }
    }
    visualize_grid_value_2d(grid_2d, 0.0);
    visualize_grid_2d(grid_2d, 0.1);
}

TEST_F(DistanceGridTest, SetAndCheckObstacle) {
    LocXyz obs_loc{5, 5, 5};
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                double dist = std::sqrt(
                    std::pow(i - obs_loc.x, 2) + 
                    std::pow(j - obs_loc.y, 2) + 
                    std::pow(k - obs_loc.z, 2));
                grid_3d->set_obstacle(LocXyz{i, j, k}, dist);
            }
        }
    }

    bool is_obs;
    double dist;
    std::tie(is_obs, dist) = grid_3d->is_obstacle(obs_loc, 0.5);
    EXPECT_TRUE(is_obs);
    EXPECT_NEAR(dist, 0.0, 0.01);

    LocXyz loc{1, 1, 1};
    double expected_dist = std::sqrt(
        std::pow(1 - obs_loc.x, 2) + 
        std::pow(1 - obs_loc.y, 2) + 
        std::pow(1 - obs_loc.z, 2));
    std::tie(is_obs, dist) = grid_3d->is_obstacle(loc, 0.5);
    EXPECT_FALSE(is_obs);
    EXPECT_NEAR(dist, expected_dist, 0.01);
}

class TreeGridTest : public ::testing::Test {
protected:
    void SetUp() override {
        grid_3d = new GridSpace::TreeGrid(10, 10, 10, 0.1);
        grid_2d = new GridSpace::TreeGrid(10, 10, 1, 0.1);
    }

    void TearDown() override {
        delete grid_3d;
        delete grid_2d;
        grid_3d = nullptr;
        grid_2d = nullptr;
    }

    GridSpace::TreeGrid* grid_3d;
    GridSpace::TreeGrid* grid_2d;
};

TEST_F(TreeGridTest, Visualize2D) {
    grid_2d->preprocess();
    visualize_grid_2d(grid_2d, 0.0);

    LocXyz obs_loc{5, 5, 0};
    grid_2d->set_obstacle(obs_loc, 1.0);
    grid_2d->preprocess();
    visualize_grid_2d(grid_2d, 0.0);
    visualize_grid_2d(grid_2d, 0.25);
    visualize_grid_2d(grid_2d, 1.0);
}

TEST_F(TreeGridTest, SetAndPreprocess) {
    double obs_radius = 1.0;
    double obj_radius = 0.5;

    LocXyz loc{5, 5, 5};
    grid_3d->set_obstacle(loc, obs_radius);
    grid_3d->preprocess();

    bool is_obs;
    double dist;
    std::tie(is_obs, dist) = grid_3d->is_obstacle(loc, obj_radius);
    EXPECT_TRUE(is_obs);
    EXPECT_NEAR(dist, 0.0, 0.01);

    LocXyz loc2{1, 1, 1};
    std::tie(is_obs, dist) = grid_3d->is_obstacle(loc2, obj_radius);
    EXPECT_FALSE(is_obs);
    double expected_dist = std::sqrt(
        std::pow(loc.x - loc2.x, 2) + 
        std::pow(loc.y - loc2.y, 2) + 
        std::pow(loc.z - loc2.z, 2));
    if (expected_dist > obs_radius + obj_radius) {
        expected_dist = 1e6;
    }
    EXPECT_NEAR(dist, expected_dist, 0.01);

    LocXyz loc3{4, 5, 5};
    std::tie(is_obs, dist) = grid_3d->is_obstacle(loc3, 0.5);
    EXPECT_TRUE(is_obs);
    expected_dist = std::sqrt(
        std::pow(loc.x - loc3.x, 2) + 
        std::pow(loc.y - loc3.y, 2) + 
        std::pow(loc.z - loc3.z, 2));
    if (expected_dist > obs_radius + obj_radius) {
        expected_dist = 1e6;
    }
    EXPECT_NEAR(dist, expected_dist, 0.01);
    
    LocXyz loc4{3, 5, 5};
    std::tie(is_obs, dist) = grid_3d->is_obstacle(loc4, 0.5);
    EXPECT_FALSE(is_obs);
    expected_dist = std::sqrt(
        std::pow(loc.x - loc4.x, 2) + 
        std::pow(loc.y - loc4.y, 2) + 
        std::pow(loc.z - loc4.z, 2));
    if (expected_dist > obs_radius + obj_radius) {
        expected_dist = 1e6;
    }
    EXPECT_NEAR(dist, expected_dist, 0.01);

    obj_radius = 2.0;
    LocXyz loc5{3, 5, 5};
    std::tie(is_obs, dist) = grid_3d->is_obstacle(loc5, obj_radius);
    EXPECT_TRUE(is_obs);
    expected_dist = std::sqrt(
        std::pow(loc.x - loc5.x, 2) + 
        std::pow(loc.y - loc5.y, 2) + 
        std::pow(loc.z - loc5.z, 2));
    if (expected_dist > obs_radius + obj_radius) {
        expected_dist = 1e6;
    }
    EXPECT_NEAR(dist, expected_dist, 0.01);
}

class BitGridTest : public ::testing::Test {
protected:
    void SetUp() override {
        grid_3d = new GridSpace::BitGrid(10, 10, 10, 0.1);
        grid_2d = new GridSpace::BitGrid(10, 10, 1, 0.1);
    }

    void TearDown() override {
        delete grid_3d;
        delete grid_2d;
        grid_3d = nullptr;
        grid_2d = nullptr;
    }

    GridSpace::BitGrid* grid_3d;
    GridSpace::BitGrid* grid_2d;
};

TEST_F(BitGridTest, SetAndCheckObstacle) {
    LocXyz loc{5, 5, 5};
    grid_3d->set_obstacle(loc, 0.0);
    bool is_obs;
    double dist;
    std::tie(is_obs, dist) = grid_3d->is_obstacle(loc, 0.5);
    EXPECT_TRUE(is_obs);
    EXPECT_DOUBLE_EQ(dist, 0.0);

    // Check adjacent cells are free
    std::tie(is_obs, dist) = grid_3d->is_obstacle(LocXyz{4, 5, 5}, 0.5);
    EXPECT_FALSE(is_obs);
    std::tie(is_obs, dist) = grid_3d->is_obstacle(LocXyz{6, 5, 5}, 0.5);
    EXPECT_FALSE(is_obs);
}

TEST_F(BitGridTest, ClearObstacle) {
    LocXyz loc{5, 5, 5};
    grid_3d->set_obstacle(loc, 0.0);
    grid_3d->clear_obstacle(loc);
    bool is_obs;
    double dist;
    std::tie(is_obs, dist) = grid_3d->is_obstacle(loc, 0.5);
    EXPECT_FALSE(is_obs);
    EXPECT_DOUBLE_EQ(dist, 0.0);
}

TEST_F(BitGridTest, GridBoundaries) {
    // Test corners
    LocXyz corners[] = {
        {0, 0, 0},
        {9, 0, 0},
        {0, 9, 0},
        {9, 9, 0},
        {0, 0, 9},
        {9, 0, 9},
        {0, 9, 9},
        {9, 9, 9}
    };

    for (const auto& corner : corners) {
        grid_3d->set_obstacle(corner, 0.0);
        bool is_obs;
        double dist;
        std::tie(is_obs, dist) = grid_3d->is_obstacle(corner, 0.5);
        EXPECT_TRUE(is_obs) << "Corner at (" << corner.x << "," << corner.y << "," << corner.z << ") not set";
    }
}

TEST_F(BitGridTest, BitPatternVerification) {
    // Create a checkerboard pattern in XY plane
    for (int x = 0; x < 10; x += 2) {
        for (int y = 0; y < 10; y += 2) {
            grid_2d->set_obstacle(LocXyz{x, y, 0}, 0.0);
        }
    }

    // Verify pattern
    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 10; y++) {
            bool is_obs;
            double dist;
            std::tie(is_obs, dist) = grid_2d->is_obstacle(LocXyz{x, y, 0}, 0.5);
            bool should_be_obstacle = (x % 2 == 0) && (y % 2 == 0);
            EXPECT_EQ(is_obs, should_be_obstacle) 
                << "Mismatch at (" << x << "," << y << ")";
        }
    }
}

TEST_F(BitGridTest, Visualize2D) {
    // Create a simple pattern
    grid_2d->set_obstacle(LocXyz{4, 4, 0}, 0.0);
    grid_2d->set_obstacle(LocXyz{4, 5, 0}, 0.0);
    grid_2d->set_obstacle(LocXyz{5, 4, 0}, 0.0);
    grid_2d->set_obstacle(LocXyz{5, 5, 0}, 0.0);
    
    visualize_grid_2d(grid_2d, 0.0);
    
    // Clear one obstacle and visualize again
    grid_2d->clear_obstacle(LocXyz{5, 5, 0});
    visualize_grid_2d(grid_2d, 0.0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
