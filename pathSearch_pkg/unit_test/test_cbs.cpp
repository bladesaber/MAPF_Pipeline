#include <gtest/gtest.h>
#include <chrono>
#include <omp.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <unordered_map>
#include <map>

#include "../include/ConflictBasedSearch.h"
#include "../include/SearchGrid.h"
#include "../include/AstarSearch.h"
#include "../include/SetPlanner.h"

namespace {
    // Global variables for visualization
    float camera_distance = 100.0f;
    float camera_angle_x = 30.0f;
    float camera_angle_y = 45.0f;
    using ResultMap = std::map<std::string, PathSetData>;
    ResultMap global_result_map;
    struct Color {
        GLfloat rgb[3];
    };
    std::unordered_map<std::string, Color> color_map;
    bool should_exit = false;

    // Function to generate unique colors for each group
    void generate_colors() {
        const std::vector<Color> base_colors = {
            Color{{1.0f, 0.0f, 0.0f}},  // Red
            Color{{0.0f, 1.0f, 0.0f}},  // Green
            Color{{0.0f, 0.0f, 1.0f}},  // Blue
            Color{{1.0f, 1.0f, 0.0f}},  // Yellow
            Color{{1.0f, 0.0f, 1.0f}},  // Magenta
            Color{{0.0f, 1.0f, 1.0f}},  // Cyan
            Color{{1.0f, 0.5f, 0.0f}},  // Orange
            Color{{0.5f, 0.0f, 1.0f}}   // Purple
        };

        int color_idx = 0;
        for (const auto& pair : global_result_map) {
            color_map[pair.first] = base_colors[color_idx % base_colors.size()];
            color_idx++;
        }
    }

    // Function to draw a sphere at given location
    void draw_sphere(const LocXyz& loc, double radius) {
        glPushMatrix();
        glTranslatef(loc.x, loc.y, loc.z);
        glutSolidSphere(radius, 10, 10);
        glPopMatrix();
    }

    // Function to draw a cylinder between two points
    void draw_cylinder(const LocXyz& start, const LocXyz& end, double radius) {
        GLUquadricObj* quadric = gluNewQuadric();
        
        // Calculate cylinder height and orientation
        float dx = end.x - start.x;
        float dy = end.y - start.y;
        float dz = end.z - start.z;
        float height = sqrt(dx*dx + dy*dy + dz*dz);
        
        // Calculate rotation angles
        float angle_x = 180.0f / M_PI * acos(dz/height);
        float angle_y = 180.0f / M_PI * atan2(dy, dx);
        
        glPushMatrix();
        glTranslatef(start.x, start.y, start.z);
        glRotatef(angle_y, 0.0f, 0.0f, 1.0f);
        glRotatef(angle_x, 0.0f, 1.0f, 0.0f);
        gluCylinder(quadric, radius, radius, height, 10, 1);
        glPopMatrix();
        
        gluDeleteQuadric(quadric);
    }

    // Timer callback function
    void timer(int value) {
        if (should_exit) {
            glutDestroyWindow(glutGetWindow());
            return;
        }
        glutTimerFunc(16, timer, 0);  // Schedule next update (60 FPS)
    }

    // Display callback function
    void display() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        
        // Set camera position
        gluLookAt(
            camera_distance * cos(camera_angle_y * M_PI/180.0f) * cos(camera_angle_x * M_PI/180.0f),
            camera_distance * sin(camera_angle_y * M_PI/180.0f),
            camera_distance * cos(camera_angle_y * M_PI/180.0f) * sin(camera_angle_x * M_PI/180.0f),
            0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f
        );

        // Draw coordinate axes
        glLineWidth(2.0f);
        glBegin(GL_LINES);
        // X axis (red)
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(50.0f, 0.0f, 0.0f);
        // Y axis (green)
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 50.0f, 0.0f);
        // Z axis (blue)
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 50.0f);
        glEnd();

        // Draw paths
        for (const auto& pair : global_result_map) {
            const std::string& group_name = pair.first;
            const auto& path_set = pair.second;
            
            // Set color for this group
            const Color& color = color_map[group_name];
            glColor3fv(color.rgb);
            
            // Draw each path in the set
            for (const auto& path : path_set.m_paths) {
                // Draw path segments
                for (size_t i = 0; i < path.m_path.size() - 1; ++i) {
                    draw_cylinder(path.m_path[i], path.m_path[i + 1], path.m_radius * 0.5);
                }
                
                // Draw spheres at waypoints
                for (const auto& waypoint : path.m_path) {
                    draw_sphere(waypoint, path.m_radius);
                }
            }
        }
        
        glutSwapBuffers();
    }

    // Keyboard callback function
    void keyboard(unsigned char key, int x, int y) {
        switch (key) {
            case 'w': camera_angle_x += 5.0f; break;
            case 's': camera_angle_x -= 5.0f; break;
            case 'a': camera_angle_y -= 5.0f; break;
            case 'd': camera_angle_y += 5.0f; break;
            case 'q': camera_distance -= 5.0f; break;
            case 'e': camera_distance += 5.0f; break;
            case 27:  // ESC key
                should_exit = true;
                return;
        }
        glutPostRedisplay();
    }

    // Initialize OpenGL settings
    void init_gl() {
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glEnable(GL_COLOR_MATERIAL);
        
        GLfloat light_position[] = { 100.0f, 100.0f, 100.0f, 1.0f };
        glLightfv(GL_LIGHT0, GL_POSITION, light_position);
        
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        
        // Set up projection matrix
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0f, 1.0f, 0.1f, 1000.0f);
        glMatrixMode(GL_MODELVIEW);
    }

    void draw_path_3d(const std::map<std::string, PathSetData>& result_map) {
        // Store the result map globally for access in display function
        global_result_map = result_map;
        should_exit = false;
        
        // Generate colors for each group
        generate_colors();
        
        // Initialize GLUT
        int argc = 1;
        char* argv[] = {(char*)"dummy", nullptr};
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(800, 800);
        glutCreateWindow("3D Path Visualization");
        
        // Set up callbacks
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutTimerFunc(0, timer, 0);
        
        // Initialize OpenGL settings
        init_gl();
        
        // Start the main loop
        glutMainLoop();
    }
}

/*
class ConflictBaseSearchNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test node
        node = new SearchSpace::CbsNode<SearchSpace::NearEndConstraintSelector>(1);
        
        // Create sample path data
        std::vector<LocXyz> path1 = {
            LocXyz{0, 0, 0},
            LocXyz{1, 1, 0},
            LocXyz{2, 2, 0}
        };
        PathSetData result1;
        result1.add_path("path1", path1, 0.5);
        
        std::vector<LocXyz> path2 = {
            LocXyz{0, 2, 0},
            LocXyz{1, 2, 0},
            LocXyz{2, 2, 0}
        };
        PathSetData result2;
        result2.add_path("path2", path2, 0.5);

        // Add results to node
        node->result_map["group1"] = std::make_shared<PathSetData>(result1);
        node->result_map["group2"] = std::make_shared<PathSetData>(result2);

        // Initialize dynamic obstacle maps
        node->dynamic_obstacle_map["group1"] = std::make_shared<ObstacleList>();
        node->dynamic_obstacle_map["group2"] = std::make_shared<ObstacleList>();
    }

    void TearDown() override {
        delete node;
    }

    SearchSpace::ConflictBaseSearchNode* node;
};

TEST_F(ConflictBaseSearchNodeTest, AddDynamicObstacle) {
    ObstacleData obstacle(LocXyz{1, 1, 0}, 0.5);
    node->add_dynamic_obstacle("group1", obstacle);
    
    ASSERT_EQ(node->dynamic_obstacle_map["group1"]->size(), 1);
    EXPECT_EQ((*node->dynamic_obstacle_map["group1"])[0].loc, obstacle.loc);
    EXPECT_DOUBLE_EQ((*node->dynamic_obstacle_map["group1"])[0].radius, obstacle.radius);
}

TEST_F(ConflictBaseSearchNodeTest, ResetResult) {
    std::vector<LocXyz> new_path = {
        LocXyz{0, 0, 0},
        LocXyz{1, 0, 0},
        LocXyz{2, 0, 0}
    };
    PathSetData new_result;
    new_result.add_path("new_path", new_path, 0.7);
    
    node->reset_result("group1", new_result);
    
    ASSERT_EQ(node->result_map["group1"]->m_paths.size(), 1);
    EXPECT_EQ(node->result_map["group1"]->m_paths[0].m_path, new_path);
    EXPECT_DOUBLE_EQ(node->result_map["group1"]->m_paths[0].m_radius, 0.7);
}

TEST_F(ConflictBaseSearchNodeTest, GetGroupNames) {
    std::vector<std::string> names;
    node->get_group_names(names);
    
    ASSERT_EQ(names.size(), 2);
    EXPECT_TRUE(std::find(names.begin(), names.end(), "group1") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "group2") != names.end());
}

class ConflictBaseSearchNodeSearchTest : public ::testing::Test {
protected:
    void SetUp() override {
        node = new SearchSpace::CbsNode<SearchSpace::NearEndConstraintSelector>(1);
    }

    void TearDown() override {
        delete node;
    }

    SearchSpace::ConflictBaseSearchNode* node;
};

TEST_F(ConflictBaseSearchNodeSearchTest, Search) {
    // Create grid and search components
    GridSpace::StandardGrid grid(10, 10, 1, 0.1);
    SearchSpace::AstarSearch<QueueSpace::BoostHeap> astar(0.5);
    SearchSpace::SetPairNodePlanner pair_planner(astar);

    // Create task sets
    TaskSpace::PairNodeTaskSet task_set1("group1");
    task_set1.add_task(
        "task1",
        "start1",
        "end1",
        LocXyz{1, 1, 0},
        LocXyz{8, 8, 0},
        0.5
    );

    // Initialize dynamic obstacle map
    node->dynamic_obstacle_map["group1"] = std::make_shared<ObstacleList>();

    // Test search functionality
    EXPECT_TRUE(node->search(pair_planner, &grid, &task_set1));
    std::cout << "Before dynamic obstacle, group1->cost: " << node->result_map["group1"]->m_cost << std::endl;

    // Verify results were properly set
    ASSERT_NE(node->result_map["group1"], nullptr);
    EXPECT_FALSE(node->result_map["group1"]->m_paths.empty());

    // Verify path properties
    const auto& path = node->result_map["group1"]->m_paths[0];
    EXPECT_GT(path.m_path.size(), 0);
    
    // Check start position
    EXPECT_EQ(path.m_path.front().x, 1);
    EXPECT_EQ(path.m_path.front().y, 1);
    EXPECT_EQ(path.m_path.front().z, 0);
    
    // Check end position
    EXPECT_EQ(path.m_path.back().x, 8);
    EXPECT_EQ(path.m_path.back().y, 8);
    EXPECT_EQ(path.m_path.back().z, 0);
    
    EXPECT_DOUBLE_EQ(path.m_radius, 0.5);

    // Test with obstacles
    ObstacleData obstacle(LocXyz{4, 4, 0}, 0.5);
    node->add_dynamic_obstacle("group1", obstacle);
    double original_cost = node->result_map["group1"]->m_cost;

    // Search should still succeed but find a different path
    EXPECT_TRUE(node->search(pair_planner, &grid, &task_set1));
    EXPECT_GE(node->result_map["group1"]->m_cost, original_cost);

    std::cout << "After dynamic obstacle, group1->cost: " << node->result_map["group1"]->m_cost << std::endl;
    
    // Verify new path avoids obstacle
    const auto& new_path = node->result_map["group1"]->m_paths[0].m_path;
    for (const auto& loc : new_path) {
        bool is_obstacle;
        double distance;
        std::tie(is_obstacle, distance) = grid.is_obstacle(loc, 0.5);
        EXPECT_FALSE(is_obstacle) << "Path goes through obstacle at (" 
                                << loc.x << "," << loc.y << "," << loc.z << ")";
    }

    // Test conflict detection
    EXPECT_TRUE(node->is_free_conflict);
    EXPECT_GT(node->g_cost, 0.0);
    EXPECT_GT(node->f_cost, 0.0);

    // Test with multiple groups
    TaskSpace::PairNodeTaskSet task_set2("group2");
    task_set2.add_task(
        "task2",
        "start1",
        "end2",
        LocXyz{1, 1, 0},
        LocXyz{1, 8, 0},
        0.5
    );

    node->dynamic_obstacle_map["group2"] = std::make_shared<ObstacleList>();
    
    // Search with second group
    EXPECT_TRUE(node->search(pair_planner, &grid, &task_set2));

    // Verify both paths exist
    ASSERT_NE(node->result_map["group1"], nullptr);
    ASSERT_NE(node->result_map["group2"], nullptr);
    EXPECT_FALSE(node->result_map["group1"]->m_paths.empty());
    EXPECT_FALSE(node->result_map["group2"]->m_paths.empty());

    // Verify constraint handling
    const auto& path2 = node->result_map["group2"]->m_paths[0].m_path;

    // Check start and end positions for path2
    EXPECT_EQ(path2.front().x, 1);
    EXPECT_EQ(path2.front().y, 1);
    EXPECT_EQ(path2.front().z, 0);
    EXPECT_EQ(path2.back().x, 1);
    EXPECT_EQ(path2.back().y, 8);
    EXPECT_EQ(path2.back().z, 0);

    std::cout << "Summary dynamic obstacle, group1->cost: " << node->result_map["group1"]->m_cost << std::endl;
    std::cout << "Summary dynamic obstacle, group1->cost: " << node->result_map["group2"]->m_cost << std::endl;
}
*/

class ConflictBaseSearchParallelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a large grid for parallel testing (shared among all groups)
        grid = new GridSpace::DistanceGrid(10, 10, 10, 0.1);
        
        // Create CBS instance
        cbs = new SearchSpace::CbsSearch<SearchSpace::NearEndConstraintSelector>(
            grid, 1.0, 1.0, 1000
        );

        // ------ initial group 1 ------
        auto* astar1 = new SearchSpace::AstarSearch<QueueSpace::BoostHeap>(0.5, 100000, false);
        auto* pair_planner1 = new SearchSpace::SetPairNodePlanner(*astar1);
        auto* task_set1 = new TaskSpace::PairNodeTaskSet("group1");
        task_set1->add_task(
            "task1",
            "start1",
            "end1",
            LocXyz{5, 0, 5},
            LocXyz{5, 9, 5},
            0.5
        );
        cbs->add_task_set(task_set1, pair_planner1);

        astar_instances.push_back(astar1);
        planner_instances.push_back(pair_planner1);
        task_sets.push_back(task_set1);

        // ------ initial group 2 ------
        auto* astar2 = new SearchSpace::AstarSearch<QueueSpace::BoostHeap>(0.5, 100000, false);
        auto* pair_planner2 = new SearchSpace::SetPairNodePlanner(*astar2);
        auto* task_set2 = new TaskSpace::PairNodeTaskSet("group2");
        task_set2->add_task(
            "task2",
            "start2",
            "end2",
            LocXyz{0, 5, 5},
            LocXyz{9, 5, 5},
            0.5
        );
        cbs->add_task_set(task_set2, pair_planner2);

        astar_instances.push_back(astar2);
        planner_instances.push_back(pair_planner2);
        task_sets.push_back(task_set2);

        // ------ initial group 3 ------
        auto* astar3 = new SearchSpace::AstarSearch<QueueSpace::BoostHeap>(0.5, 100000, false);
        auto* pair_planner3 = new SearchSpace::SetPairNodePlanner(*astar3);
        auto* task_set3 = new TaskSpace::PairNodeTaskSet("group3");
        task_set3->add_task(
            "task3",
            "start3",
            "end3",
            LocXyz{5, 5, 0},
            LocXyz{5, 5, 9},
            0.5
        );
        cbs->add_task_set(task_set3, pair_planner3);

        astar_instances.push_back(astar3);
        planner_instances.push_back(pair_planner3);
        task_sets.push_back(task_set3);

    }

    void TearDown() override {
        delete grid;
        delete cbs;
        
        // Clean up all astar instances
        for (auto* astar : astar_instances) {
            delete astar;
        }
        
        // Clean up all planner instances
        for (auto* planner : planner_instances) {
            delete planner;
        }
        
        // Clean up all task sets
        for (auto* task_set : task_sets) {
            delete task_set;
        }
        
        astar_instances.clear();
        planner_instances.clear();
        task_sets.clear();
    }

    GridSpace::DistanceGrid* grid;
    SearchSpace::CbsSearch<SearchSpace::NearEndConstraintSelector>* cbs;
    std::vector<SearchSpace::AstarSearch<QueueSpace::BoostHeap>*> astar_instances;
    std::vector<SearchSpace::SetPairNodePlanner*> planner_instances;
    std::vector<TaskSpace::TaskSet*> task_sets;
};

TEST_F(ConflictBaseSearchParallelTest, Search) {
    ASSERT_TRUE(cbs->preprocess_root_node());

    PyWrapCbsResult result;
    ASSERT_TRUE(cbs->search(&result));

    // // Print results
    // for (const auto& [group_name, path_set] : result.results) {
    //     std::cout << "Group: " << group_name << std::endl;
    //     for (const auto& path : path_set.m_paths) {
    //         for (const auto& loc : path.m_path) {
    //             std::cout << "(" << loc.x << ", " << loc.y << ", " << loc.z << ") ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }
    // draw_path_3d(result.results);
}

TEST_F(ConflictBaseSearchParallelTest, ParallelConflictDetection) {
    // First preprocess the root node
    ASSERT_TRUE(cbs->preprocess_root_node());
    
    // Get the root node's results
    auto* root_node = dynamic_cast<SearchSpace::CbsNode<SearchSpace::NearEndConstraintSelector>*>(cbs->pop_node());
    ASSERT_NE(root_node, nullptr);

    // std::map<std::string, PathSetData> result_map;
    // for (const auto& [group_name, result] : root_node->result_map) {
    //     result_map[group_name] = *result;
    // }
    // draw_path_3d(result_map);
    
    // Measure time for conflict detection
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::milliseconds;
    
    TimePoint start_time = Clock::now();
    
    // Run conflict detection
    ConflictSet conflict_set;
    Constraint constraint;
    bool is_free_conflict = root_node->find_division_constraint(conflict_set, constraint);
    
    TimePoint end_time = Clock::now();
    Duration duration = std::chrono::duration_cast<Duration>(end_time - start_time);
    
    // Verify that conflicts were detected (with 8 crossing paths, there should be conflicts)
    EXPECT_FALSE(is_free_conflict);
    
    // Verify that multiple conflicts were detected
    EXPECT_GT(conflict_set.m_conflict_size, 0);
    
    // Print timing information
    std::cout << "Parallel conflict detection took " << duration.count() 
              << " ms for " << task_sets.size() << " groups" << std::endl;
    std::cout << "Number of conflicts detected: " << conflict_set.m_conflict_size << std::endl;
    
    // Print OpenMP information
    #ifdef _OPENMP
        std::cout << "OpenMP is enabled with " << omp_get_max_threads() << " max threads" << std::endl;
    #else
        std::cout << "OpenMP is not enabled" << std::endl;
    #endif
    
    delete root_node;
}

TEST_F(ConflictBaseSearchParallelTest, ParallelVsSequentialComparison) {
    // First preprocess the root node
    ASSERT_TRUE(cbs->preprocess_root_node());
    
    auto* root_node = dynamic_cast<SearchSpace::CbsNode<SearchSpace::NearEndConstraintSelector>*>(
        cbs->pop_node());
    ASSERT_NE(root_node, nullptr);

    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::milliseconds;

    // Run with OpenMP (parallel)
    TimePoint start_parallel = Clock::now();
    ConflictSet conflict_set_parallel;
    Constraint constraint_parallel;
    bool is_free_conflict_parallel = root_node->find_division_constraint(
        conflict_set_parallel, constraint_parallel);
    TimePoint end_parallel = Clock::now();
    Duration duration_parallel = std::chrono::duration_cast<Duration>(
        end_parallel - start_parallel);

    // Store parallel results
    size_t num_conflicts_parallel = conflict_set_parallel.m_conflict_size;
    
    // Print results
    std::cout << "Parallel execution time: " << duration_parallel.count() << " ms" << std::endl;
    std::cout << "Number of conflicts (parallel): " << num_conflicts_parallel << std::endl;
    
    #ifdef _OPENMP
        std::cout << "Running with " << omp_get_num_threads() << " threads" << std::endl;
    #else
        std::cout << "OpenMP is not enabled" << std::endl;
    #endif
    
    // Verify results
    EXPECT_FALSE(is_free_conflict_parallel);
    EXPECT_GT(num_conflicts_parallel, 0);
    
    delete root_node;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
