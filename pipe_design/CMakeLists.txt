cmake_minimum_required(VERSION 3.14)
project(pipe_design LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(pipe_grid
    src/grid/ObstacleChain.cpp
    src/grid/StaticClearanceGrid.cpp
    src/grid/CompositeGrid.cpp
)
target_include_directories(pipe_grid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(pipe_planner
    src/planner/open/BinaryHeap.cpp
    src/planner/open/LazyBinaryHeap.cpp
    src/planner/open/PairingHeap.cpp
    src/planner/open/LPAOpenSet.cpp
    src/planner/open/BucketHeap.cpp
    src/planner/low/AStar.cpp
    src/planner/low/DirectedAStar.cpp
    src/planner/low/ThetaStar.cpp
    src/planner/low/LazyThetaStar.cpp
    src/planner/low/ARAStar.cpp
    src/planner/low/ANAStar.cpp
    src/heuristic/Dijkstra.cpp
)
target_include_directories(pipe_planner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pipe_planner PUBLIC pipe_grid)

add_subdirectory(tools)

enable_testing()

add_executable(test_grid tests/test_grid.cpp)
target_link_libraries(test_grid PRIVATE pipe_grid)
add_test(NAME grid_unit COMMAND test_grid)

add_executable(test_open_set tests/test_open_set.cpp)
target_link_libraries(test_open_set PRIVATE pipe_planner)
add_test(NAME open_set_unit COMMAND test_open_set)

add_executable(test_astar tests/test_astar.cpp)
target_link_libraries(test_astar PRIVATE pipe_planner)
add_test(NAME astar_unit COMMAND test_astar)

add_executable(test_anytime tests/test_anytime.cpp)
target_link_libraries(test_anytime PRIVATE pipe_planner)
add_test(NAME anytime_unit COMMAND test_anytime)

add_executable(test_heuristic tests/test_heuristic.cpp)
target_link_libraries(test_heuristic PRIVATE pipe_planner)
add_test(NAME heuristic_unit COMMAND test_heuristic)

add_executable(test_motion tests/test_motion.cpp)
target_link_libraries(test_motion PRIVATE pipe_planner)
add_test(NAME motion_unit COMMAND test_motion)

add_executable(test_heading tests/test_heading.cpp)
target_link_libraries(test_heading PRIVATE pipe_planner)
add_test(NAME heading_unit COMMAND test_heading)
