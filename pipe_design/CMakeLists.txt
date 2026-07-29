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
    src/planner/open/BucketHeap.cpp
)
target_include_directories(pipe_planner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pipe_planner PUBLIC pipe_grid)

enable_testing()

add_executable(test_grid tests/test_grid.cpp)
target_link_libraries(test_grid PRIVATE pipe_grid)
add_test(NAME grid_unit COMMAND test_grid)

add_executable(test_open_set tests/test_open_set.cpp)
target_link_libraries(test_open_set PRIVATE pipe_planner)
add_test(NAME open_set_unit COMMAND test_open_set)
