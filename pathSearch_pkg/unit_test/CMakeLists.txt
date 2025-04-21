cmake_minimum_required(VERSION 3.10)
project(BaseFunctionsTest)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PARENT_DIR "/home/admin123456/Desktop/work/MAPF_Pipeline/pathSearch_pkg")

# Add library paths
link_directories(/lib/x86_64-linux-gnu)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,/lib/x86_64-linux-gnu")

# Find required packages
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS graph)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# FLANN configuration
find_path(FLANN_INCLUDE_DIR flann/flann.hpp
    PATHS /usr/include
    PATH_SUFFIXES flann
)
find_library(FLANN_LIBRARY
    NAMES flann
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu
)

# LZ4 configuration
find_path(LZ4_INCLUDE_DIR lz4.h
    PATHS /usr/include
)
find_library(LZ4_LIBRARY
    NAMES lz4
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu
)

if(NOT FLANN_INCLUDE_DIR OR NOT FLANN_LIBRARY)
    message(FATAL_ERROR "FLANN library not found. Please install libflann-dev")
endif()

if(NOT LZ4_INCLUDE_DIR OR NOT LZ4_LIBRARY)
    message(FATAL_ERROR "LZ4 library not found. Please install liblz4-dev")
endif()

# message(STATUS "FLANN_LIBRARY: ${FLANN_LIBRARY}")
# message(STATUS "LZ4_LIBRARY: ${LZ4_LIBRARY}")

# Set include directories globally
include_directories(
    ${CMAKE_SOURCE_DIR}/..    # This will allow includes relative to project root
    ${Eigen3_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${FLANN_INCLUDE_DIR}
    ${LZ4_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${PARENT_DIR}/include
)

# Add test executables
add_executable(base_functions_test test_base_functions.cpp)
add_executable(grid_test test_grid.cpp
    ${PARENT_DIR}/include/BaseHead.h
    ${PARENT_DIR}/include/NeighbourTree.h
    ${PARENT_DIR}/src/NeighbourTree.cpp
    ${PARENT_DIR}/include/SearchGrid.h
    ${PARENT_DIR}/src/SearchGrid.cpp
)
add_executable(node_test test_node.cpp
    ${PARENT_DIR}/include/BaseHead.h
    ${PARENT_DIR}/include/SearchNode.h
    ${PARENT_DIR}/src/SearchNode.cpp
)
add_executable(heap_test test_heap.cpp
    ${PARENT_DIR}/include/BaseHead.h
    ${PARENT_DIR}/include/SearchNode.h
    ${PARENT_DIR}/src/SearchNode.cpp
    ${PARENT_DIR}/include/UnsortedPriorityQueue.h
    ${PARENT_DIR}/src/UnsortedPriorityQueue.cpp
    ${PARENT_DIR}/include/SearchHeap.h
    ${PARENT_DIR}/src/SearchHeap.cpp
)
add_executable(astar_test test_astar.cpp
    ${PARENT_DIR}/include/BaseHead.h
    ${PARENT_DIR}/include/SearchNode.h
    ${PARENT_DIR}/src/SearchNode.cpp
    ${PARENT_DIR}/include/SearchHeap.h
    ${PARENT_DIR}/src/SearchHeap.cpp
    ${PARENT_DIR}/include/SearchGrid.h
    ${PARENT_DIR}/src/SearchGrid.cpp
    ${PARENT_DIR}/include/NeighbourTree.h
    ${PARENT_DIR}/src/NeighbourTree.cpp
    ${PARENT_DIR}/include/UnsortedPriorityQueue.h
    ${PARENT_DIR}/src/UnsortedPriorityQueue.cpp
    ${PARENT_DIR}/include/AstarSearch.h
    ${PARENT_DIR}/src/AstarSearch.cpp
)
add_executable(task_manager_test test_task_manager.cpp
    ${PARENT_DIR}/include/BaseHead.h
    ${PARENT_DIR}/include/TaskManager.h
)
add_executable(
    network_graph_test test_network_graph.cpp
    ${PARENT_DIR}/include/BaseHead.h
    ${PARENT_DIR}/include/NetworkUtils.h
    ${PARENT_DIR}/src/NetworkUtils.cpp
)
add_executable(
    neighbour_tree_test test_neighbour_tree.cpp
    ${PARENT_DIR}/include/BaseHead.h
    ${PARENT_DIR}/include/NeighbourTree.h
    ${PARENT_DIR}/src/NeighbourTree.cpp
)
add_executable(constraint_selector_test test_constraint_selector.cpp
    ${PARENT_DIR}/include/BaseHead.h
    ${PARENT_DIR}/include/ConstraintSelector.h
    ${PARENT_DIR}/src/ConstraintSelector.cpp
    ${PARENT_DIR}/include/NetworkUtils.h
    ${PARENT_DIR}/src/NetworkUtils.cpp
)
add_executable(set_planner_test test_set_planner.cpp
    ${PARENT_DIR}/include/BaseHead.h
    ${PARENT_DIR}/include/SetPlanner.h
    ${PARENT_DIR}/src/SetPlanner.cpp
    ${PARENT_DIR}/include/TaskManager.h
    ${PARENT_DIR}/include/AstarSearch.h
    ${PARENT_DIR}/src/AstarSearch.cpp
    ${PARENT_DIR}/include/SearchGrid.h
    ${PARENT_DIR}/src/SearchGrid.cpp
    ${PARENT_DIR}/include/SearchNode.h
    ${PARENT_DIR}/src/SearchNode.cpp
    ${PARENT_DIR}/include/SearchHeap.h
    ${PARENT_DIR}/src/SearchHeap.cpp
    ${PARENT_DIR}/include/NeighbourTree.h
    ${PARENT_DIR}/src/NeighbourTree.cpp
    ${PARENT_DIR}/include/UnsortedPriorityQueue.h
    ${PARENT_DIR}/src/UnsortedPriorityQueue.cpp
)
add_executable(cbs_test test_cbs.cpp
    ${PARENT_DIR}/include/BaseHead.h
    ${PARENT_DIR}/include/ConflictBasedSearch.h
    ${PARENT_DIR}/src/ConflictBasedSearch.cpp
    ${PARENT_DIR}/include/ConstraintSelector.h
    ${PARENT_DIR}/src/ConstraintSelector.cpp
    ${PARENT_DIR}/include/NetworkUtils.h
    ${PARENT_DIR}/src/NetworkUtils.cpp
    ${PARENT_DIR}/include/SetPlanner.h
    ${PARENT_DIR}/src/SetPlanner.cpp
    ${PARENT_DIR}/include/TaskManager.h
    ${PARENT_DIR}/include/AstarSearch.h
    ${PARENT_DIR}/src/AstarSearch.cpp
    ${PARENT_DIR}/include/SearchGrid.h
    ${PARENT_DIR}/src/SearchGrid.cpp
    ${PARENT_DIR}/include/SearchNode.h
    ${PARENT_DIR}/src/SearchNode.cpp
    ${PARENT_DIR}/include/SearchHeap.h
    ${PARENT_DIR}/src/SearchHeap.cpp
    ${PARENT_DIR}/include/NeighbourTree.h
    ${PARENT_DIR}/src/NeighbourTree.cpp
    ${PARENT_DIR}/include/UnsortedPriorityQueue.h
    ${PARENT_DIR}/src/UnsortedPriorityQueue.cpp
)

# Link libraries for base_functions_test
target_link_libraries(base_functions_test
    PRIVATE
    GTest::GTest
    GTest::Main
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# Link libraries for grid_test
target_link_libraries(grid_test
    PRIVATE
    GTest::GTest
    GTest::Main
    Eigen3::Eigen
    ${FLANN_LIBRARY}
    ${LZ4_LIBRARY}
)

# Link libraries for node_test
target_link_libraries(node_test
    PRIVATE
    GTest::GTest
    GTest::Main
    Eigen3::Eigen
)

# Link libraries for heap_test
target_link_libraries(heap_test
    PRIVATE
    GTest::GTest
    GTest::Main
    Boost::graph
    Eigen3::Eigen
)

# Link libraries for astar_test
target_link_libraries(astar_test
    PRIVATE
    GTest::GTest
    GTest::Main
    Eigen3::Eigen
    Boost::graph
    OpenMP::OpenMP_CXX
    ${FLANN_LIBRARY}
    ${LZ4_LIBRARY}
)

# Link libraries for task_manager_test
target_link_libraries(task_manager_test
    PRIVATE
    GTest::GTest
    GTest::Main
    Eigen3::Eigen
)

# Link libraries for network_graph_test
target_link_libraries(network_graph_test
    PRIVATE
    GTest::GTest
    GTest::Main
    Eigen3::Eigen
    Boost::graph
)

# Link libraries for neighbour_tree_test
target_link_libraries(neighbour_tree_test
    PRIVATE
    GTest::GTest
    GTest::Main
    Eigen3::Eigen
    ${FLANN_LIBRARY}
    ${LZ4_LIBRARY}
)

# Link libraries for constraint_selector_test
target_link_libraries(constraint_selector_test
    PRIVATE
    GTest::GTest
    GTest::Main
    Eigen3::Eigen
    Boost::graph
)

# Link libraries for set_planner_test
target_link_libraries(set_planner_test
    PRIVATE
    GTest::GTest
    GTest::Main
    Eigen3::Eigen
    Boost::graph
    ${FLANN_LIBRARY}
    ${LZ4_LIBRARY}
)

# Link libraries for cbs_test
target_link_libraries(cbs_test
    PRIVATE
    GTest::GTest
    GTest::Main
    Eigen3::Eigen
    Boost::graph
    OpenMP::OpenMP_CXX
    ${FLANN_LIBRARY}
    ${LZ4_LIBRARY}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
)

# Enable testing
enable_testing()
add_test(NAME BaseFunctionsTest COMMAND base_functions_test)
add_test(NAME GridTest COMMAND grid_test)
add_test(NAME NodeTest COMMAND node_test)
add_test(NAME HeapTest COMMAND heap_test)
add_test(NAME AstarTest COMMAND astar_test)
add_test(NAME TaskManagerTest COMMAND task_manager_test)
add_test(NAME NetworkGraphTest COMMAND network_graph_test)
add_test(NAME NeighbourTreeTest COMMAND neighbour_tree_test)
add_test(NAME ConstraintSelectorTest COMMAND constraint_selector_test)
add_test(NAME SetPlannerTest COMMAND set_planner_test)
add_test(NAME CbsTest COMMAND cbs_test) 