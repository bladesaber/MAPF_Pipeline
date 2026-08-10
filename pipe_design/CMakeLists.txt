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
    src/planner/low/LPAStar.cpp
    src/planner/low/BidirectionalAStar.cpp
    src/planner/low/FocalAStar.cpp
    src/planner/low/AdaptiveStepAStar.cpp
    src/planner/multi/GroupSession.cpp
    src/planner/multi/PrioritySequencer.cpp
    src/planner/high/CBS.cpp
    src/conflict/PathGraph.cpp
    src/conflict/ConflictAnalyzer.cpp
    src/conflict/NearTerminalAnalyzer.cpp
    src/conflict/SampleKdTree.cpp
    src/heuristic/Dijkstra.cpp
)
target_include_directories(pipe_planner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pipe_planner PUBLIC pipe_grid)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(pipe_planner PUBLIC OpenMP::OpenMP_CXX)
endif()

add_subdirectory(tools)

enable_testing()

add_executable(test_grid tests/test_grid.cpp)
target_link_libraries(test_grid PRIVATE pipe_grid)
add_test(NAME grid_unit COMMAND test_grid)

add_executable(test_bvh tests/test_bvh.cpp)
target_link_libraries(test_bvh PRIVATE pipe_grid)
add_test(NAME bvh_unit COMMAND test_bvh)

add_executable(test_open_set tests/test_open_set.cpp)
target_link_libraries(test_open_set PRIVATE pipe_planner)
add_test(NAME open_set_unit COMMAND test_open_set)

add_executable(test_astar tests/test_astar.cpp)
target_link_libraries(test_astar PRIVATE pipe_planner)
add_test(NAME astar_unit COMMAND test_astar)

add_executable(test_bidirectional tests/test_bidirectional.cpp)
target_link_libraries(test_bidirectional PRIVATE pipe_planner)
add_test(NAME bidirectional_unit COMMAND test_bidirectional)

add_executable(test_focal tests/test_focal.cpp)
target_link_libraries(test_focal PRIVATE pipe_planner)
add_test(NAME focal_unit COMMAND test_focal)

add_executable(test_adaptive_step tests/test_adaptive_step.cpp)
target_link_libraries(test_adaptive_step PRIVATE pipe_planner)
add_test(NAME adaptive_step_unit COMMAND test_adaptive_step)

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

add_executable(test_multi tests/test_multi.cpp)
target_link_libraries(test_multi PRIVATE pipe_planner)
add_test(NAME multi_unit COMMAND test_multi)

add_executable(test_cbs tests/test_cbs.cpp)
target_link_libraries(test_cbs PRIVATE pipe_planner)
add_test(NAME cbs_unit COMMAND test_cbs)

add_executable(test_sample_kdtree tests/test_sample_kdtree.cpp)
target_link_libraries(test_sample_kdtree PRIVATE pipe_planner)
add_test(NAME sample_kdtree_unit COMMAND test_sample_kdtree)

add_executable(test_conflict_normalize tests/test_conflict_normalize.cpp)
target_link_libraries(test_conflict_normalize PRIVATE pipe_planner)
add_test(NAME conflict_normalize_unit COMMAND test_conflict_normalize)

add_executable(test_near_terminal tests/test_near_terminal.cpp)
target_link_libraries(test_near_terminal PRIVATE pipe_planner)
add_test(NAME near_terminal_unit COMMAND test_near_terminal)

add_executable(test_mst tests/test_mst.cpp)
target_include_directories(test_mst PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME mst_unit COMMAND test_mst)

find_package(Boost REQUIRED)
add_executable(bench_path_graph tests/bench_path_graph.cpp)
target_link_libraries(bench_path_graph PRIVATE pipe_planner Boost::boost)

find_path(FLANN_INCLUDE_DIR flann/flann.hpp)
find_library(FLANN_CPP_LIBRARY NAMES flann_cpp)
find_library(LZ4_LIBRARY NAMES lz4)
if(FLANN_INCLUDE_DIR AND FLANN_CPP_LIBRARY AND LZ4_LIBRARY)
  add_executable(bench_sample_kdtree tests/bench_sample_kdtree.cpp)
  target_include_directories(bench_sample_kdtree PRIVATE ${FLANN_INCLUDE_DIR})
  target_link_libraries(bench_sample_kdtree PRIVATE pipe_planner ${FLANN_CPP_LIBRARY} ${LZ4_LIBRARY})
else()
  message(WARNING "FLANN/LZ4 not found; bench_sample_kdtree skipped")
endif()

