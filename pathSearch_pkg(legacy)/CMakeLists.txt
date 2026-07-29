cmake_minimum_required(VERSION 3.10.0)
project(MapfPkg)
set(CMAKE_CXX_STANDARD 17)

# cmake -DCMAKE_BUILD_TYPE=Release ..
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add configuration types if not defined
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

# Set compile definitions based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()

cmake_policy(SET CMP0167 NEW)

set(PACKAGE_TOP_DIR "/home/admin123456/Desktop/work/MAPF_Pipeline/pathSearch_pkg")
message("PACKAGE_TOP_DIR: " ${PACKAGE_TOP_DIR})
set(ThirdPartyDir ${PACKAGE_TOP_DIR}/third_party)
message("ThirdPartyDir: " ${ThirdPartyDir})

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS graph)
# find_package(LZ4 REQUIRED)

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

add_subdirectory(${ThirdPartyDir}/pybind11)

include_directories(
    ${Eigen3_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${FLANN_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${ThirdPartyDir}/pybind11
    ${ThirdPartyDir}/pybind11/tools
    ${PACKAGE_TOP_DIR}/include
)

link_directories(
    ${FLANN_LIBRARIES}
)

add_definitions(${PCL_DEFINITIONS})

pybind11_add_module(
    MapfPkg ${PACKAGE_TOP_DIR}/src/bind_py.cpp
    ${PACKAGE_TOP_DIR}/src/AstarSearch.cpp
    ${PACKAGE_TOP_DIR}/src/ConstraintSelector.cpp
    ${PACKAGE_TOP_DIR}/src/ConflictBasedSearch.cpp
    ${PACKAGE_TOP_DIR}/src/NeighbourTree.cpp
    ${PACKAGE_TOP_DIR}/src/NetworkUtils.cpp
    ${PACKAGE_TOP_DIR}/src/SearchGrid.cpp
    ${PACKAGE_TOP_DIR}/src/SearchHeap.cpp
    ${PACKAGE_TOP_DIR}/src/SearchNode.cpp
    ${PACKAGE_TOP_DIR}/src/SetPlanner.cpp
    ${PACKAGE_TOP_DIR}/src/UnsortedPriorityQueue.cpp
)

target_link_libraries(
    MapfPkg PRIVATE
    pybind11::module
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${Boost_LIBRARIES}
    ${FLANN_LIBRARY}
    ${LZ4_LIBRARY}
)
