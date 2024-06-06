cmake_minimum_required(VERSION 3.0.2)
project(mapf_pipeline)

# g2o need super new C++ Version Support
set(CMAKE_CXX_STANDARD 17)

set(PACKAGE_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
message("PACKAGE_TOP_DIR: " ${PACKAGE_TOP_DIR})

set(Thirdparty_DIR ${PACKAGE_TOP_DIR}/Thirdparty)
message("Thirdparty_DIR: " ${Thirdparty_DIR})

find_package(PCL 1.14 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

# message("Boost_INCLUDE_DIRS: " ${Boost_INCLUDE_DIRS})
# message("PCL_INCLUDE_DIR: " ${PCL_INCLUDE_DIRS})
# message("PCL_LIBRARY_DIR: " ${PCL_LIBRARIES})

# ---------------------- Version Setting
set(IncludeDir ${PACKAGE_TOP_DIR}/include/version_10)
set(SrcDir ${PACKAGE_TOP_DIR}/src/version_10)
message("IncludeDir: " ${IncludeDir})
message("SrcDir: " ${SrcDir})
# --------------------------------------------

add_subdirectory(${Thirdparty_DIR}/pybind11)

include_directories(
        ${Thirdparty_DIR}/pybind11
        ${Thirdparty_DIR}/pybind11/tools
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        ${IncludeDir}
)

link_directories(
        ${PCL_LIBRARY_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

pybind11_add_module(
        mapf_pipeline ${PACKAGE_TOP_DIR}/include/asdk/driver_fuck.cpp
#        ${Src_DIR}/utils.cpp
#        ${Src_DIR}/instance.cpp
#        ${Src_DIR}/constrainTable.cpp
#        ${Src_DIR}/kdtree.c
#        ${Src_DIR}/pather/kdtree_xyzra.cpp
#        ${Src_DIR}/planner/AstarSolver.cpp
#        ${Src_DIR}/planner/groupAstarSolver.cpp
#        ${Src_DIR}/cbs/cbs_node.cpp
)

target_link_libraries(
        mapf_pipeline PRIVATE
        pybind11::module
        Eigen3::Eigen
        ${Boost_LIBRARIES}
        ${PCL_LIBRARIES}
)