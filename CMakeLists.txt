cmake_minimum_required(VERSION 3.0.2)
project(mapf_pipeline)

# g2o need super new C++ Version Support
set(CMAKE_CXX_STANDARD 17)

set(PACKAGE_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
message("PACKAGE_TOP_DIR: " ${PACKAGE_TOP_DIR})

set(Thirdparty_DIR ${PACKAGE_TOP_DIR}/Thirdparty)
message("Thirdparty_DIR: " ${Thirdparty_DIR})

### I don't know why it can not work
## find_package(PCL 1.8 REQUIRED)
#set(PCL_INCLUDE_DIRS "/usr/include/pcl-1.8")
### please don't use this
## file(GLOB PCL_LIBRARIES "/usr/lib/libpcl*.so*")
#file(GLOB PCL_LIBRARIES "/usr/lib/x86_64-linux-gnu/libpcl*.so*")

# find_package(VTK 6.3 REQUIRED)
# set(VTK_INCLUDE_DIRS "/usr/include/vtk-6.3")
# file(GLOB VTK_LIBRARY_DIRS "/usr/lib/libvtk*.so*")

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

LIST( APPEND CMAKE_MODULE_PATH ${PACKAGE_TOP_DIR}/Thirdparty/g2o/cmake_modules)
set(G2O_ROOT /usr/local/include/g2o)
find_package(G2O REQUIRED)
find_package(Ceres REQUIRED)

message("Boost_INCLUDE_DIRS: " ${Boost_INCLUDE_DIRS})
# message("PCL_INCLUDE_DIR: " ${PCL_INCLUDE_DIRS})
# message("PCL_LIBRARY_DIR: " ${PCL_LIBRARIES})

### ---------------------- Version Setting
set(Include_DIR ${PACKAGE_TOP_DIR}/include/version_9)
set(Src_DIR ${PACKAGE_TOP_DIR}/src/version_9)

message("Include_DIR: " ${Include_DIR})
message("Src_DIR: " ${Src_DIR})
### --------------------------------------------

add_subdirectory(${Thirdparty_DIR}/pybind11)

include_directories(
        ${Thirdparty_DIR}/pybind11
        ${Thirdparty_DIR}/pybind11/tools

        ${Include_DIR}
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        # ${PCL_INCLUDE_DIRS}
        # ${VTK_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}

        ${Include_DIR}/smooth
        ${Include_DIR}/planner
        ${Include_DIR}/pather
        ${Include_DIR}/cbs
#        ${Include_DIR}/tightSpringer
)

link_directories(
        # ${PCL_LIBRARY_DIRS}
        # ${VTK_LIBRARIES}
        ${G2O_LIBRARIES}
)

# add_definitions(${PCL_DEFINITIONS})

### Just For Debug
#add_executable(mapf_pipeline src/main.cpp)
#target_link_libraries(mapf_pipeline PRIVATE
#        Eigen3::Eigen
#        ${Boost_LIBRARIES}
#        )

### Version_1
# pybind11_add_module(mapf_pipeline ${Src_DIR}/driver.cpp
#         ${Src_DIR}/spaceTimeAstar.cpp
#         ${Src_DIR}/instance.cpp
#         ${Src_DIR}/constraintTable.cpp
#         ${Src_DIR}/cbsNode.cpp
#         ${Src_DIR}/cbs.cpp
#         )
# target_link_libraries(mapf_pipeline PRIVATE
#         pybind11::module
#         Eigen3::Eigen
#         ${Boost_LIBRARIES}
#         )

### Version 2
# pybind11_add_module(mapf_pipeline ${Src_DIR}/Aux_driver.cpp
#         ${Src_DIR}/Aux_utils.cpp
#         ${Src_DIR}/Aux_dubins.cpp
#         ${Src_DIR}/Aux_continusAstar.cpp
#         )
# target_link_libraries(mapf_pipeline PRIVATE
#         pybind11::module
#         Eigen3::Eigen
#         ${Boost_LIBRARIES}
#         )

### Version 3
pybind11_add_module(mapf_pipeline ${Src_DIR}/driver.cpp
        ${Src_DIR}/utils.cpp
        ${Src_DIR}/instance.cpp
        ${Src_DIR}/constrainTable.cpp
        # ${Src_DIR}/angleAstar.cpp
        # ${Src_DIR}/cbs.cpp
        ${Src_DIR}/kdtree.c
        # ${Src_DIR}/kdtreeWrapper.cpp

        # ${Src_DIR}/smoother.cpp
        # ${Src_DIR}/smoother_random.cpp

#        ${Src_DIR}/pather/groupPath.cpp
        ${Src_DIR}/pather/kdtree_xyzra.cpp

#        ${Src_DIR}/smooth/smootherSE3_g2o.cpp
#        ${Src_DIR}/smooth/smootherXYZ_g2o.cpp

        ${Src_DIR}/planner/AstarSolver.cpp
#        ${Src_DIR}/planner/spanningTree_groupSolver.cpp
        ${Src_DIR}/planner/groupAstarSolver.cpp

        ${Src_DIR}/cbs/cbs_node.cpp
#        ${Src_DIR}/cbs/cbs_solver.cpp

#        ${Src_DIR}/tightSpringer/springSmoother.cpp
        )
target_link_libraries(mapf_pipeline PRIVATE
        pybind11::module
        Eigen3::Eigen
        ${Boost_LIBRARIES}
        # ${PCL_LIBRARIES}
        # ${VTK_LIBRARIES}

        ${G2O_LIBRARIES} # it do not work, it cause pybind undefine symbol error??
        g2o_core
        g2o_stuff
        g2o_solver_cholmod
        g2o_solver_csparse
        g2o_solver_eigen
        g2o_solver_dense
        g2o_solver_pcg
        g2o_solver_slam2d_linear
        g2o_solver_structure_only
        # g2o_types_data
        # g2o_types_icp
        # g2o_types_sba
        # g2o_types_sclam2d
        # g2o_types_sim3
        # g2o_types_slam2d
        # g2o_types_slam2d_addons
        # g2o_types_slam3d
        # g2o_types_slam3d_addons

        Ceres::ceres
        )