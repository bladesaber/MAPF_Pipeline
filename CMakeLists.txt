cmake_minimum_required(VERSION 3.0.2)
project(mapf_pipeline)
set(CMAKE_CXX_STANDARD 17)

set(PACKAGE_TOP_DIR "/home/admin123456/Desktop/work/MAPF_Pipeline")
message("PACKAGE_TOP_DIR: " ${PACKAGE_TOP_DIR})
set(ThirdPartyDir ${PACKAGE_TOP_DIR}/thirdparty)
message("ThirdPartyDir: " ${ThirdPartyDir})

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
# message("Boost_INCLUDE_DIRS: " ${Boost_INCLUDE_DIRS})
# message("PCL_INCLUDE_DIR: " ${PCL_INCLUDE_DIRS})
# message("PCL_LIBRARY_DIR: " ${PCL_LIBRARIES})

#set(OpenFoam_Project_DIR $ENV{WM_PROJECT_DIR})
#set(OpenFoam_SRC_DIR $ENV{FOAM_SRC})
#set(OpenFoam_LIBBIN_DIR $ENV{FOAM_LIBBIN})
#set(OpenFoam_USER_LIBBIN_DIR $ENV{FOAM_USER_LIBBIN})
#message("OpenFoam_Project_DIR: " ${OpenFoam_Project_DIR})
#message("OpenFoam_SRC_DIR: " ${OpenFoam_SRC_DIR})
#message("OpenFoam_LIBBIN_DIR: " ${OpenFoam_LIBBIN_DIR})
#message("OpenFoam_USER_LIBBIN_DIR: " ${OpenFoam_USER_LIBBIN_DIR})

add_subdirectory(${ThirdPartyDir}/pybind11)

include_directories(
        ${ThirdPartyDir}/pybind11
        ${ThirdPartyDir}/pybind11/tools
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        ${PACKAGE_TOP_DIR}/include

#        ${OpenFoam_SRC_DIR}/OpenFOAM/lnInclude
#        ${OpenFoam_SRC_DIR}/finiteVolume/lnInclude
#        ${OpenFoam_SRC_DIR}/meshTools/lnInclude
#        ${OpenFoam_SRC_DIR}/OSspecific/POSIX/lnInclude
)

link_directories(
        ${PCL_LIBRARY_DIRS}
#        ${OpenFoam_LIBBIN_DIR}
#        ${OpenFoam_LIBBIN_DIR}/dummy
#        ${OpenFoam_LIBBIN_DIR}/openmpi-system
#        ${OpenFoam_USER_LIBBIN_DIR}
)

add_definitions(${PCL_DEFINITIONS})

pybind11_add_module(
        mapf_pipeline ${PACKAGE_TOP_DIR}/src/driver.cpp
        ${PACKAGE_TOP_DIR}/src/math_utils.cpp
        ${PACKAGE_TOP_DIR}/src/pcl_utils.cpp
        ${PACKAGE_TOP_DIR}/src/grid_utils.cpp
        ${PACKAGE_TOP_DIR}/src/state_utils.cpp
        ${PACKAGE_TOP_DIR}/src/collision_utils.cpp
        ${PACKAGE_TOP_DIR}/src/astar_algo.cpp
        ${PACKAGE_TOP_DIR}/src/group_astar_algo.cpp
        ${PACKAGE_TOP_DIR}/src/cbs_algo.cpp
)

target_link_libraries(
        mapf_pipeline PRIVATE
        pybind11::module
        Eigen3::Eigen
        ${Boost_LIBRARIES}
        ${PCL_LIBRARIES}
)
