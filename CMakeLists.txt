cmake_minimum_required(VERSION 3.0.2)
project(mapf_pipeline)

set(PACKAGE_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
message("PACKAGE_TOP_DIR: " ${PACKAGE_TOP_DIR})

set(Thirdparty_DIR ${PACKAGE_TOP_DIR}/Thirdparty)
message("Thirdparty_DIR: " ${Thirdparty_DIR})

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
message("Boost_INCLUDE_DIRS: " ${Boost_INCLUDE_DIRS})

add_subdirectory(${Thirdparty_DIR}/pybind11)

### Version Setting
# set(Include_DIR ${PACKAGE_TOP_DIR}/include/version_1)
# set(Src_DIR ${PACKAGE_TOP_DIR}/src/version_1)

# set(Include_DIR ${PACKAGE_TOP_DIR}/include/version_2)
# set(Src_DIR ${PACKAGE_TOP_DIR}/src/version_2)

set(Include_DIR ${PACKAGE_TOP_DIR}/include/version_3)
set(Src_DIR ${PACKAGE_TOP_DIR}/src/version_3)

message("Include_DIR: " ${Include_DIR})
message("Src_DIR: " ${Src_DIR})

include_directories(
#        ${Thirdparty_DIR}/pybind11
#        ${Thirdparty_DIR}/pybind11/tools

        include
        ${Include_DIR}
        ${Boost_INCLUDE_DIRS}
)

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
        ${Src_DIR}/angleAstar.cpp
        ${Src_DIR}/cbs.cpp
        ${Src_DIR}/kdtree.c
        )
target_link_libraries(mapf_pipeline PRIVATE
        pybind11::module
        Eigen3::Eigen
        ${Boost_LIBRARIES}
        )