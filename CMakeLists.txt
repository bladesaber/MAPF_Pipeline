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

include_directories(
#        ${Thirdparty_DIR}/pybind11
#        ${Thirdparty_DIR}/pybind11/tools
        include
        ${Boost_INCLUDE_DIRS}
)

#add_executable(mapf_pipeline src/main.cpp)
#target_link_libraries(mapf_pipeline PRIVATE
#        Eigen3::Eigen
#        ${Boost_LIBRARIES}
#        )

### Version 0.1
# pybind11_add_module(mapf_pipeline src/driver.cpp
#         src/spaceTimeAstar.cpp
#         src/instance.cpp
#         src/constraintTable.cpp
#         src/cbsNode.cpp
#         src/cbs.cpp
#         )
# target_link_libraries(mapf_pipeline PRIVATE
#         pybind11::module
#         Eigen3::Eigen
#         ${Boost_LIBRARIES}
#         )

### Version 0.2
pybind11_add_module(mapf_pipeline src/Aux_driver.cpp
        src/Aux_utils.cpp
        src/Aux_dubins.cpp
        src/Aux_continusAstar.cpp
        )
target_link_libraries(mapf_pipeline PRIVATE
        pybind11::module
        Eigen3::Eigen
        ${Boost_LIBRARIES}
        )