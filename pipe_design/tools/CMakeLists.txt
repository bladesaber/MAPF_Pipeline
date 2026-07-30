# C++ tools: file-bridge formats / runners (see tools/ROADMAP.md).
# Does not modify pipe_grid / pipe_planner sources.

add_library(pipe_tools_formats
    formats/SearchInstance.cpp
    formats/SearchReport.cpp
)
target_include_directories(pipe_tools_formats
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(pipe_tools_formats PUBLIC pipe_grid)

add_library(pipe_tools_runner
    runners/LowLevelRunner.cpp
)
target_link_libraries(pipe_tools_runner PUBLIC pipe_tools_formats pipe_planner)

add_executable(pipe_run_low runners/main_low.cpp)
target_link_libraries(pipe_run_low PRIVATE pipe_tools_runner)

add_executable(pipe_dump_heuristic runners/main_heuristic.cpp)
target_link_libraries(pipe_dump_heuristic PRIVATE pipe_tools_runner)
