#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "BaseHead.h"
#include "PyWrap.h"
#include "SearchGrid.h"
#include "AstarSearch.h"
#include "TaskManager.h"
#include "SetPlanner.h"
#include "ConflictBasedSearch.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pybind_modules
{
    void bind_package(py::module_ &m)
    {
        // ------ LocXyz
        py::class_<LocXyz>(m, "LocXyz")
            .def(py::init<double, double, double>())
            .def_readwrite("x", &LocXyz::x)
            .def_readwrite("y", &LocXyz::y)
            .def_readwrite("z", &LocXyz::z);
        
        // ------ PathData
        py::class_<PathData>(m, "PathData")
            .def(py::init<>())
            .def_readonly("name", &PathData::m_name)
            .def_readonly("path", &PathData::m_path)
            .def_readonly("radius", &PathData::m_radius)
            .def_readonly("cost", &PathData::m_cost)
            .def_readonly("time_cost", &PathData::time_cost)
            .def("estimate_cost", &PathData::estimate_cost)
            .def("get_size", &PathData::get_size);

        // ------ PathSetData
        py::class_<PathSetData>(m, "PathSetData")
            .def(py::init<>())
            .def_readonly("paths", &PathSetData::m_paths)
            .def("get_size", &PathSetData::get_size)
            .def("get_max_radius", &PathSetData::get_max_radius);
        
        // ------ CbsResult
        py::class_<CbsResult>(m, "CbsResult")
            .def(py::init<>())
            .def_readonly("m_conflict_size", &CbsResult::m_conflict_size)
            .def_readonly("paths", &CbsResult::paths);

        // ------ Conflict
        py::class_<Conflict>(m, "Conflict")
            .def(py::init<>())
            .def_readonly("group0", &Conflict::group0)
            .def_readonly("loc0", &Conflict::loc0)
            .def_readonly("radius0", &Conflict::radius0)
            .def_readonly("group1", &Conflict::group1)
            .def_readonly("loc1", &Conflict::loc1)
            .def_readonly("radius1", &Conflict::radius1)
            .def_readonly("conflict_cost", &Conflict::conflict_cost);
        
        // ------ ConflictSet
        py::class_<ConflictSet>(m, "ConflictSet")
            .def(py::init<>())
            .def_readonly("conflicts", &ConflictSet::conflicts)
            .def_readonly("conflict_index_map", &ConflictSet::conflict_index_map)
            .def_readonly("m_conflict_size", &ConflictSet::m_conflict_size)
            .def_readonly("m_group_cost", &ConflictSet::m_group_cost);

        // ------ Constraint
        py::class_<Constraint>(m, "Constraint")
            .def(py::init<>())
            .def_readonly("run_group0", &Constraint::run_group0)
            .def_readonly("obs0_loc", &Constraint::obs0_loc)
            .def_readonly("obs0_radius", &Constraint::obs0_radius)
            .def_readonly("run_group1", &Constraint::run_group1)
            .def_readonly("obs1_loc", &Constraint::obs1_loc)
            .def_readonly("obs1_radius", &Constraint::obs1_radius);

        // ------ ObstacleData
        py::class_<ObstacleData>(m, "ObstacleData")
            .def(py::init<const LocXyz &, const double>())
            .def_readonly("loc", &ObstacleData::loc)
            .def_readonly("radius", &ObstacleData::radius);
        
        // ------ SearchGrid (base class)
        py::class_<GridSpace::SearchGrid>(m, "SearchGrid")
            .def("is_in_grid", &GridSpace::SearchGrid::is_in_grid, py::arg("loc"))
            .def("is_move_valid", &GridSpace::SearchGrid::is_move_valid, 
                py::arg("loc"), py::arg("threshold"), py::arg("dir_x"), py::arg("dir_y"), py::arg("dir_z"))
            .def("is_light_of_sight", &GridSpace::SearchGrid::is_light_of_sight, 
                py::arg("loc1"), py::arg("loc2"), py::arg("threshold"))
            .def("is_constrained", &GridSpace::SearchGrid::is_constrained, py::arg("loc"), py::arg("threshold") = 0.0)
            .def("get_x_size", &GridSpace::SearchGrid::get_x_size)
            .def("get_y_size", &GridSpace::SearchGrid::get_y_size)
            .def("get_z_size", &GridSpace::SearchGrid::get_z_size)
            .def("get_grid_resolution", &GridSpace::SearchGrid::get_grid_resolution);
        
        // ------ StandardGrid
        py::class_<GridSpace::StandardGrid, GridSpace::SearchGrid>(m, "StandardGrid")
            .def(py::init<uint16_t, uint16_t, uint16_t, double>())
            .def("set_obstacle", &GridSpace::StandardGrid::set_obstacle, py::arg("loc"), py::arg("weight") = 0.0)
            .def("clear_obstacle", &GridSpace::StandardGrid::clear_obstacle, py::arg("loc"))
            .def("is_obstacle", &GridSpace::StandardGrid::is_obstacle, py::arg("loc"), py::arg("threshold") = 0.0)
            .def("preprocess", &GridSpace::StandardGrid::preprocess);

        // ------ DistanceGrid
        py::class_<GridSpace::DistanceGrid, GridSpace::SearchGrid>(m, "DistanceGrid")
            .def(py::init<uint16_t, uint16_t, uint16_t, double>())
            .def("set_obstacle", &GridSpace::DistanceGrid::set_obstacle, py::arg("loc"), py::arg("distance"))
            .def("clear_obstacle", &GridSpace::DistanceGrid::clear_obstacle, py::arg("loc"))
            .def("is_obstacle", &GridSpace::DistanceGrid::is_obstacle, py::arg("loc"), py::arg("threshold") = 0.0)
            .def("preprocess", &GridSpace::DistanceGrid::preprocess)
            .def("set_obstacles", &GridSpace::DistanceGrid::set_obstacles, py::arg("locs"), py::arg("distances"));

        // ------ BitGrid
        py::class_<GridSpace::BitGrid, GridSpace::SearchGrid>(m, "BitGrid")
            .def(py::init<uint16_t, uint16_t, uint16_t, double>())
            .def("set_obstacle", &GridSpace::BitGrid::set_obstacle, py::arg("loc"), py::arg("weight") = 0.0)
            .def("clear_obstacle", &GridSpace::BitGrid::clear_obstacle, py::arg("loc"))
            .def("is_obstacle", &GridSpace::BitGrid::is_obstacle, py::arg("loc"), py::arg("threshold") = 0.0)
            .def("preprocess", &GridSpace::BitGrid::preprocess);

        // ------ SparseGrid
        py::class_<GridSpace::SparseGrid, GridSpace::SearchGrid>(m, "SparseGrid")
            .def(py::init<uint16_t, uint16_t, uint16_t, double>())
            .def("set_obstacle", &GridSpace::SparseGrid::set_obstacle, py::arg("loc"), py::arg("weight") = 0.0)
            .def("clear_obstacle", &GridSpace::SparseGrid::clear_obstacle, py::arg("loc"))
            .def("is_obstacle", &GridSpace::SparseGrid::is_obstacle, py::arg("loc"), py::arg("threshold") = 0.0)
            .def("preprocess", &GridSpace::SparseGrid::preprocess);

        // ------ TreeGrid
        py::class_<GridSpace::TreeGrid, GridSpace::SearchGrid>(m, "TreeGrid")
            .def(py::init<uint16_t, uint16_t, uint16_t, double>())
            .def("set_obstacle", &GridSpace::TreeGrid::set_obstacle, py::arg("loc"), py::arg("radius"))
            .def("clear_obstacle", &GridSpace::TreeGrid::clear_obstacle, py::arg("loc"))
            .def("is_obstacle", &GridSpace::TreeGrid::is_obstacle, py::arg("loc"), py::arg("threshold") = 0.0)
            .def("preprocess", &GridSpace::TreeGrid::preprocess);

        // ------ SearchGridSet
        py::class_<GridSpace::SearchGridSet, GridSpace::SearchGrid>(m, "SearchGridSet")
            .def(py::init<const GridSpace::SearchGrid*>())
            .def("set_obstacle", &GridSpace::SearchGridSet::set_obstacle, py::arg("loc"), py::arg("weight") = 0.0)
            .def("clear_obstacle", &GridSpace::SearchGridSet::clear_obstacle, py::arg("loc"))
            .def("is_obstacle", &GridSpace::SearchGridSet::is_obstacle, py::arg("loc"), py::arg("threshold") = 0.0)
            .def("is_constrained", &GridSpace::SearchGridSet::is_constrained, py::arg("loc"), py::arg("threshold") = 0.0)
            .def("set_constraint", &GridSpace::SearchGridSet::set_constraint, py::arg("loc"), py::arg("weight"))
            .def("preprocess", &GridSpace::SearchGridSet::preprocess);
        
        // ------ BaseSearch
        py::class_<SearchSpace::BaseSearch>(m, "BaseSearch")
            .def("reset", &SearchSpace::BaseSearch::reset)
            .def("get_radius", &SearchSpace::BaseSearch::get_radius)
            .def("set_radius", &SearchSpace::BaseSearch::set_radius);

        // ------ AstarSearch<QueueSpace::BoostHeap>
        py::class_<SearchSpace::AstarSearch<QueueSpace::BoostHeap>, SearchSpace::BaseSearch>(m, "AstarSearchBoost")
            .def(py::init<double, size_t, bool>(),
                py::arg("radius"),
                py::arg("max_iterations") = 100000,
                py::arg("with_constraint_table") = false)
            .def("reset", &SearchSpace::AstarSearch<QueueSpace::BoostHeap>::reset)
            .def("search", &SearchSpace::AstarSearch<QueueSpace::BoostHeap>::search,
                py::arg("grid"),
                py::arg("starts"),
                py::arg("goals"),
                py::arg("path"));

        // ------ AstarSearch<QueueSpace::BucketPriorityQueue>
        py::class_<SearchSpace::AstarSearch<QueueSpace::BucketPriorityQueue>, SearchSpace::BaseSearch>(m, "AstarSearchBucket")
            .def(py::init<double, size_t, bool>(),
                py::arg("radius"),
                py::arg("max_iterations") = 100000,
                py::arg("with_constraint_table") = false)
            .def("reset", &SearchSpace::AstarSearch<QueueSpace::BucketPriorityQueue>::reset)
            .def("search", &SearchSpace::AstarSearch<QueueSpace::BucketPriorityQueue>::search,
                py::arg("grid"),
                py::arg("starts"),
                py::arg("goals"),
                py::arg("path"));

        // ------ WeightedAstarSearch<QueueSpace::BoostHeap>
        py::class_<SearchSpace::WeightedAstarSearch<QueueSpace::BoostHeap>, 
                  SearchSpace::AstarSearch<QueueSpace::BoostHeap>>(m, "WeightedAstarSearchBoost")
            .def(py::init<double, size_t, double, double, bool>(),
                py::arg("radius"),
                py::arg("max_iterations"),
                py::arg("g_weight"),
                py::arg("h_weight"),
                py::arg("with_constraint_table") = false);

        // ------ WeightedAstarSearch<QueueSpace::BucketPriorityQueue>
        py::class_<SearchSpace::WeightedAstarSearch<QueueSpace::BucketPriorityQueue>,
                  SearchSpace::AstarSearch<QueueSpace::BucketPriorityQueue>>(m, "WeightedAstarSearchBucket")
            .def(py::init<double, size_t, double, double, bool>(),
                py::arg("radius"),
                py::arg("max_iterations"),
                py::arg("g_weight"),
                py::arg("h_weight"),
                py::arg("with_constraint_table") = false);

        // ------ ThetaStarSearch<QueueSpace::BoostHeap>
        py::class_<SearchSpace::ThetaStarSearch<QueueSpace::BoostHeap>,
                  SearchSpace::WeightedAstarSearch<QueueSpace::BoostHeap>>(m, "ThetaStarSearchBoost")
            .def(py::init<double, size_t, double, double>(),
                py::arg("radius"),
                py::arg("max_iterations"),
                py::arg("g_weight"),
                py::arg("h_weight"));

        // ------ ThetaStarSearch<QueueSpace::BucketPriorityQueue>
        py::class_<SearchSpace::ThetaStarSearch<QueueSpace::BucketPriorityQueue>,
                  SearchSpace::WeightedAstarSearch<QueueSpace::BucketPriorityQueue>>(m, "ThetaStarSearchBucket")
            .def(py::init<double, size_t, double, double>(),
                py::arg("radius"),
                py::arg("max_iterations"),
                py::arg("g_weight"),
                py::arg("h_weight"));

        // ------ ScaledWeightedAstarSearch<QueueSpace::BoostHeap>
        py::class_<SearchSpace::ScaledWeightedAstarSearch<QueueSpace::BoostHeap>,
                  SearchSpace::WeightedAstarSearch<QueueSpace::BoostHeap>>(m, "ScaledWeightedAstarSearchBoost")
            .def(py::init<double, size_t, double, double, double, uint32_t, bool>(),
                py::arg("radius"),
                py::arg("max_iterations"),
                py::arg("g_weight"),
                py::arg("h_weight"),
                py::arg("threshold1"),
                py::arg("scale_factor1"),
                py::arg("with_constraint_table") = false);
        
        // ------ ScaledWeightedAstarSearch<QueueSpace::BucketPriorityQueue>
        py::class_<SearchSpace::ScaledWeightedAstarSearch<QueueSpace::BucketPriorityQueue>,
                  SearchSpace::WeightedAstarSearch<QueueSpace::BucketPriorityQueue>>(m, "ScaledWeightedAstarSearchBucket")
            .def(py::init<double, size_t, double, double, double, uint32_t, bool>(),
                py::arg("radius"),
                py::arg("max_iterations"),
                py::arg("g_weight"),
                py::arg("h_weight"),
                py::arg("threshold1"),
                py::arg("scale_factor1"),
                py::arg("with_constraint_table") = false);

        // ------ DoubleScaledWeightedAstarSearch<QueueSpace::BoostHeap>
        py::class_<SearchSpace::DoubleScaledWeightedAstarSearch<QueueSpace::BoostHeap>,
                  SearchSpace::ScaledWeightedAstarSearch<QueueSpace::BoostHeap>>(m, "DoubleScaledWeightedAstarSearchBoost")
            .def(py::init<double, size_t, double, double, double, uint32_t, double, uint32_t, bool>(),
                py::arg("radius"),
                py::arg("max_iterations"),
                py::arg("g_weight"),
                py::arg("h_weight"),
                py::arg("threshold1"),
                py::arg("scale_factor1"),
                py::arg("threshold2"),
                py::arg("scale_factor2"),
                py::arg("with_constraint_table") = false);
        
        // ------ DoubleScaledWeightedAstarSearch<QueueSpace::BucketPriorityQueue>
        py::class_<SearchSpace::DoubleScaledWeightedAstarSearch<QueueSpace::BucketPriorityQueue>,
                  SearchSpace::ScaledWeightedAstarSearch<QueueSpace::BucketPriorityQueue>>(m, "DoubleScaledWeightedAstarSearchBucket")
            .def(py::init<double, size_t, double, double, double, uint32_t, double, uint32_t, bool>(),
                py::arg("radius"),
                py::arg("max_iterations"),
                py::arg("g_weight"),
                py::arg("h_weight"),
                py::arg("threshold1"),
                py::arg("scale_factor1"),
                py::arg("threshold2"),
                py::arg("scale_factor2"),
                py::arg("with_constraint_table") = false);

        // ------ BiDirectionalAstarSearch<QueueSpace::BoostHeap>
        py::class_<SearchSpace::BiDirectionalAstarSearch<QueueSpace::BoostHeap>,
                  SearchSpace::WeightedAstarSearch<QueueSpace::BoostHeap>>(m, "BiDirectionalAstarSearchBoost")
            .def(py::init<double, size_t, double, double>(),
                py::arg("radius"),
                py::arg("max_iterations"),
                py::arg("g_weight"),
                py::arg("h_weight"))
            .def("reset", &SearchSpace::BiDirectionalAstarSearch<QueueSpace::BoostHeap>::reset)
            .def("search", &SearchSpace::BiDirectionalAstarSearch<QueueSpace::BoostHeap>::search,
                py::arg("grid"),
                py::arg("starts"),
                py::arg("goals"),
                py::arg("path"));

        // ------ BiDirectionalAstarSearch<QueueSpace::BucketPriorityQueue>
        py::class_<SearchSpace::BiDirectionalAstarSearch<QueueSpace::BucketPriorityQueue>,
                  SearchSpace::WeightedAstarSearch<QueueSpace::BucketPriorityQueue>>(m, "BiDirectionalAstarSearchBucket")
            .def(py::init<double, size_t, double, double>(),
                py::arg("radius"),
                py::arg("max_iterations"),
                py::arg("g_weight"),
                py::arg("h_weight"))
            .def("reset", &SearchSpace::BiDirectionalAstarSearch<QueueSpace::BucketPriorityQueue>::reset)
            .def("search", &SearchSpace::BiDirectionalAstarSearch<QueueSpace::BucketPriorityQueue>::search,
                py::arg("grid"),
                py::arg("starts"),
                py::arg("goals"),
                py::arg("path"));
        
        // ------ TaskData (base class)
        py::class_<TaskSpace::TaskData>(m, "TaskData")
            .def_readonly("name", &TaskSpace::TaskData::name)
            .def_readonly("priority", &TaskSpace::TaskData::priority);

        // ------ TaskPairNodeData
        py::class_<TaskSpace::TaskPairNodeData, TaskSpace::TaskData>(m, "TaskPairNodeData")
            .def_readonly("start_name", &TaskSpace::TaskPairNodeData::start_name)
            .def_readonly("end_name", &TaskSpace::TaskPairNodeData::end_name)
            .def_readonly("start_locs", &TaskSpace::TaskPairNodeData::start_locs)
            .def_readonly("end_locs", &TaskSpace::TaskPairNodeData::end_locs)
            .def_readonly("radius", &TaskSpace::TaskPairNodeData::radius);

        // ------ TaskSet (base class)
        py::class_<TaskSpace::TaskSet>(m, "TaskSet")
            .def_readonly("name", &TaskSpace::TaskSet::name)
            .def("get_task_size", &TaskSpace::TaskSet::get_task_size);

        // ------ PairNodeTaskSet
        py::class_<TaskSpace::PairNodeTaskSet, TaskSpace::TaskSet>(m, "PairNodeTaskSet")
            .def(py::init<const std::string>(), py::arg("name"))
            .def("add_task", py::overload_cast<const std::string,
                                              const std::string,
                                              const std::string,
                                              const std::vector<LocXyz>,
                                              const std::vector<LocXyz>,
                                              const double>(&TaskSpace::PairNodeTaskSet::add_task),
                py::arg("name"),
                py::arg("start_name"),
                py::arg("end_name"),
                py::arg("start_locs"),
                py::arg("end_locs"),
                py::arg("radius"))
            .def("add_landmark_to_task", &TaskSpace::PairNodeTaskSet::add_landmark_to_task,
                py::arg("priority"),
                py::arg("landmark_name"),
                py::arg("landmark_loc"))
            .def("get_task", &TaskSpace::PairNodeTaskSet::get_task,
                py::arg("priority"))
            .def_readonly("m_max_radius", &TaskSpace::PairNodeTaskSet::m_max_radius);

        // ------ SetPlanner (base class)
        py::class_<SearchSpace::SetPlanner>(m, "SetPlanner")
            .def("set_tasks", py::overload_cast<TaskSpace::TaskSet&>(&SearchSpace::SetPlanner::set_tasks),
                py::arg("task_set"));

        // ------ SetPairNodePlanner
        py::class_<SearchSpace::SetPairNodePlanner, SearchSpace::SetPlanner>(m, "SetPairNodePlanner")
            .def(py::init<SearchSpace::BaseSearch&>(), py::arg("planner"))
            .def("set_tasks", py::overload_cast<TaskSpace::PairNodeTaskSet&>(&SearchSpace::SetPairNodePlanner::set_tasks),
                py::arg("task_set"))
            .def("preprocess", &SearchSpace::SetPairNodePlanner::preprocess)
            .def("search", &SearchSpace::SetPairNodePlanner::search,
                py::arg("grid"),
                py::arg("result_sets"));
        
        // ------ ConflictBaseSearchNode (base class)
        py::class_<SearchSpace::ConflictBaseSearchNode>(m, "ConflictBaseSearchNode")
            .def_readonly("g_cost", &SearchSpace::ConflictBaseSearchNode::g_cost)
            .def_readonly("f_cost", &SearchSpace::ConflictBaseSearchNode::f_cost)
            .def_readonly("is_free_conflict", &SearchSpace::ConflictBaseSearchNode::is_free_conflict)
            .def_readonly("m_conflict_set", &SearchSpace::ConflictBaseSearchNode::m_conflict_set)
            .def_readonly("m_constraint", &SearchSpace::ConflictBaseSearchNode::m_constraint)
            .def("add_dynamic_obstacle", &SearchSpace::ConflictBaseSearchNode::add_dynamic_obstacle,
                py::arg("run_group"),
                py::arg("new_obstacle"))
            .def("get_group_names", &SearchSpace::ConflictBaseSearchNode::get_group_names,
                py::arg("names"))
            .def_static("set_weights", &SearchSpace::ConflictBaseSearchNode::set_weights,
                py::arg("g_weight"),
                py::arg("h_weight"));

        // ------ CbsNode<SearchSpace::NearEndConstraintSelector>
        py::class_<SearchSpace::CbsNode<SearchSpace::NearEndConstraintSelector>,
                  SearchSpace::ConflictBaseSearchNode>(m, "CbsNodeNearEnd")
            .def("find_division_constraint", 
                &SearchSpace::CbsNode<SearchSpace::NearEndConstraintSelector>::find_division_constraint,
                py::arg("conflict_set"),
                py::arg("selected_constraint"));

        // ------ ConflictBaseSearch (base class)
        py::class_<SearchSpace::ConflictBaseSearch>(m, "ConflictBaseSearch")
            .def("add_task_set", &SearchSpace::ConflictBaseSearch::add_task_set,
                py::arg("task_set"),
                py::arg("planner"))
            .def("init_root_node", &SearchSpace::ConflictBaseSearch::init_root_node)
            .def("search", &SearchSpace::ConflictBaseSearch::search,
                py::arg("result"));

        // ------ CbsSearch<SearchSpace::NearEndConstraintSelector>
        py::class_<SearchSpace::CbsSearch<SearchSpace::NearEndConstraintSelector>,
                  SearchSpace::ConflictBaseSearch>(m, "CbsSearchNearEnd")
            .def(py::init<GridSpace::SearchGrid*, double, double, int16_t>(),
                py::arg("grid"),
                py::arg("g_weight") = 1.0,
                py::arg("h_weight") = 1.0,
                py::arg("max_iteration") = 10000);

        // ------ PyWrapAstarRunner
        py::class_<PyWrapAstarRunner>(m, "PyWrapAstarRunner")
            .def(py::init<std::string, GridSpace::SearchGrid*, SearchSpace::BaseSearch*>())
            .def_readonly("result", &PyWrapAstarRunner::result)
            .def_readonly("name", &PyWrapAstarRunner::name)
            .def("add_dynamic_obstacle", &PyWrapAstarRunner::add_dynamic_obstacle,
                py::arg("obstacle_data"))
            .def("run", &PyWrapAstarRunner::run,
                py::arg("starts"),
                py::arg("goals"));
        
        // ------ PyWrapGroupAstarRunner
        py::class_<PyWrapGroupAstarRunner>(m, "PyWrapGroupAstarRunner")
            .def(py::init<std::string, GridSpace::SearchGrid*, SearchSpace::BaseSearch*>())
            .def_readonly("result", &PyWrapGroupAstarRunner::result)
            .def_readonly("name", &PyWrapGroupAstarRunner::name)
            .def("add_dynamic_obstacle", &PyWrapGroupAstarRunner::add_dynamic_obstacle,
                py::arg("obstacle_data"))
            .def("add_task", &PyWrapGroupAstarRunner::add_task,
                py::arg("name"),
                py::arg("start_name"),
                py::arg("end_name"),
                py::arg("start_locs"),
                py::arg("end_locs"),
                py::arg("radius"))
            .def("run", &PyWrapGroupAstarRunner::run);
        
        // ------ PyWrapCbsRunner
        py::class_<PyWrapCbsRunner>(m, "PyWrapCbsRunner")
            .def(py::init<std::string, GridSpace::SearchGrid*>())
            .def_readonly("result", &PyWrapCbsRunner::result)
            .def_readonly("name", &PyWrapCbsRunner::name)
            .def("add_set", &PyWrapCbsRunner::add_set,
                py::arg("name"),
                py::arg("planner"))
            .def("add_dynamic_obstacle", &PyWrapCbsRunner::add_dynamic_obstacle,
                py::arg("task_name"),
                py::arg("obstacle_data"))
            .def("add_task", &PyWrapCbsRunner::add_task,
                py::arg("task_set_name"),
                py::arg("task_name"),
                py::arg("start_name"),
                py::arg("end_name"),
                py::arg("start_locs"),
                py::arg("end_locs"),
                py::arg("radius"))
            .def("set_cbs_planner", &PyWrapCbsRunner::set_cbs_planner,
                py::arg("cbs_planner"))
            .def("init_root_node", &PyWrapCbsRunner::init_root_node)
            .def("run", &PyWrapCbsRunner::run)
            .def("single_step_search", &PyWrapCbsRunner::single_step_search);

    };
}