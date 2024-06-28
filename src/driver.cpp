//
// Created by admin123456 on 2024/5/30.
//

#include "common.h"
#include "math_utils.h"
#include "pcl_utils.h"
#include "grid_utils.h"
#include "state_utils.h"
#include "collision_utils.h"
#include "constraint_avoid_table.h"
#include "astar_algo.h"
#include "group_astar_algo.h"
#include "conflict_utils.h"
#include "cbs_algo.h"

#include "pybind_utils.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std;

//PYBIND11_MAKE_OPAQUE(std::vector<double>);

PYBIND11_MODULE(mapf_pipeline, m) {
    m.doc() = "mapf pipeline"; // optional module docstring

    // ------ Function Define
    // py::bind_vector<std::vector<double>>(m, "VectorDouble");
    //m.def("test", &test);

    // ------ Class Define
    py::class_<PybindUtils::Matrix2D>(m, "Matrix2D", py::buffer_protocol())
            .def_buffer([](PybindUtils::Matrix2D &m) -> py::buffer_info {
                return py::buffer_info(
                        m.data(),
                        sizeof(float),
                        py::format_descriptor<double>::format(),
                        2,
                        {m.rows(), m.cols()},
                        {sizeof(float) * m.cols(), sizeof(float)}
                );
            });

    py::class_<pcl::PointXYZ>(m, "PointXYZ_pcl")
            .def_readonly("x", &pcl::PointXYZ::x)
            .def_readonly("y", &pcl::PointXYZ::y)
            .def_readonly("z", &pcl::PointXYZ::z);

    py::class_<PclUtils::KDTree>(m, "KDtree_pcl")
            .def(py::init<>())
            .def_readonly("result_idxs_1D", &PclUtils::KDTree::result_idxs_1D)
            .def_readonly("result_distance_1D", &PclUtils::KDTree::result_distance_1D)
            .def("create_tree", &PclUtils::KDTree::create_tree)
            .def("nearestKSearch", &PclUtils::KDTree::nearestKSearch, "x"_a, "y"_a, "z"_a, "k"_a)
            .def("radiusSearch", &PclUtils::KDTree::nearestKSearch, "x"_a, "y"_a, "z"_a, "radius"_a)
            .def("update_data", &PclUtils::KDTree::update_data, "data"_a)
            .def("insert_data", py::overload_cast<double, double, double>(&PclUtils::KDTree::insert_data),
                 "x"_a, "y"_a, "z"_a)
            .def("get_point_from_data", &PclUtils::KDTree::get_point_from_data, "idx"_a)
            .def("get_pcd_size", &PclUtils::KDTree::get_pcd_size)
            .def("clear_data", &PclUtils::KDTree::clear_data);

    m.attr("candidate_1D") = &Grid::candidate_1D;
    m.attr("candidate_2D") = &Grid::candidate_2D;
    m.attr("candidate_3D") = &Grid::candidate_3D;

    py::class_<Grid::DiscreteGridEnv>(m, "DiscreteGridEnv")
            .def(py::init<int, int, int, double, double, double, double, double, double>(),
                 "size_of_x"_a, "size_of_y"_a, "size_of_z"_a,
                 "x_init"_a, "y_init"_a, "z_init"_a,
                 "x_grid_length"_a, "y_grid_length"_a, "z_grid_length"_a)
            .def("xyz2grid", &Grid::DiscreteGridEnv::xyz2grid, "x"_a, "y"_a, "z"_a)
            .def("xyz2flag", &Grid::DiscreteGridEnv::xyz2flag, "x"_a, "y"_a, "z"_a)
            .def("grid2flag", &Grid::DiscreteGridEnv::grid2flag, "x_grid"_a, "y_grid"_a, "z_grid"_a)
            .def("flag2grid", &Grid::DiscreteGridEnv::flag2grid, "loc_flag"_a)
            .def("flag2xyz", &Grid::DiscreteGridEnv::flag2xyz, "loc_flag"_a)
            .def("grid2xyz", &Grid::DiscreteGridEnv::grid2xyz, "x_grid"_a, "y_grid"_a, "z_grid"_a)
            .def("get_manhattan_cost", &Grid::DiscreteGridEnv::get_manhattan_cost, "loc_flag0"_a, "loc_flag1"_a)
            .def("get_euler_cost", &Grid::DiscreteGridEnv::get_euler_cost, "loc_flag0"_a, "loc_flag1"_a)
            .def("get_valid_neighbors", &Grid::DiscreteGridEnv::get_valid_neighbors, "loc_flag"_a, "step_scale"_a,
                 "candidates"_a);

    py::class_<Grid::DynamicStepStateDetector>(m, "DynamicStepStateDetector")
            .def(py::init<Grid::DiscreteGridEnv &>(), "grid"_a)
            .def("insert_start_flags", &Grid::DynamicStepStateDetector::insert_start_flags,
                 "loc_flag"_a, "direction_x"_a, "direction_x"_a, "direction_x"_a, "force"_a)
            .def("insert_target_flags", &Grid::DynamicStepStateDetector::insert_target_flags,
                 "loc_flag"_a, "direction_x"_a, "direction_x"_a, "direction_x"_a, "force"_a)
            .def("get_start_info", &Grid::DynamicStepStateDetector::get_start_info, "loc_flag"_a)
            .def("get_target_info", &Grid::DynamicStepStateDetector::get_target_info, "loc_flag"_a)
            .def("get_start_pos_flags", &Grid::DynamicStepStateDetector::get_start_pos_flags)
            .def("get_target_pos_flags", &Grid::DynamicStepStateDetector::get_target_pos_flags)
            .def("is_target", &Grid::DynamicStepStateDetector::is_target, "loc_flag"_a)
            .def("adjust_scale", &Grid::DynamicStepStateDetector::adjust_scale, "loc_flag"_a, "current_scale"_a)
            .def("update_dynamic_info", &Grid::DynamicStepStateDetector::update_dynamic_info, "shrink_distance"_a,
                 "scale"_a)
            .def("clear", &Grid::DynamicStepStateDetector::clear);

    py::class_<CollisionDetector>(m, "CollisionDetector")
            .def(py::init<>())
            .def("update_data", &CollisionDetector::update_data, "data")
            .def("create_tree", &CollisionDetector::create_tree)
            .def("is_valid", py::overload_cast<double, double, double, double>(&CollisionDetector::is_valid))
            .def("is_valid", py::overload_cast<double, double, double, double, double, double, double>(
                    &CollisionDetector::is_valid));

    py::class_<ConflictAvoidTable>(m, "ConflictAvoidTable")
            .def(py::init<>())
            .def("insert", &ConflictAvoidTable::insert, "flag"_a)
            .def("get_num_of_conflict", &ConflictAvoidTable::get_num_of_conflict, "flag"_a)
            .def("get_data", &ConflictAvoidTable::get_data);

    py::class_<PathResult>(m, "PathResult")
            .def(py::init<Grid::DiscreteGridEnv *>())
            .def("get_path_flags", &PathResult::get_path_flags)
            .def("get_radius", &PathResult::get_radius)
            .def("get_length", &PathResult::get_length)
            .def("get_step_length", &PathResult::get_step_length)
            .def("get_path", &PathResult::get_path);

    py::class_<StandardAStarSolver>(m, "StandardAStarSolver")
            .def(py::init<Grid::DiscreteGridEnv &, CollisionDetector &, CollisionDetector &, Grid::DynamicStepStateDetector &>())
            .def_readonly("num_expanded", &StandardAStarSolver::num_expanded)
            .def_readonly("num_generated", &StandardAStarSolver::num_generated)
            .def_readonly("search_time_cost", &StandardAStarSolver::search_time_cost)
            .def("update_configuration", &StandardAStarSolver::update_configuration,
                 "pipe_radius"_a, "search_step_scale"_a, "grid_expand_candidates"_a,
                 "use_curvature_cost"_a, "curvature_cost_weight"_a,
                 "use_avoid_table"_a = true, "use_theta_star"_a = false)
            .def("find_path", &StandardAStarSolver::find_path, "res_path"_a, "max_iter"_a, "avoid_table"_a);

    py::class_<TaskInfo>(m, "TaskInfo")
            .def(py::init<
                         string, string, string, vector<Cell_Flag_Orient>, vector<Cell_Flag_Orient>,
                         double, int, double, int, vector<tuple<int, int, int>>, bool, double, bool, bool>(),
                 "task_name"_a, "begin_tag"_a, "final_tag"_a, "begin_marks"_a, "final_marks"_a,
                 "search_radius"_a, "step_scale"_a, "shrink_distance"_a, "shrink_scale"_a, "expand_grid_cell"_a,
                 "with_curvature_cost"_a, "curvature_cost_weight"_a,
                 "use_constraint_avoid_table"_a, "with_theta_star"_a)
            .def_readonly("task_name", &TaskInfo::task_name)
            .def_readonly("begin_tag", &TaskInfo::begin_tag)
            .def_readonly("final_tag", &TaskInfo::final_tag)
            .def_readonly("begin_marks", &TaskInfo::begin_marks)
            .def_readonly("final_marks", &TaskInfo::final_marks)
            .def_readwrite("search_radius", &TaskInfo::search_radius)
            .def_readwrite("step_scale", &TaskInfo::step_scale)
            .def_readwrite("shrink_distance", &TaskInfo::shrink_distance)
            .def_readwrite("shrink_scale", &TaskInfo::shrink_scale)
            .def_readwrite("expand_grid_cell", &TaskInfo::expand_grid_cell)
            .def_readwrite("with_theta_star", &TaskInfo::with_theta_star)
            .def_readwrite("use_constraint_avoid_table", &TaskInfo::use_constraint_avoid_table)
            .def_readwrite("with_curvature_cost", &TaskInfo::with_curvature_cost)
            .def_readwrite("curvature_cost_weight", &TaskInfo::curvature_cost_weight);

    py::class_<GroupAstar>(m, "GroupAstar")
            .def(py::init<Grid::DiscreteGridEnv *, CollisionDetector *>())
            .def("update_task_tree", &GroupAstar::update_task_tree, "task_list"_a)
            .def("find_path", &GroupAstar::find_path, "dynamic_obstacles"_a, "max_iter"_a, "avoid_table"_a)
            .def("get_task_tree", &GroupAstar::get_task_tree)
            .def("get_res", &GroupAstar::get_res)
            .def("reset", &GroupAstar::reset);

    py::class_<ConflictCell>(m, "ConflictCell")
            .def(py::init<size_t, double, double, double, double, size_t, double, double, double, double>(),
                 "idx0"_a, "x0"_a, "y0"_a, "z0"_a, "radius0"_a,
                 "idx1"_a, "x1"_a, "y1"_a, "z1"_a, "radius1"_a)
            .def_readonly("idx0", &ConflictCell::idx0)
            .def_readonly("x0", &ConflictCell::x0)
            .def_readonly("y0", &ConflictCell::y0)
            .def_readonly("z0", &ConflictCell::z0)
            .def_readonly("radius0", &ConflictCell::radius0)
            .def_readonly("idx1", &ConflictCell::idx1)
            .def_readonly("x1", &ConflictCell::x1)
            .def_readonly("y1", &ConflictCell::y1)
            .def_readonly("z1", &ConflictCell::z1)
            .def_readonly("radius1", &ConflictCell::radius1);

    py::class_<CbsNode>(m, "CbsNode")
            .def(py::init<size_t>(), "node_id"_a)
            .def_readonly("node_id", &CbsNode::node_id)
            .def_readonly("num_expanded", &CbsNode::num_expanded)
            .def_readonly("num_generated", &CbsNode::num_generated)
            .def_readonly("search_time_cost", &CbsNode::search_time_cost)
            .def("compute_g_val", &CbsNode::compute_g_val)
            .def("compute_h_val", &CbsNode::compute_h_val)
            .def("get_f_val", &CbsNode::get_f_val)
            .def("update_constrains_map", &CbsNode::update_constrains_map, "group_idx"_a, "group_dynamic_obstacles"_a)
            .def("update_group_path", &CbsNode::update_group_path, "group_idx"_a, "max_iter"_a)
            .def("update_group_cell", &CbsNode::update_group_cell,
                 "group_idx"_a, "group_task_tree"_a, "group_grid"_a, "obstacle_detector"_a, "group_dynamic_obstacles"_a)
            .def("is_conflict_free", &CbsNode::is_conflict_free)
            .def("find_inner_conflict_point2point", &CbsNode::find_inner_conflict_point2point)
            .def("find_inner_conflict_segment2segment", &CbsNode::find_inner_conflict_segment2segment)
            .def("copy_from_node", &CbsNode::copy_from_node, "rhs"_a)
            .def("get_group_path", &CbsNode::get_group_path, "group_idx"_a, "name"_a)
            .def("get_conflict_length", &CbsNode::get_conflict_length, "group_idx"_a)
            .def("get_conflict_cells", &CbsNode::get_conflict_cells)
            .def("get_constrain", &CbsNode::get_constrain, "group_idx"_a)
            .def("get_conflict_size", &CbsNode::get_conflict_size);

    py::class_<CbsSolver>(m, "CbsSolver")
            .def(py::init<>())
            .def("push_node", &CbsSolver::push_node, "node"_a)
            .def("pop_node", &CbsSolver::pop_node)
            .def("is_openList_empty", &CbsSolver::is_openList_empty);
}