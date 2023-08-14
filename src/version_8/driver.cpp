#include <pybind11/pybind11.h>
#include "iostream"

#include "common.h"
#include "utils.h"
#include "vector3d.h"
#include "instance.h"
#include "conflict.h"
#include "constrainTable.h"
#include "AstarSolver.h"

#include "vertex_XYZ.h"
#include "g2oSmoother_xyz.h"

#include "cbs_node.h"
#include "cbs_solver.h"
#include "groupAstarSolver.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(mapf_pipeline, m) {
    m.doc() = "mapf pipeline"; // optional module docstring

    m.def("point2LineSegmentDistance", &point2LineSegmentDistance);

    py::class_<Instance>(m, "Instance")
            .def(py::init<int, int, int>())
            .def_readonly("num_of_x", &Instance::num_of_x)
            .def_readonly("num_of_y", &Instance::num_of_y)
            .def_readonly("num_of_z", &Instance::num_of_z)
            .def_readonly("map_size", &Instance::map_size)
            .def("info", &Instance::info)
            .def("getXCoordinate", &Instance::getXCoordinate, "curr"_a)
            .def("getYCoordinate", &Instance::getYCoordinate, "curr"_a)
            .def("getZCoordinate", &Instance::getZCoordinate, "curr"_a)
            .def("getCoordinate", &Instance::getCoordinate, "curr"_a)
            .def("getNeighbors", &Instance::getNeighbors, "curr"_a)
            .def("linearizeCoordinate", py::overload_cast<int, int, int>(&Instance::linearizeCoordinate, py::const_),
                 "x"_a, "y"_a, "z"_a)
            .def("linearizeCoordinate",
                 py::overload_cast<const std::tuple<int, int, int> &>(&Instance::linearizeCoordinate, py::const_),
                 "curr"_a)
            .def("isValidPos", &Instance::isValidPos, "x"_a, "y"_a, "z"_a);

    py::class_<ConstraintTable>(m, "ConstraintTable")
            .def(py::init<>())
            .def("insert2CT", py::overload_cast<double, double, double, double>(&ConstraintTable::insert2CT),
                 "x"_a, "y"_a, "z"_a, "radius"_a)
            .def("insert2CT", py::overload_cast<ConstrainType>(&ConstraintTable::insert2CT), "constrain"_a)
            .def("isConstrained", py::overload_cast<double, double, double, double>(&ConstraintTable::isConstrained),
                 "x"_a, "y"_a, "z"_a, "radius"_a)
            .def("isConstrained", py::overload_cast<double, double, double, double, double, double, double>(
                         &ConstraintTable::isConstrained),
                 "lineStart_x"_a, "lineStart_y"_a, "lineStart_z"_a, "lineEnd_x"_a, "lineEnd_y"_a, "lineEnd_z"_a,
                 "radius"_a
            )
            .def("islineOnSight", &ConstraintTable::islineOnSight,
                 "lineStart_x"_a, "lineStart_y"_a, "lineStart_z"_a, "lineEnd_x"_a, "lineEnd_y"_a, "lineEnd_z"_a,
                 "radius"_a
            );

    m.def("sampleDetailPath", &PlannerNameSpace::sampleDetailPath, "path_xyzr"_a, "stepLength"_a);

    py::class_<PlannerNameSpace::AStarSolver>(m, "AStarSolver")
            .def(py::init<bool, bool>(), "with_AnyAngle"_a, "with_OrientCost"_a)
            .def_readonly("num_expanded", &PlannerNameSpace::AStarSolver::num_expanded)
            .def_readonly("num_generated", &PlannerNameSpace::AStarSolver::num_generated)
            .def_readonly("runtime_search", &PlannerNameSpace::AStarSolver::runtime_search)
            .def_readonly("runtime_build_CT", &PlannerNameSpace::AStarSolver::runtime_build_CT)
                    // .def("findPath", &PlannerNameSpace::AStarSolver::findPath,
                    //     "radius"_a, "constraints"_a, "instance"_a, "start_loc"_a, "goal_locs"_a, "check_EndPosValid"_a
                    // )
            .def("findPath", &PlannerNameSpace::AStarSolver::findPath,
                 "radius"_a, "constraint_table"_a, "obstacle_table"_a, "instance"_a, "start_locs"_a, "goal_locs"_a
            );

    py::class_<Conflict>(m, "Conflict")
            .def(py::init<>())
            .def(py::init<
                    size_ut, double, double, double, double,
                    size_ut, double, double, double, double>())
            .def_readonly("groupIdx1", &Conflict::groupIdx1)
            .def_readonly("conflict1_x", &Conflict::conflict1_x)
            .def_readonly("conflict1_y", &Conflict::conflict1_y)
            .def_readonly("conflict1_z", &Conflict::conflict1_z)
            .def_readonly("conflict1_radius", &Conflict::conflict1_radius)
            .def_readonly("groupIdx2", &Conflict::groupIdx2)
            .def_readonly("conflict2_x", &Conflict::conflict2_x)
            .def_readonly("conflict2_y", &Conflict::conflict2_y)
            .def_readonly("conflict2_z", &Conflict::conflict2_z)
            .def_readonly("conflict2_radius", &Conflict::conflict2_radius)
            .def_readonly("constrain1", &Conflict::constrain1)
            .def_readonly("constrain2", &Conflict::constrain2)
            .def("conflictExtend", &Conflict::conflictExtend)
            .def("info", &Conflict::info);

    py::class_<PlannerNameSpace::TaskInfo>(m, "TaskInfo")
            .def(py::init<size_t, double, size_t, double>())
            .def_readonly("loc0", &PlannerNameSpace::TaskInfo::loc0)
            .def_readonly("loc1", &PlannerNameSpace::TaskInfo::loc1)
            .def_readonly("radius0", &PlannerNameSpace::TaskInfo::radius0)
            .def_readonly("radius1", &PlannerNameSpace::TaskInfo::radius1)
            .def_readonly("res_path", &PlannerNameSpace::TaskInfo::res_path);

    py::class_<PlannerNameSpace::GroupAstarSolver>(m, "GroupAstarSolver")
            .def(py::init<>())
            .def_readonly("taskTree", &PlannerNameSpace::GroupAstarSolver::taskTree)
            .def("addTask", &PlannerNameSpace::GroupAstarSolver::addTask, "loc0"_a, "radius0"_a, "loc1"_a, "radius1"_a)
            .def("findPath", &PlannerNameSpace::GroupAstarSolver::findPath, "solver"_a, "constraints"_a,
                 "obstacle_table"_a, "instance"_a, "stepLength"_a);

    py::class_<CBSNameSpace::CBSNode>(m, "CBSNode")
            .def(py::init<double>(), "stepLength"_a)
            .def_readwrite("node_id", &CBSNameSpace::CBSNode::node_id)
            .def_readwrite("depth", &CBSNameSpace::CBSNode::depth)
            .def_readonly("g_val", &CBSNameSpace::CBSNode::g_val)
            .def_readonly("h_val", &CBSNameSpace::CBSNode::h_val)
            .def_readonly("firstConflict", &CBSNameSpace::CBSNode::firstConflict)
            .def_readonly("isConflict", &CBSNameSpace::CBSNode::isConflict)
            .def_readonly("rectangleExcludeAreas", &CBSNameSpace::CBSNode::rectangleExcludeAreas)
            .def("update_Constrains", &CBSNameSpace::CBSNode::update_Constrains, "groupIdx"_a, "new_constrains"_a)
            .def("findFirstPipeConflict", &CBSNameSpace::CBSNode::findFirstPipeConflict)
            .def("compute_Heuristics", &CBSNameSpace::CBSNode::compute_Heuristics)
            .def("compute_Gval", &CBSNameSpace::CBSNode::compute_Gval)
            .def("copy", &CBSNameSpace::CBSNode::copy, "rhs"_a)
            .def("add_GroupAgent", &CBSNameSpace::CBSNode::add_GroupAgent, "groupIdx"_a)
            .def("getConstrains", &CBSNameSpace::CBSNode::getConstrains, "groupIdx"_a)
            .def("addTask_to_GroupAgent", &CBSNameSpace::CBSNode::addTask_to_GroupAgent, "groupIdx"_a, "loc0"_a,
                 "radius0"_a, "loc1"_a, "radius1"_a)
            .def("getGroupAgentResPath", &CBSNameSpace::CBSNode::getGroupAgentResPath, "groupIdx"_a)
            .def("info", &CBSNameSpace::CBSNode::info, "with_constrainInfo"_a = false, "with_pathInfo"_a = false)
            .def("add_rectangleExcludeArea", &CBSNameSpace::CBSNode::add_rectangleExcludeArea,
                 "xmin"_a, "ymin"_a, "zmin"_a, "xmax"_a, "ymax"_a, "zmax"_a)
            .def("clear_rectangleExcludeArea", &CBSNameSpace::CBSNode::clear_rectangleExcludeArea)
            .def("isIn_rectangleExcludeAreas", &CBSNameSpace::CBSNode::isIn_rectangleExcludeAreas, "x"_a, "y"_a, "z"_a);

    py::class_<CBSNameSpace::CBSSolver>(m, "CBSSolver")
            .def(py::init<>())
            .def("pushNode", &CBSNameSpace::CBSSolver::pushNode, "node"_a)
            .def("popNode", &CBSNameSpace::CBSSolver::popNode)
            .def("is_openList_empty", &CBSNameSpace::CBSSolver::is_openList_empty)
            .def("isGoal", &CBSNameSpace::CBSSolver::isGoal)
            .def("addSearchEngine", &CBSNameSpace::CBSSolver::addSearchEngine, "groupIdx"_a, "with_AnyAngle"_a,
                 "with_OrientCost"_a)
            .def("update_GroupAgentPath", &CBSNameSpace::CBSSolver::update_GroupAgentPath,
                 "groupIdx"_a, "node"_a, "instance"_a)
            .def("add_obstacle", &CBSNameSpace::CBSSolver::add_obstacle, "x"_a, "y"_a, "z"_a, "radius"_a);

    py::class_<SmootherNameSpace::NxGraphNode>(m, "NxGraphNode")
            .def(py::init<size_t, double, double, double, double, bool>())
            .def_readonly("nodeIdx", &SmootherNameSpace::NxGraphNode::nodeIdx)
            .def_readonly("x", &SmootherNameSpace::NxGraphNode::x)
            .def_readonly("y", &SmootherNameSpace::NxGraphNode::y)
            .def_readonly("z", &SmootherNameSpace::NxGraphNode::z)
            .def_readonly("radius", &SmootherNameSpace::NxGraphNode::radius)
            .def_readonly("fixed", &SmootherNameSpace::NxGraphNode::fixed);

    py::class_<SmootherNameSpace::FlexSmootherXYZ_Runner>(m, "FlexSmootherXYZ_Runner")
            .def(py::init<>())
            .def_readonly("graphNode_map", &SmootherNameSpace::FlexSmootherXYZ_Runner::graphNode_map)
            .def("initOptimizer", &SmootherNameSpace::FlexSmootherXYZ_Runner::initOptimizer)
            .def("add_obstacle", &SmootherNameSpace::FlexSmootherXYZ_Runner::add_obstacle, "x"_a, "y"_a, "z"_a, "radius"_a)
            .def("clear_graph", &SmootherNameSpace::FlexSmootherXYZ_Runner::clear_graph)
            .def("add_graphNode", &SmootherNameSpace::FlexSmootherXYZ_Runner::add_graphNode,
                 "nodeIdx"_a, "x"_a, "y"_a, "z"_a, "radius"_a, "fixed"_a)
            .def("clear_graphNodeMap", &SmootherNameSpace::FlexSmootherXYZ_Runner::clear_graphNodeMap)
            .def("add_vertex", &SmootherNameSpace::FlexSmootherXYZ_Runner::add_vertex, "nodeIdx"_a)
            .def("is_g2o_graph_empty", &SmootherNameSpace::FlexSmootherXYZ_Runner::is_g2o_graph_empty)
            .def("is_g2o_graph_edges_empty", &SmootherNameSpace::FlexSmootherXYZ_Runner::is_g2o_graph_edges_empty)
            .def("is_g2o_graph_vertices_empty", &SmootherNameSpace::FlexSmootherXYZ_Runner::is_g2o_graph_vertices_empty)
            .def("add_elasticBand", &SmootherNameSpace::FlexSmootherXYZ_Runner::add_elasticBand,
                 "nodeIdx0"_a, "nodeIdx1"_a, "kSpring"_a, "weight"_a)
            .def("add_kinematicEdge", &SmootherNameSpace::FlexSmootherXYZ_Runner::add_kinematicEdge,
                 "nodeIdx0"_a, "nodeIdx1"_a, "nodeIdx2"_a, "kSpring"_a, "weight"_a)
            .def("add_kinematicVertexEdge", &SmootherNameSpace::FlexSmootherXYZ_Runner::add_kinematicVertexEdge,
                 "nodeIdx0"_a, "nodeIdx1"_a, "vec_i"_a, "vec_j"_a, "vec_k"_a, "kSpring"_a, "weight"_a)
            .def("add_obstacleEdge", &SmootherNameSpace::FlexSmootherXYZ_Runner::add_obstacleEdge,
                 "nodeIdx"_a, "searchScale"_a, "repleScale"_a, "kSpring"_a, "weight"_a)
            .def("add_pipeConflictEdge", &SmootherNameSpace::FlexSmootherXYZ_Runner::add_pipeConflictEdge,
                 "nodeIdx"_a, "groupIdx"_a, "searchScale"_a, "repleScale"_a, "kSpring"_a, "weight"_a)
            .def("updateNodeMap_Vertex", &SmootherNameSpace::FlexSmootherXYZ_Runner::updateNodeMap_Vertex)
            .def("updateGroupTrees", &SmootherNameSpace::FlexSmootherXYZ_Runner::updateGroupTrees,
                 "groupIdx"_a, "nodeIdxs"_a)
            .def("optimizeGraph", &SmootherNameSpace::FlexSmootherXYZ_Runner::optimizeGraph,
                 "no_iterations"_a, "verbose"_a)
            .def("info", &SmootherNameSpace::FlexSmootherXYZ_Runner::info);

    // it don't work, I don't know why
    // m.def("printPointer", &printPointer<KDTreeData>, "a"_a, "tag"_a);

}
