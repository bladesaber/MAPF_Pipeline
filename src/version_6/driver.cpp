#include <pybind11/pybind11.h>
#include "iostream"

#include "common.h"
#include "utils.h"
#include "vector3d.h"
#include "instance.h"
#include "conflict.h"
#include "constrainTable.h"
#include "AstarSolver.h"

#include "groupPath.h"
#include "vertex_XYZ.h"
#include "smootherXYZ_g2o.h"

#include "cbs_node.h"
#include "cbs_solver.h"
#include "spanningTree_groupSolver.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(mapf_pipeline, m) {
    m.doc() = "mapf pipeline"; // optional module docstring

    // m.def("debug", &test);

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
        .def("linearizeCoordinate", py::overload_cast<int, int, int>(&Instance::linearizeCoordinate, py::const_), "x"_a, "y"_a, "z"_a)
        .def("linearizeCoordinate", py::overload_cast<const std::tuple<int, int, int>&>(&Instance::linearizeCoordinate, py::const_), "curr"_a);

    py::class_<ConstraintTable>(m, "ConstraintTable")
        .def(py::init<>())
        .def("insert2CT", py::overload_cast<double, double, double, double>(&ConstraintTable::insert2CT), "x"_a, "y"_a, "z"_a, "radius"_a)
        .def("insert2CT", py::overload_cast<ConstrainType>(&ConstraintTable::insert2CT), "constrain"_a)
        .def("isConstrained", py::overload_cast<double, double, double, double>(&ConstraintTable::isConstrained), "x"_a, "y"_a, "z"_a, "radius"_a)
        .def("isConstrained", py::overload_cast<double, double, double, double, double, double, double>(&ConstraintTable::isConstrained), 
            "lineStart_x"_a, "lineStart_y"_a, "lineStart_z"_a, "lineEnd_x"_a, "lineEnd_y"_a, "lineEnd_z"_a, "radius"_a
        )
        .def("islineOnSight", &ConstraintTable::islineOnSight,
            "lineStart_x"_a, "lineStart_y"_a, "lineStart_z"_a, "lineEnd_x"_a, "lineEnd_y"_a, "lineEnd_z"_a, "radius"_a
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
        .def("findPath", 
            py::overload_cast<double, std::vector<ConstrainType>, Instance&, std::vector<size_t>&, std::vector<size_t>&>(
                &PlannerNameSpace::AStarSolver::findPath
            ), "radius"_a, "constraints"_a, "instance"_a, "start_locs"_a, "goal_locs"_a
        )
        .def("findPath", 
            py::overload_cast<double, ConstraintTable&, Instance&, std::vector<size_t>&, std::vector<size_t>&>(
                &PlannerNameSpace::AStarSolver::findPath
            ), "radius"_a, "constrain_table"_a, "instance"_a, "start_locs"_a, "goal_locs"_a
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
        .def(py::init<>())
        .def_readonly("link_sign0", &PlannerNameSpace::TaskInfo::link_sign0)
        .def_readonly("link_sign1", &PlannerNameSpace::TaskInfo::link_sign1)
        .def_readonly("radius0", &PlannerNameSpace::TaskInfo::radius0)
        .def_readonly("radius1", &PlannerNameSpace::TaskInfo::radius1)
        .def_readonly("res_path", &PlannerNameSpace::TaskInfo::res_path);

    py::class_<PlannerNameSpace::SpanningTree_GroupSolver>(m, "SpanningTree_GroupSolver")
        .def(py::init<>())
        .def_readonly("task_seq", &PlannerNameSpace::SpanningTree_GroupSolver::task_seq)
        .def("getSequence_miniumSpanningTree", &PlannerNameSpace::SpanningTree_GroupSolver::getSequence_miniumSpanningTree, "instance"_a, "locs"_a)
        .def("insertPipe", &PlannerNameSpace::SpanningTree_GroupSolver::insertPipe, "pipeMap"_a, "instance"_a)
        .def("findPath", &PlannerNameSpace::SpanningTree_GroupSolver::findPath, "solver"_a, "constraints"_a, "instance"_a, "stepLength"_a);

    py::class_<CBSNameSpace::CBSNode>(m, "CBSNode")
        .def(py::init<double>(), "stepLength"_a)
        .def_readwrite("node_id", &CBSNameSpace::CBSNode::node_id)
        .def_readwrite("depth", &CBSNameSpace::CBSNode::depth)
        .def_readonly("g_val", &CBSNameSpace::CBSNode::g_val)
        .def_readonly("h_val", &CBSNameSpace::CBSNode::h_val)
        .def_readonly("firstConflict", &CBSNameSpace::CBSNode::firstConflict)
        .def_readonly("isConflict", &CBSNameSpace::CBSNode::isConflict)
        .def("update_Constrains", &CBSNameSpace::CBSNode::update_Constrains, "groupIdx"_a, "new_constrains"_a)
        .def("findFirstPipeConflict", &CBSNameSpace::CBSNode::findFirstPipeConflict)
        .def("compute_Heuristics", &CBSNameSpace::CBSNode::compute_Heuristics)
        .def("compute_Gval", &CBSNameSpace::CBSNode::compute_Gval)
        .def("copy", &CBSNameSpace::CBSNode::copy, "rhs"_a)
        .def("add_GroupAgent", &CBSNameSpace::CBSNode::add_GroupAgent, "groupIdx"_a, "pipeMap"_a, "instance"_a)
        .def("getConstrains", &CBSNameSpace::CBSNode::getConstrains, "groupIdx"_a)
        .def("info", &CBSNameSpace::CBSNode::info, "with_constrainInfo"_a=false, "with_pathInfo"_a=false)
        .def("getGroupAgent", &CBSNameSpace::CBSNode::getGroupAgent, "groupIdx"_a);

    py::class_<CBSNameSpace::CBSSolver>(m, "CBSSolver")
        .def(py::init<>())
        .def("pushNode", &CBSNameSpace::CBSSolver::pushNode, "node"_a)
        .def("popNode", &CBSNameSpace::CBSSolver::popNode)
        .def("is_openList_empty", &CBSNameSpace::CBSSolver::is_openList_empty)
        .def("isGoal", &CBSNameSpace::CBSSolver::isGoal)
        .def("addSearchEngine", &CBSNameSpace::CBSSolver::addSearchEngine, "groupIdx"_a, "with_AnyAngle"_a, "with_OrientCost"_a)
        .def("update_GroupAgentPath", &CBSNameSpace::CBSSolver::update_GroupAgentPath, "groupIdx"_a, "node"_a, "instance"_a);

    py::class_<PathNameSpace::PathNode>(m , "PathNode")
        .def(py::init<size_t, size_t, double, double, double, double>())
        .def_readonly("nodeIdx", &PathNameSpace::PathNode::nodeIdx)
        .def_readonly("groupIdx", &PathNameSpace::PathNode::groupIdx)
        .def_readonly("x", &PathNameSpace::PathNode::x)
        .def_readonly("y", &PathNameSpace::PathNode::y)
        .def_readonly("z", &PathNameSpace::PathNode::z)
        .def_readonly("radius", &PathNameSpace::PathNode::radius);

    py::class_<PathNameSpace::FlexGraphNode>(m, "FlexGraphNode")
        .def(py::init<size_t, double, double, double, double>())
        .def_readonly("nodeIdx", &PathNameSpace::FlexGraphNode::nodeIdx)
        .def_readonly("x", &PathNameSpace::FlexGraphNode::x)
        .def_readonly("y", &PathNameSpace::FlexGraphNode::y)
        .def_readonly("z", &PathNameSpace::FlexGraphNode::z)
        .def_readonly("alpha", &PathNameSpace::FlexGraphNode::alpha)
        .def_readonly("theta", &PathNameSpace::FlexGraphNode::theta)
        .def_readonly("radius", &PathNameSpace::FlexGraphNode::radius)
        .def_readonly("fixed", &PathNameSpace::FlexGraphNode::fixed)
        .def("vertex_x", &PathNameSpace::FlexGraphNode::vertex_x)
        .def("vertex_y", &PathNameSpace::FlexGraphNode::vertex_y)
        .def("vertex_z", &PathNameSpace::FlexGraphNode::vertex_z);
    
    py::class_<PathNameSpace::GroupPath>(m, "GroupPath")
        .def(py::init<size_t>())
        .def_readonly("pathNodeMap", &PathNameSpace::GroupPath::pathNodeMap)
        .def_readonly("graphPathMap", &PathNameSpace::GroupPath::graphPathMap)
        .def_readonly("graphNodeMap", &PathNameSpace::GroupPath::graphNodeMap)
        .def("insertPath", &PathNameSpace::GroupPath::insertPath, "path_xyzr"_a)
        // .def("setMaxRadius", &PathNameSpace::GroupPath::setMaxRadius, "radius"_a)
        .def("extractPath", py::overload_cast<size_t, size_t>(&PathNameSpace::GroupPath::extractPath), "start_nodeIdx"_a, "goal_nodeIdx"_a)
        .def("extractPath", py::overload_cast<double, double, double, double, double, double>(&PathNameSpace::GroupPath::extractPath), 
            "start_x"_a, "start_y"_a, "start_z"_a, "end_x"_a, "end_y"_a, "end_z"_a
        );
        // .def("findNodeIdx", &PathNameSpace::GroupPath::findNodeIdx, "x"_a, "y"_a, "z"_a);

    py::class_<SmootherNameSpace::SmootherXYZG2O>(m, "SmootherXYZG2O")
        .def(py::init<>())
        .def_readonly("groupMap", &SmootherNameSpace::SmootherXYZG2O::groupMap)
        .def("initOptimizer", &SmootherNameSpace::SmootherXYZG2O::initOptimizer)
        .def("add_Path", &SmootherNameSpace::SmootherXYZG2O::add_Path, "groupIdx"_a, "path_xyzr"_a)
        .def("add_OptimizePath", &SmootherNameSpace::SmootherXYZG2O::add_OptimizePath,
            "groupIdx"_a, "pathIdx"_a, 
            "start_x"_a, "start_y"_a, "start_z"_a,
            "end_x"_a, "end_y"_a, "end_z"_a,
            "startDire"_a, "endDire"_a
        )
        .def("insertStaticObs", &SmootherNameSpace::SmootherXYZG2O::insertStaticObs,
            "x"_a, "y"_a, "z"_a, "radius"_a, "alpha"_a, "theta"_a
        )
        .def("build_graph", &SmootherNameSpace::SmootherXYZG2O::build_graph, 
            "elasticBand_weight"_a, 
            "kinematic_weight"_a,
            "obstacle_weight"_a,
            "pipeConflict_weight"_a,
            "boundary_weight"_a
        )
        .def("loss_info", &SmootherNameSpace::SmootherXYZG2O::loss_info,
            "elasticBand_weight"_a, 
            "kinematic_weight"_a,
            "obstacle_weight"_a,
            "pipeConflict_weight"_a,
            "boundary_weight"_a
        )
        .def("setMaxRadius", &SmootherNameSpace::SmootherXYZG2O::setMaxRadius, "groupIdx"_a, "radius"_a)
        .def("setBoundary", &SmootherNameSpace::SmootherXYZG2O::setBoundary, 
            "xmin"_a, "ymin"_a, "zmin"_a, "xmax"_a, "ymax"_a, "zmax"_a
        )
        .def("setFlexible_percentage", &SmootherNameSpace::SmootherXYZG2O::setFlexible_percentage, "groupIdx"_a, "flexible_percentage"_a)
        .def("extractPath", &SmootherNameSpace::SmootherXYZG2O::extractPath, 
            "groupIdx"_a, "start_x"_a, "start_y"_a, "start_z"_a, "end_x"_a, "end_y"_a, "end_z"_a
        )
        .def("optimizeGraph", &SmootherNameSpace::SmootherXYZG2O::optimizeGraph, "no_iterations"_a, "verbose"_a)
        .def("clear_graph", &SmootherNameSpace::SmootherXYZG2O::clear_graph)
        .def("update2groupVertex", &SmootherNameSpace::SmootherXYZG2O::update2groupVertex)
        .def("info", &SmootherNameSpace::SmootherXYZG2O::info);

    // it don't work, I don't know why
    // m.def("printPointer", &printPointer<KDTreeData>, "a"_a, "tag"_a);

}
