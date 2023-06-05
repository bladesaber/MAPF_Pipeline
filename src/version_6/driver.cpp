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
#include "groupObjSolver.h"

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

    m.def("sampleDetailPath", &PlannerNameSpace::sampleDetailPath, "instance"_a, "path_xyzr"_a, "stepLength"_a);

    py::class_<PlannerNameSpace::AStarSolver>(m, "AStarSolver")
        .def(py::init<bool, bool>())
        .def_readonly("num_expanded", &PlannerNameSpace::AStarSolver::num_expanded)
        .def_readonly("num_generated", &PlannerNameSpace::AStarSolver::num_generated)
        .def_readonly("runtime_search", &PlannerNameSpace::AStarSolver::runtime_search)
        .def_readonly("runtime_build_CT", &PlannerNameSpace::AStarSolver::runtime_build_CT)
        .def("findPath", &PlannerNameSpace::AStarSolver::findPath, 
            "radius"_a, "constraints"_a, "instance"_a, "start_loc"_a, "goal_locs"_a
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

    py::class_<PlannerNameSpace::PathObjectInfo>(m, "PathObjectInfo")
        .def(py::init<size_t, bool>(), "pathIdx"_a, "fixed_end"_a)
        .def_readonly("pathIdx", &PlannerNameSpace::PathObjectInfo::pathIdx)
        .def_readonly("start_loc", &PlannerNameSpace::PathObjectInfo::start_loc)
        .def_readonly("goal_locs", &PlannerNameSpace::PathObjectInfo::goal_locs)
        .def_readonly("radius", &PlannerNameSpace::PathObjectInfo::radius)
        .def_readonly("fixed_end", &PlannerNameSpace::PathObjectInfo::fixed_end)
        .def_readonly("res_path", &PlannerNameSpace::PathObjectInfo::res_path);

    py::class_<PlannerNameSpace::MultiObjs_GroupSolver>(m, "MultiObjs_GroupSolver")
        .def(py::init<>())
        .def_readonly("objectiveMap", &PlannerNameSpace::MultiObjs_GroupSolver::objectiveMap)
        .def("insert_objs", &PlannerNameSpace::MultiObjs_GroupSolver::insert_objs, "locs"_a, "radius_list"_a, "instance"_a)
        .def("getSequence_miniumSpanningTree", &PlannerNameSpace::MultiObjs_GroupSolver::getSequence_miniumSpanningTree, "instance"_a, "locs"_a)
        .def("findPath", &PlannerNameSpace::MultiObjs_GroupSolver::findPath, "solver"_a, "constraints"_a, "instance"_a, "stepLength"_a);

    py::class_<CBSNameSpace::CBSNode>(m, "CBSNode")
        .def(py::init<double>(), "stepLength"_a)
        .def_readwrite("node_id", &CBSNameSpace::CBSNode::node_id)
        .def_readwrite("depth", &CBSNameSpace::CBSNode::depth)
        .def_readonly("g_val", &CBSNameSpace::CBSNode::g_val)
        .def_readonly("h_val", &CBSNameSpace::CBSNode::h_val)
        .def_readonly("firstConflict", &CBSNameSpace::CBSNode::firstConflict)
        .def_readonly("isConflict", &CBSNameSpace::CBSNode::isConflict)
        // .def_readonly("groupAgentMap", &CBSNameSpace::CBSNode::groupAgentMap)
        .def("update_Constrains", &CBSNameSpace::CBSNode::update_Constrains, "groupIdx"_a, "new_constrains"_a)
        // .def("update_GroupAgentPath", &CBSNameSpace::CBSNode::update_GroupAgentPath)
        .def("findFirstPipeConflict", &CBSNameSpace::CBSNode::findFirstPipeConflict)
        .def("compute_Heuristics", &CBSNameSpace::CBSNode::compute_Heuristics)
        .def("compute_Gval", &CBSNameSpace::CBSNode::compute_Gval)
        .def("copy", &CBSNameSpace::CBSNode::copy, "rhs"_a)
        .def("add_GroupAgent", &CBSNameSpace::CBSNode::add_GroupAgent, "groupIdx"_a, "locs"_a, "radius_list"_a, "instance"_a)
        .def("getConstrains", &CBSNameSpace::CBSNode::getConstrains, "groupIdx");

    py::class_<CBSNameSpace::CBSSolver>(m, "CBSSolver")
        .def(py::init<>())
        .def("pushNode", &CBSNameSpace::CBSSolver::pushNode, "node"_a)
        .def("popNode", &CBSNameSpace::CBSSolver::popNode)
        .def("is_openList_empty", &CBSNameSpace::CBSSolver::is_openList_empty)
        .def("isGoal", &CBSNameSpace::CBSSolver::isGoal)
        .def("addSearchEngine", &CBSNameSpace::CBSSolver::addSearchEngine, "groupIdx"_a, "with_AnyAngle"_a, "with_OrientCost"_a)
        .def("update_GroupAgentPath", &CBSNameSpace::CBSSolver::update_GroupAgentPath, "groupIdx"_a, "node"_a, "instance"_a);

    py::class_<PathNameSpace::GroupPathNode>(m, "GroupPathNode")
        .def(py::init<size_t, size_t, size_t, double, double, double, double>())
        .def_readonly("nodeIdx", &PathNameSpace::GroupPathNode::nodeIdx)
        .def_readonly("groupIdx", &PathNameSpace::GroupPathNode::groupIdx)
        .def_readonly("x", &PathNameSpace::GroupPathNode::x)
        .def_readonly("y", &PathNameSpace::GroupPathNode::y)
        .def_readonly("z", &PathNameSpace::GroupPathNode::z)
        .def_readonly("alpha", &PathNameSpace::GroupPathNode::alpha)
        .def_readonly("theta", &PathNameSpace::GroupPathNode::theta)
        .def_readonly("radius", &PathNameSpace::GroupPathNode::radius)
        .def("vertex_x", &PathNameSpace::GroupPathNode::vertex_x)
        .def("vertex_y", &PathNameSpace::GroupPathNode::vertex_y)
        .def("vertex_z", &PathNameSpace::GroupPathNode::vertex_z);
    
    py::class_<PathNameSpace::GroupPath>(m, "GroupPath")
        .def(py::init<size_t>())
        .def_readonly("pathIdxs_set", &PathNameSpace::GroupPath::pathIdxs_set)
        .def_readonly("nodeMap", &PathNameSpace::GroupPath::nodeMap)
        .def("insertPath", &PathNameSpace::GroupPath::insertPath, 
            "pathIdx"_a, "path_xyzr"_a, "fixed_start"_a, "fixed_end"_a, "startDire"_a, "endDire"_a, "merge_path"_a
        )
        .def("extractPath", &PathNameSpace::GroupPath::extractPath, "pathIdx"_a)
        .def("create_pathTree", &PathNameSpace::GroupPath::create_pathTree);

    py::class_<SmootherNameSpace::SmootherXYZG2O>(m, "SmootherXYZG2O")
        .def(py::init<>())
        .def_readonly("groupMap", &SmootherNameSpace::SmootherXYZG2O::groupMap)
        .def("initOptimizer", &SmootherNameSpace::SmootherXYZG2O::initOptimizer)
        .def("addPath", &SmootherNameSpace::SmootherXYZG2O::addPath, 
            "groupIdx"_a, "pathIdx"_a, "path_xyzr"_a, "fixed_start"_a, "fixed_end"_a, "startDire"_a, "endDire"_a, "merge_path"_a
        )
        .def("insertStaticObs", &SmootherNameSpace::SmootherXYZG2O::insertStaticObs,
            "x"_a, "y"_a, "z"_a, "radius"_a, "alpha"_a, "theta"_a
        )
        .def("build_graph", &SmootherNameSpace::SmootherXYZG2O::build_graph, 
            "elasticBand_weight"_a, 
            "kinematic_weight"_a,
            "obstacle_weight"_a,
            "pipeConflict_weight"_a
        )
        .def("loss_info", &SmootherNameSpace::SmootherXYZG2O::loss_info,
            "elasticBand_weight"_a, 
            "kinematic_weight"_a,
            "obstacle_weight"_a,
            "pipeConflict_weight"_a
        )
        .def("optimizeGraph", &SmootherNameSpace::SmootherXYZG2O::optimizeGraph, "no_iterations"_a, "verbose"_a)
        .def("clear_graph", &SmootherNameSpace::SmootherXYZG2O::clear_graph)
        .def("update2groupVertex", &SmootherNameSpace::SmootherXYZG2O::update2groupVertex)
        .def("info", &SmootherNameSpace::SmootherXYZG2O::info);

    // it don't work, I don't know why
    // m.def("printPointer", &printPointer<KDTreeData>, "a"_a, "tag"_a);

}
