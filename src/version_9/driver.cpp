#include <pybind11/pybind11.h>
#include "iostream"

#include "common.h"
#include "utils.h"
#include "vector3d.h"
#include "instance.h"
#include "conflict.h"
#include "constrainTable.h"
#include "AstarSolver.h"

//#include "vertex_XYZ.h"
//#include "g2oSmoother_xyz.h"
#include "tightSpringer/springSmoother.h"

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

    py::class_<TightSpringNameSpace::Springer_Plane>(m, "SpringerPlane")
            .def(py::init<std::string, size_t, double, double, double, bool>())
            .def_readonly("name", &TightSpringNameSpace::Springer_Plane::name)
            .def_readonly("nodeIdx", &TightSpringNameSpace::Springer_Plane::nodeIdx)
            .def_readonly("x", &TightSpringNameSpace::Springer_Plane::x)
            .def_readonly("y", &TightSpringNameSpace::Springer_Plane::y)
            .def_readonly("z", &TightSpringNameSpace::Springer_Plane::z)
            .def_readonly("fixed", &TightSpringNameSpace::Springer_Plane::fixed);

    py::class_<TightSpringNameSpace::Springer_Cell>(m, "SpringerCell")
            .def(py::init<size_t, double, double, double, double, bool>())
            .def_readonly("nodeIdx", &TightSpringNameSpace::Springer_Cell::nodeIdx)
            .def_readonly("x", &TightSpringNameSpace::Springer_Cell::x)
            .def_readonly("y", &TightSpringNameSpace::Springer_Cell::y)
            .def_readonly("z", &TightSpringNameSpace::Springer_Cell::z)
            .def_readonly("radius", &TightSpringNameSpace::Springer_Cell::radius)
            .def_readonly("fixed", &TightSpringNameSpace::Springer_Cell::fixed);

    py::class_<TightSpringNameSpace::Springer_Connector>(m, "SpringerConnector")
            .def(py::init<std::string, size_t, double, double, double, bool>())
            .def_readonly("name", &TightSpringNameSpace::Springer_Connector::name)
            .def_readonly("nodeIdx", &TightSpringNameSpace::Springer_Connector::nodeIdx)
            .def_readonly("x", &TightSpringNameSpace::Springer_Connector::x)
            .def_readonly("y", &TightSpringNameSpace::Springer_Connector::y)
            .def_readonly("z", &TightSpringNameSpace::Springer_Connector::z)
            .def_readonly("fixed", &TightSpringNameSpace::Springer_Connector::fixed);

    py::class_<TightSpringNameSpace::Springer_Structor>(m, "SpringerStructor")
            .def(py::init<std::string, size_t, std::string, double, double, double, double, double, bool>())
            .def_readonly("name", &TightSpringNameSpace::Springer_Structor::name)
            .def_readonly("nodeIdx", &TightSpringNameSpace::Springer_Structor::nodeIdx)
            .def_readonly("xyzTag", &TightSpringNameSpace::Springer_Structor::xyzTag)
            .def_readonly("x", &TightSpringNameSpace::Springer_Structor::x)
            .def_readonly("y", &TightSpringNameSpace::Springer_Structor::y)
            .def_readonly("z", &TightSpringNameSpace::Springer_Structor::z)
            .def_readonly("radian", &TightSpringNameSpace::Springer_Structor::radian)
            .def_readonly("shell_radius", &TightSpringNameSpace::Springer_Structor::shell_radius)
            .def_readonly("fixRadian", &TightSpringNameSpace::Springer_Structor::fixRadian)
            .def_readonly("fixed", &TightSpringNameSpace::Springer_Structor::fixed);

    py::class_<TightSpringNameSpace::SpringerSmooth_Runner>(m, "SpringerSmooth_Runner")
            .def(py::init<>())
            .def_readonly("plane_NodeMap", &TightSpringNameSpace::SpringerSmooth_Runner::plane_NodeMap)
            .def_readonly("cell_NodeMap", &TightSpringNameSpace::SpringerSmooth_Runner::cell_NodeMap)
            .def_readonly("connector_NodeMap", &TightSpringNameSpace::SpringerSmooth_Runner::connector_NodeMap)
            .def_readonly("structor_NodeMap", &TightSpringNameSpace::SpringerSmooth_Runner::structor_NodeMap)
            .def("initOptimizer", &TightSpringNameSpace::SpringerSmooth_Runner::initOptimizer, "method"_a)
            .def("clear_graph", &TightSpringNameSpace::SpringerSmooth_Runner::clear_graph)
            .def("is_g2o_graph_empty", &TightSpringNameSpace::SpringerSmooth_Runner::is_g2o_graph_empty)
            .def("optimizeGraph", &TightSpringNameSpace::SpringerSmooth_Runner::optimizeGraph)
            .def("info", &TightSpringNameSpace::SpringerSmooth_Runner::info)
            .def("add_Plane", &TightSpringNameSpace::SpringerSmooth_Runner::add_Plane,
                 "name"_a, "nodeIdx"_a, "x"_a, "y"_a, "z"_a, "fixed"_a)
            .def("add_Cell", &TightSpringNameSpace::SpringerSmooth_Runner::add_Cell,
                 "nodeIdx"_a, "x"_a, "y"_a, "z"_a, "radius"_a, "fixed"_a)
            .def("add_Connector", &TightSpringNameSpace::SpringerSmooth_Runner::add_Connector,
                 "name"_a, "nodeIdx"_a, "x"_a, "y"_a, "z"_a, "fixed"_a)
            .def("add_Structor", &TightSpringNameSpace::SpringerSmooth_Runner::add_Structor,
                 "name"_a, "nodeIdx"_a, "xyzTag"_a, "x"_a, "y"_a, "z"_a, "radian"_a, "shell_radius"_a, "fixed"_a)
            .def("add_vertexes", &TightSpringNameSpace::SpringerSmooth_Runner::add_vertexes)
            .def("clear_graphNodes", &TightSpringNameSpace::SpringerSmooth_Runner::clear_graphNodes)
            .def("clear_vertexes", &TightSpringNameSpace::SpringerSmooth_Runner::clear_vertexes)
            .def("update_nodeMapVertex", &TightSpringNameSpace::SpringerSmooth_Runner::update_nodeMapVertex)
            .def("addEdge_structorToPlane_valueShift", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_structorToPlane_valueShift,
                 "planeIdx"_a, "structorIdx"_a, "xyzTag"_a, "shiftValue"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_connectorToStruct_valueShift", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_connectorToStruct_valueShift,
                 "connectorIdx"_a, "structorIdx"_a, "xyzTag"_a, "shiftValue"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_connectorToStruct_radiusFixed", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_connectorToStruct_radiusFixed,
                 "connectorIdx"_a, "structorIdx"_a, "xyzTag"_a, "radius"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_connectorToStruct_poseFixed", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_connectorToStruct_poseFixed,
                 "connectorIdx"_a, "structorIdx"_a, "xyzTag"_a, "shapeX"_a, "shapeY"_a, "shapeZ"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_cellToConnector_elasticBand", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_cellToConnector_elasticBand,
                 "cellIdx"_a, "connectorIdx"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_cellToCell_elasticBand", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_cellToCell_elasticBand,
                 "cellIdx0"_a, "cellIdx1"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_cellToConnector_kinematicPoint", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_cellToConnector_kinematicPoint,
                 "cellIdx"_a, "connectorIdx"_a, "vecX"_a, "vecY"_a, "vecZ"_a, "targetValue"_a, "fromCell"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_cellToConnector_kinematicSegment", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_cellToConnector_kinematicSegment,
                 "connectorIdx"_a, "cellIdx0"_a, "cellIdx1"_a, "targetValue"_a, "fromConnector"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_cellToCell_kinematicSegment", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_cellToCell_kinematicSegment,
                 "cellIdx0"_a, "cellIdx1"_a, "cellIdx2"_a, "targetValue"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_structorToPlane_planeRepel", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_structorToPlane_planeRepel,
                 "structorIdx"_a, "planeIdx"_a, "planeTag"_a, "compareTag"_a, "conflict_xyzs"_a, "bound_shift"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_cellToPlane_planeRepel", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_cellToPlane_planeRepel,
                 "cellIdx"_a, "planeIdx"_a, "planeTag"_a, "compareTag"_a, "bound_shift"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_cellToCell_shapeRepel", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_cellToCell_shapeRepel,
                 "cellIdx0"_a, "cellIdx1"_a, "bound_shift"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_cellToStructor_shapeRepel", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_cellToStructor_shapeRepel,
                 "cellIdx"_a, "structorIdx"_a, "shapeX"_a, "shapeY"_a, "shapeZ"_a, "bound_shift"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_structorToStructor_shapeRepel", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_structorToStructor_shapeRepel,
                 "structorIdx0"_a, "structorIdx1"_a, "shapeX_0"_a, "shapeY_0"_a, "shapeZ_0"_a,
                 "shapeX_1"_a, "shapeY_1"_a, "shapeZ_1"_a, "bound_shift"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_minVolume", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_minVolume,
                 "minPlaneIdx"_a, "maxPlaneIdx"_a, "scale"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_structor_poseCluster", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_structor_poseCluster,
                 "structorIdx"_a, "scale"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_connector_poseCluster", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_connector_poseCluster,
                 "connectorIdx"_a, "scale"_a, "kSpring"_a, "weight"_a)
            .def("addEdge_minAxes", &TightSpringNameSpace::SpringerSmooth_Runner::addEdge_minAxes,
                 "minPlaneIdx"_a, "maxPlaneIdx"_a, "xyzTag"_a, "scale"_a, "kSpring"_a, "weight"_a);


    // it don't work, I don't know why
    // m.def("printPointer", &printPointer<KDTreeData>, "a"_a, "tag"_a);

}
