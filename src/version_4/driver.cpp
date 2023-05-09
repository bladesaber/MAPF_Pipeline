#include <pybind11/pybind11.h>
#include "iostream"

#include "common.h"
#include "utils.h"
#include "vector3d.h"
#include "kdtreeWrapper.h"
#include "instance.h"
#include "conflict.h"
#include "constrainTable.h"
#include "angleAstar.h"
#include "cbs.h"
#include "smoother.h"
#include "smoother_random.h"
#include "groupPath.h"

#include "test.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(mapf_pipeline, m) {
    m.doc() = "mapf pipeline"; // optional module docstring

    // m.def("debug_kdtree", &debug_kdtree);
    // m.def("debug_sharePtr", &debug_sharePtr);
    // m.def("debug_setTuple", &debug_setTuple);

    m.def("mod2pi", &mod2pi, "theta"_a);
    m.def("mod2singlePi", &mod2singlePi, "theta"_a);
    m.def("polar3D_to_vec3D", &polar3D_to_vec3D, "alpha"_a, "beta"_a, "length"_a);
    m.def("vec3D_to_polar3D", &vec3D_to_polar3D, "vec_x"_a, "vec_y"_a, "vec_z"_a);
    m.def("point2LineDistance", &point2LineDistance, 
        "lineStart_x"_a, "lineStart_y"_a, "lineStart_z"_a,
        "lineEnd_x"_a, "lineEnd_y"_a, "lineEnd_z"_a,
        "point_x"_a, "point_y"_a, "point_z"_a
    );
    m.def("point2LineSegmentDistance", &point2LineSegmentDistance, 
        "lineStart_x"_a, "lineStart_y"_a, "lineStart_z"_a,
        "lineEnd_x"_a, "lineEnd_y"_a, "lineEnd_z"_a,
        "point_x"_a, "point_y"_a, "point_z"_a
    );
    m.def("norm2_distance", &norm2_distance, "x0"_a, "y0"_a, "z0"_a, "x1"_a, "y1"_a, "z1"_a);

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
        .def("isConstrained", py::overload_cast<Instance&, int, int, double>(&ConstraintTable::isConstrained), "instance"_a, "parent_loc"_a, "child_loc"_a, "radius"_a)
        .def("islineOnSight", &ConstraintTable::islineOnSight, "instance"_a, "parent_loc"_a, "child_loc"_a, "bound"_a);

    py::class_<AngleAStar>(m, "AngleAStar")
        .def(py::init<double>())
        .def_readonly("num_expanded", &AngleAStar::num_expanded)
        .def_readonly("num_generated", &AngleAStar::num_generated)
        .def_readonly("runtime_search", &AngleAStar::runtime_search)
        .def_readonly("runtime_build_CT", &AngleAStar::runtime_build_CT)
        // .def_readonly("runtime_build_CAT", &AngleAStar::runtime_build_CAT)
        .def("findPath", &AngleAStar::findPath, "constraints"_a, "instance"_a, "start_state"_a, "goal_state"_a);

    py::class_<KDTreeData>(m, "KDTreeData")
        .def(py::init<double, double, size_t>())
        .def_readonly("radius", &KDTreeData::radius)
        .def_readonly("length", &KDTreeData::length);
    
    py::class_<KDTreeRes>(m, "KDTreeRes")
        .def(py::init<double, double, double, KDTreeData*>())
        .def(py::init<>())
        .def_readonly("x", &KDTreeRes::x)
        .def_readonly("y", &KDTreeRes::y)
        .def_readonly("z", &KDTreeRes::z)
        .def_readonly("data", &KDTreeRes::data);

    py::class_<KDTreeWrapper>(m, "KDTreeWrapper")
        .def(py::init<>())
        .def("insertPoint3D", &KDTreeWrapper::insertPoint3D, "x"_a, "y"_a, "z"_a, "data"_a)
        .def("insertPath3D", &KDTreeWrapper::insertPath3D, "path"_a, "radius"_a)
        .def("nearest", &KDTreeWrapper::nearest, "x"_a, "y"_a, "z"_a, "res"_a)
        .def("clear", &KDTreeWrapper::clear)
        .def("debug_insert", &KDTreeWrapper::debug_insert)
        .def("debug_search", &KDTreeWrapper::debug_search);

    py::class_<Conflict>(m, "Conflict")
        .def(py::init<>())
        .def(py::init<
            size_ut, 
            double, double, double, double, double,
            size_ut, 
            double, double, double, double, double
            >())
        .def_readonly("agent1", &Conflict::agent1)
        .def_readonly("conflict1_x", &Conflict::conflict1_x)
        .def_readonly("conflict1_y", &Conflict::conflict1_y)
        .def_readonly("conflict1_z", &Conflict::conflict1_z)
        .def_readonly("conflict1_radius", &Conflict::conflict1_radius)
        .def_readonly("agent2", &Conflict::agent2)
        .def_readonly("conflict2_x", &Conflict::conflict2_x)
        .def_readonly("conflict2_y", &Conflict::conflict2_y)
        .def_readonly("conflict2_z", &Conflict::conflict2_z)
        .def_readonly("conflict2_radius", &Conflict::conflict2_radius)
        .def_readonly("constrain1", &Conflict::constrain1)
        .def_readonly("constrain2", &Conflict::constrain2)
        .def("conflictExtend", &Conflict::conflictExtend)
        .def("getMinLength", &Conflict::getMinLength)
        .def("info", &Conflict::info);

    py::class_<AgentInfo>(m, "AgentInfo")
        .def(py::init<>())
        .def(py::init<
            size_ut, size_ut, double, 
            std::tuple<int, int, int>, 
            std::tuple<int, int, int>
            >(), "agentIdx"_a, "groupIdx"_a, "radius"_a, "startPos"_a, "endPos"_a)
        // .def(py::init<const AgentInfo&>(), "rhs"_a)
        .def_readonly("agentIdx", &AgentInfo::agentIdx)
        .def_readonly("groupIdx", &AgentInfo::groupIdx)
        .def_readonly("radius", &AgentInfo::radius)
        .def_readonly("isConflict", &AgentInfo::isConflict)
        .def_readonly("firstConflict", &AgentInfo::firstConflict)
        .def_readonly("conflictSet", &AgentInfo::conflictSet)
        .def_readonly("conflictNum", &AgentInfo::conflictNum)
        .def_readonly("findPath_Success", &AgentInfo::findPath_Success)
        .def("getConstrains", &AgentInfo::getConstrains)
        .def("getDetailPath", &AgentInfo::getDetailPath)
        .def("update_Constrains", &AgentInfo::update_Constrains, "new_constrains"_a)
        .def("update_DetailPath_And_Tree", &AgentInfo::update_DetailPath_And_Tree, "path"_a)
        .def("copy", &AgentInfo::copy, "rhs"_a)
        .def("info", &AgentInfo::info)
        .def("debug", &AgentInfo::debug);

    py::class_<CBSNode>(m, "CBSNode")
        .def(py::init<int>(), "num_of_agents"_a)
        .def_readwrite("node_id", &CBSNode::node_id)
        .def_readwrite("depth", &CBSNode::depth)
        .def_readonly("g_val", &CBSNode::g_val)
        .def_readonly("h_val", &CBSNode::h_val)
        .def_readonly("num_of_agents", &CBSNode::num_of_agents)
        .def_readonly("agentMap", &CBSNode::agentMap)
        // .def("updateAgentConflict", &CBSNode::updateAgentConflict, "agentIdx"_a)
        .def("findAllAgentConflict", &CBSNode::findAllAgentConflict)
        .def("update_Constrains", &CBSNode::update_Constrains, "agentIdx"_a, "new_constrains"_a)
        .def("update_DetailPath_And_Tree", &CBSNode::update_DetailPath_And_Tree, "agentIdx"_a, "path"_a)
        .def("addAgent", &CBSNode::addAgent, "agentIdx"_a, "groupIdx"_a, "radius"_a, "startPos"_a, "endPos"_a)
        .def("copy", &CBSNode::copy, "rhs"_a)
        .def("debug", &CBSNode::debug);

    py::class_<CBS>(m, "CBS")
        .def(py::init<>())
        .def_readonly("runtime_search", &CBS::runtime_search)
        .def("pushNode", &CBS::pushNode, "node"_a)
        .def("popNode", &CBS::popNode)
        .def("is_openList_empty", &CBS::is_openList_empty)
        .def("addSearchEngine", &CBS::addSearchEngine, "agentIdx"_a, "radius"_a)
        .def("sampleDetailPath", &CBS::sampleDetailPath, "path"_a, "instance"_a, "stepLength"_a)
        .def("compute_Heuristics", &CBS::compute_Heuristics, "node"_a)
        .def("compute_Gval", &CBS::compute_Gval, "node"_a)
        .def("isGoal", &CBS::isGoal, "node"_a)
        .def("update_AgentPath", &CBS::update_AgentPath, "instance"_a, "node"_a, "agentIdx"_a)
        .def("info", &CBS::info);

    py::class_<Vector3D>(m, "Vector3D")
        .def(py::init<const double, const double, const double>())
        .def("getX", &Vector3D::getX)
        .def("getY", &Vector3D::getY)
        .def("getZ", &Vector3D::getZ);
    
    py::class_<GroupPathNode>(m, "GroupPathNode")
        .def(py::init<size_ut, size_ut, double, double, double, double>())
        .def_readonly("nodeIdx", &GroupPathNode::nodeIdx)
        .def_readonly("x", &GroupPathNode::x)
        .def_readonly("y", &GroupPathNode::y)
        .def_readonly("z", &GroupPathNode::z)
        .def_readonly("radius", &GroupPathNode::radius)
        .def_readonly("parentIdxsMap", &GroupPathNode::parentIdxsMap)
        .def_readonly("childIdxsMap", &GroupPathNode::childIdxsMap);

    py::class_<GroupPath>(m, "GroupPath")
        .def(py::init<size_ut>())
        .def_readonly("startPathIdxMap", &GroupPath::startPathIdxMap)
        .def_readonly("endPathIdxMap", &GroupPath::endPathIdxMap)
        .def_readonly("nodeMap", &GroupPath::nodeMap)
        .def_readonly("pathIdxs_set", &GroupPath::pathIdxs_set)
        .def("insertPath", &GroupPath::insertPath, "pathIdx"_a, "detailPath"_a, "radius"_a)
        .def("extractPath", &GroupPath::extractPath, "pathIdx"_a);

    py::class_<GroupSmoothInfo>(m, "GroupSmoothInfo")
        .def(py::init<GroupPath*>())
        .def_readonly("path", &GroupSmoothInfo::path)
        .def_readonly("groupIdx", &GroupSmoothInfo::groupIdx)
        .def_readonly("grads", &GroupSmoothInfo::grads);

    py::class_<RandomStep_Smoother>(m, "RandomStep_Smoother")
        .def(py::init<double, double, double, double, double, double, double>(), 
            "xmin"_a, "xmax"_a, "ymin"_a, "ymax"_a, "zmin"_a, "zmax"_a, "stepReso"_a=0.1
        )
        .def("smoothPath", &RandomStep_Smoother::smoothPath, "updateTimes"_a)
        .def("addDetailPath", &RandomStep_Smoother::addDetailPath, "groupIdx"_a, "pathIdx"_a, "detailPath"_a, "radius"_a)
        .def("paddingPath", &RandomStep_Smoother::paddingPath, 
            "detailPath"_a, "startPadding"_a, "endPadding"_a,
            "x_shift"_a, "y_shift"_a, "z_shift"_a
        )
        .def("detailSamplePath", &RandomStep_Smoother::detailSamplePath, "path"_a, "stepLength"_a)
        .def_readonly("groupMap", &RandomStep_Smoother::groupMap)
        .def_readwrite("wSmoothness", &RandomStep_Smoother::wSmoothness)
        .def_readwrite("wObstacle", &RandomStep_Smoother::wObstacle)
        .def_readwrite("wCurvature", &RandomStep_Smoother::wCurvature);

    // it don't work, I don't know why
    // m.def("printPointer", &printPointer<KDTreeData>, "a"_a, "tag"_a);

}
