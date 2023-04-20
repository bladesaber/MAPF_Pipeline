#include <pybind11/pybind11.h>
#include "iostream"

#include "common.h"
#include "utils.h"
#include "kdtreeWrapper.h"
#include "instance.h"
#include "constrainTable.h"
#include "angleAstar.h"
#include "cbs.h"

#include "test.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(mapf_pipeline, m) {
    m.doc() = "mapf pipeline"; // optional module docstring

    m.def("debug_kdtree", &debug_kdtree);

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
        .def("isConstrained", &ConstraintTable::isConstrained, "x"_a, "y"_a, "z"_a, "radius"_a)
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
        .def(py::init<double, double>())
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

    // py::class_<AgentInfo>(m, "AgentInfo")
    //     .def(py::init<size_ut, double>())
    //     .def_readonly("agentIdx", &AgentInfo::agentIdx)
    //     .def_readonly("radius", &AgentInfo::radius)
    //     .def_readonly("detailPath", &AgentInfo::detailPath)
    //     .def_readonly("pathTree", &AgentInfo::pathTree)
    //     .def_readonly("isConflict", &AgentInfo::isConflict)
    //     .def_readonly("firstConflict", &AgentInfo::firstConflict)
    //     .def_readonly("firstConflictLength", &AgentInfo::firstConflictLength)
    //     .def_readonly("costMap", &AgentInfo::costMap);

    // py::class_<CBSNode>(m, "CBSNode")
    //     .def(py::init<int>());

    // py::class_<CBS>(m, "CBS")
    //     .def(py::init<>())
    //     .def("sampleDetailPath", &CBS::sampleDetailPath, "path"_a, "instance"_a, "stepLength"_a)
    //     .def("findConflictFromTree", &CBS::findConflictFromTree, "tree"_a, "path"_a, "bound"_a);

    // it don't work, I don't know why
    // m.def("printPointer", &printPointer<KDTreeData>, "a"_a, "tag"_a);

}
