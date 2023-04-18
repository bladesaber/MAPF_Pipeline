#include <pybind11/pybind11.h>
#include "iostream"

#include "common.h"
#include "utils.h"
#include "instance.h"
#include "constrainTable.h"
#include "angleAstar.h"
#include "cbs.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(mapf_pipeline, m) {
    m.doc() = "mapf pipeline"; // optional module docstring

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
        .def("insert2CT", &ConstraintTable::insert2CT, "loc"_a, "radius"_a)
        // .def("insert2CAT", &ConstraintTable::insert2CAT, "loc"_a)
        .def("getCT", &ConstraintTable::getCT)
        // .def("getCAT", &ConstraintTable::getCAT)
        // .def("insertConstrains2CT", &ConstraintTable::insertConstrains2CT, "constrains"_a)
        // .def("insertPath2CAT", &ConstraintTable::insertPath2CAT, "path"_a)
        .def("isConstrained", &ConstraintTable::isConstrained, "loc"_a)
        // .def("getNumOfConflictsForStep", &ConstraintTable::getNumOfConflictsForStep, "curr_loc"_a, "next_loc"_a)
        .def("islineOnSight", &ConstraintTable::islineOnSight, "instance"_a, "parent_loc"_a, "child_loc"_a, "bound"_a);

    py::class_<AngleAStar>(m, "AngleAStar")
        .def(py::init<double>())
        .def_readonly("num_expanded", &AngleAStar::num_expanded)
        .def_readonly("num_generated", &AngleAStar::num_generated)
        .def_readonly("runtime_search", &AngleAStar::runtime_search)
        .def_readonly("runtime_build_CT", &AngleAStar::runtime_build_CT)
        // .def_readonly("runtime_build_CAT", &AngleAStar::runtime_build_CAT)
        .def("findPath", &AngleAStar::findPath, "constraints"_a, "instance"_a, "start_state"_a, "goal_state"_a);

    py::class_<KDTreeWrapper>(m, "KDTreeWrapper")
        .def(py::init<>())
        .def("free", &KDTreeWrapper::free)
        .def("insertPoint", &KDTreeWrapper::insertPoint, "x"_a, "y"_a, "z"_a)
        .def("insertPath", &KDTreeWrapper::insertPath, "path"_a)
        .def("nearest", &KDTreeWrapper::nearest, "x"_a, "y"_a, "z"_a);

    py::class_<CBS>(m, "CBS")
        .def(py::init<>())
        .def("sampleDetailPath", &CBS::sampleDetailPath, "path"_a, "instance"_a, "stepLength"_a)
        .def("findConflictFromTree", &CBS::findConflictFromTree, "tree"_a, "path"_a, "bound"_a);

}
