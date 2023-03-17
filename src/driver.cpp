//
// Created by quan on 23-3-13.
//

#include <pybind11/pybind11.h>
#include "iostream"

#include "test.h"
#include "common.h"
#include "spaceTimeAstar.h"
#include "instance.h"
#include "constraintTable.h"
#include "conflict.h"
#include "cbs.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(mapf_pipeline, m) {
    m.doc() = "mapf pipeline"; // optional module docstring

    // Debug Functions
    m.def("testPring", &testPring, "A function for test", "i"_a, "j"_a);
    m.def("testPring_vector", &debugPring_vector, "A function for test vector", "i"_a);
    m.def("testPring_list", &debugPring_list, "A function for test list", "i"_a);
    m.def("testPring_map", &debugPring_map, "A function for test map", "i"_a);
    m.def("testPring_pair", &debugPring_pair, "A function for test pair", "i"_a);
    m.def("testPring_tuple", &debugPring_tuple, "A function for test tuple", "i"_a);
    m.def("debugPring", &debugPring, "A function for debug");

    py::class_<Instance>(m, "Instance")
        .def(py::init<int, int>())
        .def_readonly("num_of_cols", &Instance::num_of_cols)
        .def_readonly("num_of_rows", &Instance::num_of_rows)
        .def_readonly("map_size", &Instance::map_size)
        .def("print", &Instance::print)
        .def("getRowCoordinate", &Instance::getRowCoordinate, "curr"_a)
        .def("getColCoordinate", &Instance::getColCoordinate, "curr"_a)
        .def("getCoordinate", &Instance::getCoordinate, "curr"_a)
        .def("getNeighbors", &Instance::getNeighbors, "curr"_a)
        .def("linearizeCoordinate", py::overload_cast<int, int>(&Instance::linearizeCoordinate, py::const_), "row"_a, "col"_a)
        .def("linearizeCoordinate", py::overload_cast<const std::pair<int, int>&>(&Instance::linearizeCoordinate, py::const_), "curr"_a)
        .def("getManhattanDistance", py::overload_cast<int, int>(&Instance::getManhattanDistance, py::const_), "loc1"_a, "loc2"_a)
        .def("getManhattanDistance", py::overload_cast<const std::pair<int, int>&, const std::pair<int, int>&>(
            &Instance::getManhattanDistance, py::const_), "loc1"_a, "loc2"_a);

    py::class_<Instance3D, Instance>(m, "Instance3D")
        .def(py::init<int, int, int>())
        .def_readonly("num_of_cols", &Instance3D::num_of_cols)
        .def_readonly("num_of_rows", &Instance3D::num_of_rows)
        .def_readonly("num_of_z", &Instance3D::num_of_z)
        .def_readonly("map_size", &Instance3D::map_size)
        .def("print", &Instance3D::print)
        .def("getRowCoordinate", &Instance3D::getRowCoordinate, "curr"_a)
        .def("getColCoordinate", &Instance3D::getColCoordinate, "curr"_a)
        .def("getZCoordinate", &Instance3D::getZCoordinate, "curr"_a)
        .def("getCoordinate", &Instance3D::getCoordinate, "curr"_a)
        .def("getNeighbors", &Instance3D::getNeighbors, "curr"_a)
        .def("linearizeCoordinate", py::overload_cast<int, int, int>(&Instance3D::linearizeCoordinate, py::const_), "row"_a, "col"_a, "z"_a)
        .def("linearizeCoordinate", py::overload_cast<const std::tuple<int, int, int>&>(&Instance3D::linearizeCoordinate, py::const_), "curr"_a)
        .def("getManhattanDistance", py::overload_cast<int, int>(&Instance3D::getManhattanDistance, py::const_), "loc1"_a, "loc2"_a)
        .def("getManhattanDistance", py::overload_cast<const std::tuple<int, int, int>&, const std::tuple<int, int, int>&>(
            &Instance3D::getManhattanDistance, py::const_), "loc1"_a, "loc2"_a)
        .def("printCoordinate", &Instance3D::printCoordinate);

    py::enum_<constraint_type>(m, "constraint_type")
            .value("LEQLENGTH", constraint_type::LEQLENGTH)
            .value("GLENGTH", constraint_type::GLENGTH)
            .value("RANGE", constraint_type::RANGE)
            .value("BARRIER", constraint_type::BARRIER)
            .value("VERTEX", constraint_type::VERTEX)
            .value("EDGE", constraint_type::EDGE)
            .value("POSITIVE_VERTEX", constraint_type::POSITIVE_VERTEX)
            .value("POSITIVE_EDGE", constraint_type::POSITIVE_EDGE)
            .value("POSITIVE_BARRIER", constraint_type::POSITIVE_BARRIER)
            .value("POSITIVE_RANGE", constraint_type::POSITIVE_RANGE)
            .export_values();

    py::class_<ConstraintTable>(m, "ConstraintTable")
        .def(py::init<>())
        .def("insert2CT", &ConstraintTable::insert2CT, "loc"_a)
        .def("insert2CAT", &ConstraintTable::insert2CAT, "loc"_a)
        .def("getCT", &ConstraintTable::getCT)
        .def("getCAT", &ConstraintTable::getCAT)
        .def("insertConstrains2CT", &ConstraintTable::insertConstrains2CT, "constrains"_a)
        .def("insertPath2CAT", &ConstraintTable::insertPath2CAT, "path"_a)
        .def("isConstrained", &ConstraintTable::isConstrained, "loc"_a)
        .def("getNumOfConflictsForStep", &ConstraintTable::getNumOfConflictsForStep, "curr_loc"_a, "next_loc"_a);
    
    py::class_<SpaceTimeAStar>(m, "SpaceTimeAStar")
        .def(py::init<int>())
        .def_readwrite("focus_optimal", &SpaceTimeAStar::focus_optimal)
        .def_readwrite("focus_w", &SpaceTimeAStar::focus_w)
        .def_readwrite("bandwith", &SpaceTimeAStar::bandwith)
        .def_readonly("num_expanded", &SpaceTimeAStar::num_expanded)
        .def_readonly("num_generated", &SpaceTimeAStar::num_generated)
        .def_readonly("runtime_search", &SpaceTimeAStar::runtime_search)
        .def_readonly("runtime_build_CT", &SpaceTimeAStar::runtime_build_CT)
        .def_readonly("runtime_build_CAT", &SpaceTimeAStar::runtime_build_CAT)
        .def("findPath", py::overload_cast<
            std::map<int, Path>&, 
            std::map<int, std::vector<Constraint>>&,
            Instance&,
            const std::pair<int, int>&,
            const std::pair<int, int>&
            >(&SpaceTimeAStar::findPath), 
            "paths"_a, "constraints"_a, "instance"_a, "start_state"_a, "goal_state"_a)
        .def("findPath", py::overload_cast<
            std::map<int, Path>&, 
            std::map<int, std::vector<Constraint>>&,
            Instance3D&,
            const std::tuple<int, int, int>&,
            const std::tuple<int, int, int>&
            >(&SpaceTimeAStar::findPath), 
            "paths"_a, "constraints"_a, "instance"_a, "start_state"_a, "goal_state"_a);

}
