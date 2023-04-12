#include <pybind11/pybind11.h>
#include "iostream"

#include "Aux_common.h"
#include "test.h"
#include "Aux_utils.h"
#include "Aux_dubins.h"
#include "Aux_continusAstar.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(mapf_pipeline, m) {
    m.doc() = "mapf pipeline"; // optional module docstring

    // DEBUG Fun
    // m.def("debugNumpy", &debugNumpy);

    py::enum_<SegmentType>(m, "SegmentType")
        .value("L_SEG", SegmentType::L_SEG)
        .value("S_SEG", SegmentType::S_SEG)
        .value("R_SEG", SegmentType::R_SEG)
        .export_values();
    py::enum_<DubinsPathType>(m, "DubinsPathType")
        .value("LSL", DubinsPathType::LSL)
        .value("LRL", DubinsPathType::LRL)
        .value("LSR", DubinsPathType::LSR)
        .value("RLR", DubinsPathType::RLR)
        .value("RSL", DubinsPathType::RSL)
        .value("RSR", DubinsPathType::RSR)
        .export_values();
    py::enum_<DubinsErrorCodes>(m, "DubinsErrorCodes")
        .value("EDUBOK", DubinsErrorCodes::EDUBOK)
        .value("EDUBCOCONFIGS", DubinsErrorCodes::EDUBCOCONFIGS)
        .value("EDUBPARAM", DubinsErrorCodes::EDUBPARAM)
        .value("EDUBBADRHO", DubinsErrorCodes::EDUBBADRHO)
        .value("EDUBNOPATH", DubinsErrorCodes::EDUBNOPATH)
        .export_values();
    py::class_<DubinsPath>(m, "DubinsPath")
        .def(py::init<>())
        .def_readonly("q0", &DubinsPath::q0)
        .def_readonly("q1", &DubinsPath::q1)
        .def_readonly("rho", &DubinsPath::rho)
        .def_readonly("param", &DubinsPath::param)
        .def_readonly("type", &DubinsPath::type)
        .def_readonly("lengths", &DubinsPath::lengths)
        .def_readonly("total_length", &DubinsPath::total_length)
        .def_readonly("start_center", &DubinsPath::start_center)
        .def_readonly("start_range", &DubinsPath::start_range)
        .def_readonly("final_center", &DubinsPath::final_center)
        .def_readonly("final_range", &DubinsPath::final_range)
        .def_readonly("line_sxy", &DubinsPath::line_sxy)
        .def_readonly("line_fxy", &DubinsPath::line_fxy);
    m.def("compute_dubins_path", &compute_dubins_path, "result"_a, "xyz_s"_a, "xyz_t"_a, "rho"_a, "pathType"_a);
    m.def("compute_dubins_info", &compute_dubins_info, "path"_a);
    m.def("sample_dubins_path", &sample_dubins_path, "path"_a, "sample_size"_a);
    
    m.def("mod2pi", &mod2pi, "theta"_a);
    m.def("mod2singlePi", &mod2singlePi, "theta"_a);
    m.def("polar3D_to_vec3D", &polar3D_to_vec3D, "alpha"_a, "beta"_a, "length"_a);
    m.def("vec3D_to_polar3D", &polar3D_to_vec3D, "vec_x"_a, "vec_y"_a, "vec_z"_a);

    py::class_<HybridAstarNode>(m, "HybridAstarNode")
        .def(py::init<double, double, double, double, double, HybridAstarNode*, bool>(), "x"_a, "y"_a, "z"_a, "alpha"_a, "beta"_a, "parent"_a, "in_openlist"_a)
        .def("setRoundCoodr", py::overload_cast<int, int, int, int, int>(&HybridAstarNode::setRoundCoodr), "x"_a, "y"_a, "z"_a, "alpha"_a, "beta"_a)
        .def("setRoundCoodr", py::overload_cast<std::tuple<int, int, int, int, int>>(&HybridAstarNode::setRoundCoodr), "pos"_a)
        .def("getFVal", &HybridAstarNode::getFVal)
        .def("getCoodr", &HybridAstarNode::getCoodr)
        .def("getRoundCoodr", &HybridAstarNode::getRoundCoodr)
        .def("copy", &HybridAstarNode::copy, "rhs"_a)
        .def("equal", &HybridAstarNode::equal, "rhs"_a)
        .def_readonly("x", &HybridAstarNode::x)
        .def_readonly("y", &HybridAstarNode::y)
        .def_readonly("z", &HybridAstarNode::z)
        .def_readonly("alpha", &HybridAstarNode::alpha)
        .def_readonly("beta", &HybridAstarNode::beta)
        .def_readonly("x_round", &HybridAstarNode::x_round)
        .def_readonly("y_round", &HybridAstarNode::y_round)
        .def_readonly("z_round", &HybridAstarNode::z_round)
        .def_readonly("alpha_round", &HybridAstarNode::alpha_round)
        .def_readonly("beta_round", &HybridAstarNode::beta_round)
        .def_readonly("hashTag", &HybridAstarNode::hashTag)
        .def_readonly("parent", &HybridAstarNode::parent)
        .def_readwrite("g_val", &HybridAstarNode::g_val)
        .def_readwrite("h_val", &HybridAstarNode::h_val)
        .def_readwrite("dubins_solutions", &HybridAstarNode::dubins_solutions)
        .def_readwrite("invert_yz", &HybridAstarNode::invert_yz)
        .def_readwrite("dubinsPath3D", &HybridAstarNode::dubinsPath3D)
        .def_readwrite("dubinsLength3D", &HybridAstarNode::dubinsLength3D);

    py::class_<HybridAstar>(m, "HybridAstar")
        .def(py::init<>())
        .def("pushNode", &HybridAstar::pushNode, "node"_a)
        .def("popNode", &HybridAstar::popNode)
        .def("is_openList_empty", &HybridAstar::is_openList_empty)
        .def("release", &HybridAstar::release);
}
