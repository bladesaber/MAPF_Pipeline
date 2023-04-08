#include <pybind11/pybind11.h>
#include "iostream"

#include "test.h"
#include "Aux_common.h"
#include "Aux_dubins.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(mapf_pipeline, m) {
    m.doc() = "mapf pipeline"; // optional module docstring

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

}
