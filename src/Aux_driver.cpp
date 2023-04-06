#include <pybind11/pybind11.h>
#include "iostream"

#include "test.h"
#include "Aux_common.h"
#include "Aux_dubins.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(mapf_pipeline, m) {
    m.doc() = "mapf pipeline"; // optional module docstring

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
        .def_readonly("type", &DubinsPath::type);
    m.def("compute_dubins_path", &compute_dubins_path, "result"_a, "xyz_s"_a, "xyz_t"_a, "rho"_a, "pathType"_a);

}
