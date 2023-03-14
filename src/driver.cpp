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
#include "cbsNode.h"
#include "conflict.h"
#include "cbs.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(mapf_pipeline, m) {
    m.doc() = "mapf pipeline"; // optional module docstring
    m.def("add", &add, "A function that adds two numbers", "i"_a, "j"_a);
}
