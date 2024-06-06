//
// Created by admin123456 on 2024/5/30.
//

#include "common.h"
#include "pybind11/stl_bind.h"

#include "pcl_utils.h"
#include "pybind_utils.h"

namespace py = pybind11;
using namespace pybind11::literals;

//PYBIND11_MAKE_OPAQUE(std::vector<double>);

PYBIND11_MODULE(mapf_pipeline, m) {
    m.doc() = "mapf pipeline"; // optional module docstring

    // ------ Function Define
    // py::bind_vector<std::vector<double>>(m, "VectorDouble");
    // m.def("test", &test);

    // ------ Class Define
    py::class_<PybindUtils::Matrix2D>(m, "Matrix2D", py::buffer_protocol())
            .def_buffer([](PybindUtils::Matrix2D &m) -> py::buffer_info {
                return py::buffer_info(
                        m.data(),
                        sizeof(float),
                        py::format_descriptor<double>::format(),
                        2,
                        {m.rows(), m.cols()},
                        {sizeof(float) * m.cols(), sizeof(float)}
                );
            });

    py::class_<pcl::PointXYZ>(m, "PointXYZ_pcl")
            .def_readonly("x", &pcl::PointXYZ::x)
            .def_readonly("y", &pcl::PointXYZ::y)
            .def_readonly("z", &pcl::PointXYZ::z);

    py::class_<PclUtils::KDTree>(m, "KDtree_pcd")
            .def(py::init<>())
            .def_readonly("result_idxs_1D", &PclUtils::KDTree::result_idxs_1D)
            .def_readonly("result_distance_1D", &PclUtils::KDTree::result_distance_1D)
            .def("create_tree", &PclUtils::KDTree::create_tree, "data"_a)
            .def("nearestKSearch", &PclUtils::KDTree::nearestKSearch, "x"_a, "y"_a, "z"_a, "k"_a)
            .def("radiusSearch", &PclUtils::KDTree::nearestKSearch, "x"_a, "y"_a, "z"_a, "radius"_a)
            .def("update_data", &PclUtils::KDTree::update_data, "data"_a)
            .def("get_point_from_data", &PclUtils::KDTree::get_point_from_data, "idx"_a);

}