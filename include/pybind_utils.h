//
// Created by admin123456 on 2024/5/30.
//

#ifndef MAPF_PIPELINE_PYBIND_UTILS_H
#define MAPF_PIPELINE_PYBIND_UTILS_H

#include "common.h"

namespace PybindUtils {
    class Matrix2D {
    public:
        Matrix2D(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) { m_data = new double[rows * cols]; }

        ~Matrix2D() { delete m_data; }

        double *data() { return m_data; }

        size_t rows() const { return m_rows; }

        size_t cols() const { return m_cols; }

    private:
        size_t m_rows, m_cols;
        double *m_data;
    };

}

#endif //MAPF_PIPELINE_PYBIND_UTILS_H
