//
// Created by admin123456 on 2023/8/31.
//

#ifndef MAPF_PIPELINE_SPRINGVERTEX_PLANE_H
#define MAPF_PIPELINE_SPRINGVERTEX_PLANE_H

#include "eigen3/Eigen/Core"
#include "g2o/config.h"
#include <g2o/core/base_vertex.h>
#include <g2o/core/hyper_graph_action.h>
#include <g2o/stuff/misc.h>
#include "math.h"

using namespace std;

namespace TightSpringNameSpace {
    class SpringVertexPlane : public g2o::BaseVertex<3, Eigen::Vector3d> {
    private:
        std::string name;

    public:
        SpringVertexPlane(bool fixed = false) {
            setToOriginImpl();
            setFixed(fixed);
        }

        SpringVertexPlane(double x, double y, double z, bool fixed = false) {
            _estimate.coeffRef(0) = x;
            _estimate.coeffRef(1) = y;
            _estimate.coeffRef(2) = z;
            setFixed(fixed);
        }

        ~SpringVertexPlane() {}

        virtual void setToOriginImpl() override {
            _estimate.setZero();
        }

        inline double &x() {
            return _estimate.coeffRef(0);
        }

        inline const double &x() const {
            return _estimate.coeffRef(0);
        }

        inline double &y() {
            return _estimate.coeffRef(1);
        }

        inline const double &y() const {
            return _estimate.coeffRef(1);
        }

        inline double &z() {
            return _estimate.coeffRef(2);
        }

        inline const double &z() const {
            return _estimate.coeffRef(2);
        }

        virtual void oplusImpl(const double *update) override {
            Eigen::Vector3d update_vec(update);
            double length = update_vec.norm();
            if (length > 0.1) {
                _estimate += update_vec / length * max(min(length, 1.0), 0.1);
            }

//            _estimate.coeffRef(0) += update[0];
//            _estimate.coeffRef(1) += update[1];
//            _estimate.coeffRef(2) += update[2];

            std::printf("Name: %s (%f, %f, %f) %f\n", name.c_str(), update[0], update[1], update[2], -1.0);
        }

        virtual bool read(std::istream &is) override {
            is >> _estimate.coeffRef(0) >> _estimate.coeffRef(1) >> _estimate.coeffRef(2);
            return true;
        }

        virtual bool write(std::ostream &os) const override {
            os << _estimate.coeffRef(0) << " " << _estimate.coeffRef(1) << " " << _estimate.coeffRef(2);
            return os.good();
        }

        void setName(std::string tag) {
            this->name = tag;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}

#endif //MAPF_PIPELINE_SPRINGVERTEX_PLANE_H
