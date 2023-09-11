//
// Created by admin123456 on 2023/9/4.
//

#ifndef MAPF_PIPELINE_SPRINGTARGETEDGE_MINVOLUME_H
#define MAPF_PIPELINE_SPRINGTARGETEDGE_MINVOLUME_H

#include "eigen3/Eigen/Core"
#include <g2o/core/base_binary_edge.h>

#include "tightSpringer/springVertex_plane.h"

namespace TightSpringNameSpace {
    class SpringTarEdge_MinVolume : public g2o::BaseBinaryEdge<1, double, SpringVertexPlane, SpringVertexPlane> {
    public:
        using typename g2o::BaseBinaryEdge<1, double, SpringVertexPlane, SpringVertexPlane>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, SpringVertexPlane, SpringVertexPlane>::computeError;

        virtual bool read(std::istream &is) {
            return true;
        }

        virtual bool write(std::ostream &os) const {
            return os.good();
        }

        ErrorVector &getError() {
            computeError();
            return _error;
        }

        SpringTarEdge_MinVolume(double scale, double kSpring = 1.0) : scale(scale), kSpring(kSpring) {
            this->setMeasurement(0.);
        }

        void computeError() {
            const SpringVertexPlane *vertex_min = static_cast<const SpringVertexPlane *>(_vertices[0]);
            const SpringVertexPlane *vertex_max = static_cast<const SpringVertexPlane *>(_vertices[1]);

            _error[0] = (vertex_max->x() - vertex_min->x()) * scale *
                        (vertex_max->y() - vertex_min->y()) * scale *
                        (vertex_max->z() - vertex_min->z()) * scale * kSpring;
        }

        virtual void linearizeOplus() override{
            // vertex_min is thought to be fixed
            const SpringVertexPlane *vertex_min = static_cast<const SpringVertexPlane *>(_vertices[0]);
            const SpringVertexPlane *vertex_max = static_cast<const SpringVertexPlane *>(_vertices[1]);
            _jacobianOplusXj(0, 0) = (vertex_max->y() - vertex_min->y()) * (vertex_max->z() - vertex_min->z()) * pow(scale, 3);
            _jacobianOplusXj(0, 1) = (vertex_max->x() - vertex_min->x()) * (vertex_max->z() - vertex_min->z()) * pow(scale, 3);
            _jacobianOplusXj(0, 2) = (vertex_max->x() - vertex_min->x()) * (vertex_max->y() - vertex_min->y()) * pow(scale, 3);

            // _jacobianOplusXj(0, 0) = 1;
            // _jacobianOplusXj(0, 1) = 1;
            // _jacobianOplusXj(0, 2) = 1;

            // std::cout << "Xi(Min) Row:" << _jacobianOplusXi.rows() << " Col:" << _jacobianOplusXi.cols()
            //      << " Value: " << _jacobianOplusXi << std::endl;
             std::cout << "Xi(Max) Row:" << _jacobianOplusXj.rows() << " Col:" << _jacobianOplusXj.cols() << " Value: "
                << _jacobianOplusXj << std::endl;
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, SpringVertexPlane, SpringVertexPlane>::_error;
        using g2o::BaseBinaryEdge<1, double, SpringVertexPlane, SpringVertexPlane>::_vertices;

    private:
        double scale;
        double kSpring;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}

#endif //MAPF_PIPELINE_SPRINGTARGETEDGE_MINVOLUME_H
