//
// Created by admin123456 on 2023/9/7.
//

#ifndef MAPF_PIPELINE_SPRINGEDGE_MINAXES_H
#define MAPF_PIPELINE_SPRINGEDGE_MINAXES_H

#include "assert.h"
#include "eigen3/Eigen/Core"
#include <g2o/core/base_binary_edge.h>

#include "tightSpringer/springVertex_plane.h"

namespace TightSpringNameSpace {
    class SpringTarEdge_MinAxes : public g2o::BaseBinaryEdge<1, double, SpringVertexPlane, SpringVertexPlane> {
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

        SpringTarEdge_MinAxes(std::string xyzTag, double scale, double kSpring = 1.0) :scale(scale), kSpring(kSpring) {
            assert(xyzTag == "X" || xyzTag == "Y" || xyzTag == "Z");
            this->xyzTag = xyzTag;

            this->setMeasurement(0.);
        }

        void computeError() {
            const SpringVertexPlane *vertex_min = static_cast<const SpringVertexPlane *>(_vertices[0]);
            const SpringVertexPlane *vertex_max = static_cast<const SpringVertexPlane *>(_vertices[1]);

            if (xyzTag == "X"){
                _error[0] = (vertex_max->x() - vertex_min->x()) * scale * kSpring;
            } else if (xyzTag == "Y"){
                _error[0] = (vertex_max->y() - vertex_min->y()) * scale * kSpring;
            } else{
                _error[0] = (vertex_max->z() - vertex_min->z()) * scale * kSpring;
            }
        }

        virtual void linearizeOplus() override{
            // vertex_min is thought to be fixed
            if (xyzTag == "X"){
                _jacobianOplusXj(0, 0) = 1.0 * scale;

            } else if (xyzTag == "Y"){
                _jacobianOplusXj(0, 1) = 1.0 * scale;

            } else{
                _jacobianOplusXj(0, 2) = 1.0 * scale;
            }
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, SpringVertexPlane, SpringVertexPlane>::_error;
        using g2o::BaseBinaryEdge<1, double, SpringVertexPlane, SpringVertexPlane>::_vertices;

    private:
        std::string xyzTag;
        double scale;
        double kSpring;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}

#endif //MAPF_PIPELINE_SPRINGEDGE_MINAXES_H
