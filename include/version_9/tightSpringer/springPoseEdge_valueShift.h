//
// Created by admin123456 on 2023/8/29.
//

#ifndef MAPF_PIPELINE_SPRINGEDGE_XYZSHIFT_H
#define MAPF_PIPELINE_SPRINGEDGE_XYZSHIFT_H

#include "assert.h"
#include "eigen3/Eigen/Core"
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>

#include "tightSpringer/springVertex_cell.h"
#include "tightSpringer/springVertex_structor.h"
#include "tightSpringer/springVertex_plane.h"

namespace TightSpringNameSpace {

    // structor <---> plane
    // connector(cell) <---> structor

    template<typename T1, typename T2>
    class SpringPoseEdge_ValueShift : public g2o::BaseBinaryEdge<1, double, T1, T2>{
    public:
        using typename g2o::BaseBinaryEdge<1, double, T1, T2>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, T1, T2>::computeError;

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

        SpringPoseEdge_ValueShift(std::string xyzTag, double shiftValue, double kSpring = 1.0) :
            shiftValue(shiftValue), kSpring(kSpring) {
            assert(xyzTag == "X" || xyzTag == "Y" || xyzTag == "Z");
            this->xyzTag = xyzTag;

            this->setMeasurement(0.);
        }

        void computeError() {
            const T1 *vertex0 = static_cast<const T1 *>(_vertices[0]);
            const T2 *vertex1 = static_cast<const T2 *>(_vertices[1]);

            double dist;
            if (xyzTag == "X") {
                dist = vertex0->x() - vertex1->x();
            } else if (xyzTag == "Y") {
                dist = vertex0->y() - vertex1->y();
            } else {
                dist = vertex0->z() - vertex1->z();
            }

            _error[0] = std::abs(shiftValue - dist) * kSpring;
        }

        /*
        virtual void linearizeOplus() override{
            if (xyzTag == "X"){
                _jacobianOplusXj(0, 0) = 1.0;

            } else if (xyzTag == "Y"){
                _jacobianOplusXj(0, 1) = 1.0;

            } else{
                _jacobianOplusXj(0, 2) = 1.0;
            }
        }
        */

    protected:
        using g2o::BaseBinaryEdge<1, double, T1, T2>::_error;
        using g2o::BaseBinaryEdge<1, double, T1, T2>::_vertices;

    private:
        std::string xyzTag;
        double kSpring;
        double shiftValue;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}

#endif //MAPF_PIPELINE_SPRINGEDGE_XYZSHIFT_H
