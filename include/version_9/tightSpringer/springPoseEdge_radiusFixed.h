//
// Created by admin123456 on 2023/8/29.
//

#ifndef MAPF_PIPELINE_SPRINGEDGE_RADIUSSHIFT_H
#define MAPF_PIPELINE_SPRINGEDGE_RADIUSSHIFT_H

#include "eigen3/Eigen/Core"
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>

#include "tightSpringer/springVertex_cell.h"
#include "tightSpringer/springVertex_structor.h"

namespace TightSpringNameSpace {

    // connector(cell) <---> structor

    template<typename T1, typename T2>
    class SpringPoseEdge_RadiusFixed : public g2o::BaseBinaryEdge<1, double, T1, T2> {
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

        SpringPoseEdge_RadiusFixed(std::string xyzTag, double radius, double kSpring = 1.0) :
                radius(radius), kSpring(kSpring) {
            assert(xyzTag == "X" || xyzTag == "Y" || xyzTag == "Z");
            this->xyzTag = xyzTag;

            this->setMeasurement(0.);
        }

        void computeError() {
            const T1 *vertex0 = static_cast<const T1 *>(_vertices[0]);
            const T2 *vertex1 = static_cast<const T2 *>(_vertices[1]);

            double dist;
            if (xyzTag == "X") {
                dist = std::sqrt(std::pow(vertex0->y() - vertex1->y(), 2) + std::pow(vertex0->z() - vertex1->z(), 2));
            } else if (xyzTag == "Y"){
                dist = std::sqrt(std::pow(vertex0->x() - vertex1->x(), 2) + std::pow(vertex0->z() - vertex1->z(), 2));
            } else {
                dist = std::sqrt(std::pow(vertex0->x() - vertex1->x(), 2) + std::pow(vertex0->y() - vertex1->y(), 2));
            }

            _error[0] = std::abs(radius - dist) * kSpring;
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, T1, T2>::_error;
        using g2o::BaseBinaryEdge<1, double, T1, T2>::_vertices;

    private:
        std::string xyzTag;
        double kSpring;
        double radius;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };

}

#endif //MAPF_PIPELINE_SPRINGEDGE_RADIUSSHIFT_H
