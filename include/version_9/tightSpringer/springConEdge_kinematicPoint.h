//
// Created by admin123456 on 2023/8/31.
//

#ifndef MAPF_PIPELINE_SPRINGEDGE_KINEMATICSPOINT_H
#define MAPF_PIPELINE_SPRINGEDGE_KINEMATICSPOINT_H

#include "eigen3/Eigen/Core"
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>

#include "tightSpringer/springVertex_cell.h"
#include "tightSpringer/springVertex_structor.h"

using namespace std;

/*
namespace TightSpringNameSpace {

    // connector <---> cell or cell <---> connector

    class SpringConEdge_KinematicPoint : public g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell> {
    public:
        using typename g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell>::computeError;

        ErrorVector &getError() {
            computeError();
            return _error;
        }

        virtual bool read(std::istream &is) {
            return true;
        }

        virtual bool write(std::ostream &os) const {
            return os.good();
        }

        SpringConEdge_KinematicPoint(
                Eigen::Vector3d orientation, double targetValue, double kSpring = 10.0
        ) : orientation(orientation), targetValue(targetValue), kSpring(kSpring) {
            this->setMeasurement(0.);
        }

        void computeError() {
            const SpringVertexCell *vertex0 = static_cast<const SpringVertexCell *>(_vertices[0]);
            const SpringVertexCell *vertex1 = static_cast<const SpringVertexCell *>(_vertices[1]);

            double length0 = (vertex1->position() - vertex0->position()).norm();
            double length1 = orientation.norm();
            double cos_theta = (vertex1->position() - vertex0->position()).dot(orientation) / (length0 * length1);

            _error[0] = std::abs(targetValue - cos_theta) * kSpring;
        }

        double lost_calc() {
            throw "Not Implementation";
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell>::_error;
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell>::_vertices;

    private:
        Eigen::Vector3d orientation;
        double kSpring;
        double targetValue;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };
}
*/
#endif //MAPF_PIPELINE_SPRINGEDGE_KINEMATICSPOINT_H
