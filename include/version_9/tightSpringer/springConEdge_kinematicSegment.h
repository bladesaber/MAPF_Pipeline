//
// Created by admin123456 on 2023/8/31.
//

#ifndef MAPF_PIPELINE_SPRINGEDGE_KINEMATICSEGMENT_H
#define MAPF_PIPELINE_SPRINGEDGE_KINEMATICSEGMENT_H

#include "eigen3/Eigen/Core"
#include <g2o/core/base_multi_edge.h>

#include "tightSpringer/springVertex_cell.h"

using namespace std;

namespace TightSpringNameSpace {

    // connector <---> cell <---> cell or cell <---> cell <---> cell

    class SpringConEdge_KinematicSegment : public g2o::BaseMultiEdge<1, double> {
    public:
        using typename g2o::BaseMultiEdge<1, double>::ErrorVector;
        using g2o::BaseMultiEdge<1, double>::computeError;

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

        SpringConEdge_KinematicSegment(double targetValue, double kSpring = 3.0) :
                targetValue(targetValue), kSpring(kSpring) {
            this->setMeasurement(0.);
            this->resize(3);
        }

        void computeError() {
            const SpringVertexCell *vertex0 = static_cast<const SpringVertexCell *>(_vertices[0]);
            const SpringVertexCell *vertex1 = static_cast<const SpringVertexCell *>(_vertices[1]);
            const SpringVertexCell *vertex2 = static_cast<const SpringVertexCell *>(_vertices[2]);

            double vecLength0 = (vertex1->position() - vertex0->position()).norm();
            double vecLength1 = (vertex2->position() - vertex1->position()).norm();
            double cos_theta =
                    (vertex1->position() - vertex0->position()).dot((vertex2->position() - vertex1->position())) /
                    (vecLength0 * vecLength1);

            _error[0] = std::abs(targetValue - cos_theta) * kSpring;
        }

        static double lost_calc() {
            throw "Not Implementation";
        }

    protected:
        using g2o::BaseMultiEdge<1, double>::_error;
        using g2o::BaseMultiEdge<1, double>::_vertices;

    private:
        double kSpring;
        double targetValue;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };
}

#endif //MAPF_PIPELINE_SPRINGEDGE_KINEMATICSEGMENT_H
