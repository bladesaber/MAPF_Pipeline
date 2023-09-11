//
// Created by admin123456 on 2023/8/29.
//

#ifndef MAPF_PIPELINE_SPRINGEDGE_POSEFIXED_H
#define MAPF_PIPELINE_SPRINGEDGE_POSEFIXED_H

#include "eigen3/Eigen/Core"
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>

#include "tightSpringer/springVertex_cell.h"
#include "tightSpringer/springVertex_structor.h"

using namespace std;

namespace TightSpringNameSpace {

    // connector(cell) <---> structor

    class SpringPoseEdge_PoseFixed : public g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexStructor> {
    public:
        using typename g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexStructor>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexStructor>::computeError;

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

        SpringPoseEdge_PoseFixed(
                double shapeX, double shapeY, double shapeZ, double kSpring = 1.0
        ) : shapeX(shapeX), shapeY(shapeY), shapeZ(shapeZ), kSpring(kSpring) {
            this->setMeasurement(0.);
        }

        void computeError() {
            const SpringVertexCell *vertex0 = static_cast<const SpringVertexCell *>(_vertices[0]);
            const SpringVertexStructor *vertex1 = static_cast<const SpringVertexStructor *>(_vertices[1]);

            assert(vertex1->xyzTag == "X" || vertex1->xyzTag == "Y" || vertex1->xyzTag == "Z");

            double dist;
            double x1, y1, z1;
            vertex1->compute_shapeX(shapeX, shapeY, shapeZ, x1);
            vertex1->compute_shapeY(shapeX, shapeY, shapeZ, y1);
            vertex1->compute_shapeZ(shapeX, shapeY, shapeZ, z1);

            dist = sqrt(
                    pow(x1 - vertex0->x(), 2) + pow(y1 - vertex0->y(), 2) + pow(z1 - vertex0->z(), 2)
            );

            _error[0] = dist * kSpring;
        }

        double lost_calc() {
            throw "Not Implementation";
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexStructor>::_error;
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexStructor>::_vertices;

    private:
        double shapeX, shapeY, shapeZ;
        double kSpring;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}

#endif //MAPF_PIPELINE_SPRINGEDGE_POSEFIXED_H
