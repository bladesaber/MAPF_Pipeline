//
// Created by admin123456 on 2023/8/31.
//

#ifndef MAPF_PIPELINE_SPRINGEDGE_ELASTICBAND_H
#define MAPF_PIPELINE_SPRINGEDGE_ELASTICBAND_H

#include "eigen3/Eigen/Core"
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>

#include "tightSpringer/springVertex_cell.h"

/*
namespace TightSpringNameSpace {

    class SpringConEdge_ElasticBand : public g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell> {
    public:
        using typename g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell>::computeError;

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

        SpringConEdge_ElasticBand(double kSpring = 1.0) : kSpring(kSpring) {
            this->setMeasurement(0.);
        }

        void computeError() {
            const SpringVertexCell *vertex0 = static_cast<const SpringVertexCell *>(_vertices[0]);
            const SpringVertexCell *vertex1 = static_cast<const SpringVertexCell *>(_vertices[1]);

            _error[0] = (vertex0->position() - vertex1->position()).norm() * kSpring;
        }

        double lost_calc() {
            throw "Not Implementation";
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell>::_error;
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell>::_vertices;

    private:
        double kSpring;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}
*/
#endif //MAPF_PIPELINE_SPRINGEDGE_ELASTICBAND_H
