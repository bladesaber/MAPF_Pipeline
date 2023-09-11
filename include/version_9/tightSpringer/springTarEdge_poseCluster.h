//
// Created by admin123456 on 2023/9/7.
//

#ifndef MAPF_PIPELINE_SPRINGTAREDGE_POSECLUSTER_H
#define MAPF_PIPELINE_SPRINGTAREDGE_POSECLUSTER_H

#include "eigen3/Eigen/Core"
#include <g2o/core/base_unary_edge.h>

//#include "tightSpringer/springVertex_structor.h"
//#include "tightSpringer/springVertex_cell.h"

using namespace std;

namespace TightSpringNameSpace {

    template<typename T1>
    class SpringTarEdge_PoseCluster : public g2o::BaseUnaryEdge<1, double, T1> {
    public:
        using typename g2o::BaseUnaryEdge<1, double, T1>::ErrorVector;
        using g2o::BaseUnaryEdge<1, double, T1>::computeError;

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

        SpringTarEdge_PoseCluster(double scale, double kSpring = 1.0) : scale(scale), kSpring(kSpring) {
            this->setMeasurement(0.);
        }

        void computeError() {
            const T1 *vertex = static_cast<const T1 *>(_vertices[0]);
            _error[0] = pow(vertex->x() * scale, 2) + pow(vertex->y() * scale, 2) + pow(vertex->z() * scale, 2);
        }

        double lost_calc() {
            throw "Not Implementation";
        }

    protected:
        using g2o::BaseUnaryEdge<1, double, T1>::_error;
        using g2o::BaseUnaryEdge<1, double, T1>::_vertices;

    private:
        double scale;
        double kSpring;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}

#endif //MAPF_PIPELINE_SPRINGTAREDGE_POSECLUSTER_H
