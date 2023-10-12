#ifndef MAPF_PIPELINE_EDGE_SHORTEST_BAND_H
#define MAPF_PIPELINE_EDGE_SHORTEST_BAND_H

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>

// #include "vertex_SE3.h"
#include "vertex_XYZ.h"

namespace SmootherNameSpace {

/*
class EdgeSE3_ElasticBand: public g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>{
public:
  using typename g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>::ErrorVector;
  using g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>::computeError;

  virtual bool read(std::istream& is)
  {
    return true;
  }
   
  virtual bool write(std::ostream& os) const
  {
    return os.good();
  }

  ErrorVector& getError()
  {
    computeError();
    return _error;
  }

  EdgeSE3_ElasticBand() {
    this->setMeasurement(0.);
  }

  void computeError() {
    const VertexSE3* pose1 = static_cast<const VertexSE3*>(_vertices[0]);
    const VertexSE3* pose2 = static_cast<const VertexSE3*>(_vertices[1]);

    _error[0] = (pose2->position() - pose1->position()).squaredNorm();
  }

protected:
  using g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>::_error;
  using g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>::_vertices;
      
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   

};
*/

    class EdgeXYZ_ElasticBand : public g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZ> {
    public:
        using typename g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZ>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZ>::computeError;

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

        EdgeXYZ_ElasticBand(double kSpring = 1.0) : kSpring(kSpring){
            this->setMeasurement(0.);
        }

        void computeError() {
            const VertexXYZ *pose1 = static_cast<const VertexXYZ *>(_vertices[0]);
            const VertexXYZ *pose2 = static_cast<const VertexXYZ *>(_vertices[1]);

//            _error[0] = (pose2->position() - pose1->position()).squaredNorm() * kSpring;

             _error[0] = (pose2->position() - pose1->position()).norm() * kSpring;

            // _error[0] = std::pow(length - (pose2->position() - pose1->position()).norm(), 2.0) * kSpring;
        }

        static double lost_calc(VertexXYZ *pose1, VertexXYZ *pose2, double kSpring = 1.0) {
            double loss;

//            loss = (pose2->position() - pose1->position()).squaredNorm();

             loss = (pose2->position() - pose1->position()).norm() * kSpring;

            // loss std::abs(1.0 - (pose2->position() - pose1->position()).norm()) * kSpring;

            std::cout << "  ElasticBand EdgeLoss:" << loss << std::endl;

            return loss;
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZ>::_error;
        using g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZ>::_vertices;

    private:
        double kSpring;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };

}

#endif