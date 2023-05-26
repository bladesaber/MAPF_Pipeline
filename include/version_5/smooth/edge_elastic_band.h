#ifndef MAPF_PIPELINE_EDGE_SHORTEST_BAND_H
#define MAPF_PIPELINE_EDGE_SHORTEST_BAND_H

#include <g2o/core/base_binary_edge.h>
#include "vertex_pose.h"

namespace SmootherNameSpace{

class EdgeElasticBand: public g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>{
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

  EdgeElasticBand() {
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

}

#endif