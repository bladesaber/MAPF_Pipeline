#ifndef MAPF_PIPELINE_EDGE_PIPE_CONFLICT_H
#define MAPF_PIPELINE_EDGE_PIPE_CONFLICT_H

#include <g2o/core/base_binary_edge.h>
#include "vertex_pose.h"

namespace SmootherNameSpace{

class EdgePipeConflict: public g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>{
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

  EdgePipeConflict(double bound, double bound_epsilon):bound(bound), bound_epsilon(bound_epsilon){
    this->setMeasurement(0.);
  }

  void computeError(){
    const VertexSE3 *pose1 = static_cast<const VertexSE3*>(_vertices[0]);
    const VertexSE3 *pose2 = static_cast<const VertexSE3*>(_vertices[1]);
    double dist = (pose2->position() - pose1->position()).norm();

    _error[0] = penaltyBoundFromBelow(dist, bound, bound_epsilon);
  }

private:
  double bound;
  double bound_epsilon;

  inline double penaltyBoundFromBelow(const double& var, const double& a,const double& epsilon)
  {
    if (var >= a + epsilon)
    {
      return 0.;
    }
    else{
      return -var + (a + epsilon);
    }
  }

protected:
  using g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>::_error;
  using g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>::_vertices;
      
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   

};

}

#endif