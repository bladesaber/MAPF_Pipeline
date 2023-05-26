#ifndef MAPF_PIPELINE_EDGE_CURVATURE_H
#define MAPF_PIPELINE_EDGE_CURVATURE_H

#include "eigen3/Eigen/Core"
#include <g2o/core/base_binary_edge.h>
#include "vertex_pose.h"

namespace SmootherNameSpace{

class EdgeCurvature: public g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>{
public:
  using typename g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>::ErrorVector;
  using g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>::computeError;

  ErrorVector& getError()
  {
    computeError();
    return _error;
  }
  
  virtual bool read(std::istream& is)
  {
    return true;
  }
 
  virtual bool write(std::ostream& os) const
  {
    return os.good();
  }

  EdgeCurvature(double min_turning_radius):min_turning_radius(min_turning_radius){
    this->setMeasurement(0.);
  }

  void computeError()
  {
    const VertexSE3* conf1 = static_cast<const VertexSE3*>(_vertices[0]);
    const VertexSE3* conf2 = static_cast<const VertexSE3*>(_vertices[1]);

    Eigen::Vector3d deltaS = conf2->position() - conf1->position();

    Eigen::Vector3d vec1 = conf1->pose().orientation2UnitVec();
    Eigen::Vector3d vec2 = conf2->pose().orientation2UnitVec();
    double angle_diff = std::acos( vec1.dot(vec2) );

    _error[0] = penaltyBoundFromBelow(deltaS.norm() / angle_diff, min_turning_radius, 0.0);     
  }

protected:
  using g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>::_error;
  using g2o::BaseBinaryEdge<1, double, VertexSE3, VertexSE3>::_vertices;

private:
    double min_turning_radius;

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

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

}

#endif