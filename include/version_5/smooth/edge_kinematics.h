#ifndef MAPF_PIPELINE_EDGE_CURVATURE_H
#define MAPF_PIPELINE_EDGE_CURVATURE_H

#include "eigen3/Eigen/Core"
#include <g2o/core/base_binary_edge.h>
#include "vertex_pose.h"

namespace SmootherNameSpace{

class EdgeKinematics: public g2o::BaseBinaryEdge<2, double, VertexSE3, VertexSE3>{
public:
  using typename g2o::BaseBinaryEdge<2, double, VertexSE3, VertexSE3>::ErrorVector;
  using g2o::BaseBinaryEdge<2, double, VertexSE3, VertexSE3>::computeError;

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

  EdgeKinematics(double min_turning_radius):min_turning_radius(min_turning_radius){
    this->setMeasurement(0.);
  }

  void computeError()
  {
    const VertexSE3* conf1 = static_cast<const VertexSE3*>(_vertices[0]);
    const VertexSE3* conf2 = static_cast<const VertexSE3*>(_vertices[1]);

    Eigen::Vector3d deltaS = conf2->position() - conf1->position();
    Eigen::Vector3d deltaS_vec = deltaS / deltaS.norm();

    Eigen::Vector3d vec1 = Eigen::Vector3d(
      cos(conf1->theta()) * cos(conf1->alpha()), 
      cos(conf1->theta()) * sin(conf1->alpha()),
      sin(conf1->theta())
    );
    Eigen::Vector3d vec2 = Eigen::Vector3d(
      cos(conf2->theta()) * cos(conf2->alpha()), 
      cos(conf2->theta()) * sin(conf2->alpha()),
      sin(conf2->theta())
    );

    if (deltaS_vec.dot(vec1) == 0.0){
      _error[0] = 0.0;
    }else if (deltaS_vec.dot(vec2) == 0.0){
      _error[0] = 0.0;
    }else if (vec1.dot(vec2) == 0.0){
      _error[0] = 0.0;
    }else{
      Eigen::Vector3d norm_vec = vec1.cross(deltaS_vec);
      _error[0] = norm_vec.dot(vec2);
    }
    
    _error[1] = ( (1.0 - vec1.dot(deltaS_vec)) + (1.0 - deltaS_vec.dot(vec2)) ) * 0.5;
    // _error[2] = fabs(vec1.dot(deltaS_vec) - deltaS_vec.dot(vec2));

    // TODO curvature is not a good choice
    // double curvature = deltaS.norm() / (std::acos( vec1.dot(vec2) ) + 0.00001);
    // _error[1] = penaltyBoundFromUp(curvature, 1.0 / 3.0);
  }

protected:
  using g2o::BaseBinaryEdge<2, double, VertexSE3, VertexSE3>::_error;
  using g2o::BaseBinaryEdge<2, double, VertexSE3, VertexSE3>::_vertices;

private:
    double min_turning_radius;

    inline double penaltyBoundFromUp(const double& var, const double& a)
    {
      if (var <= a)
      {
        return 0.;
      }
      else{
        return var - a;
      }
    }

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

}

#endif