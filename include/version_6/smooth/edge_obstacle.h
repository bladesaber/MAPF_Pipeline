#ifndef MAPF_PIPELINE_EDGE_OBSTACLE_H
#define MAPF_PIPELINE_EDGE_OBSTACLE_H

#include "eigen3/Eigen/Core"
#include <g2o/core/base_unary_edge.h>

// #include "vertex_SE3.h"
#include "vertex_XYZ.h"

namespace SmootherNameSpace{

/*
// errorDim, errorType, Vertex1Type
class EdgeObstacle:public g2o::BaseUnaryEdge<1, double, VertexSE3>{
public:
  
  using typename g2o::BaseUnaryEdge<1, double, VertexSE3>::ErrorVector;
  using g2o::BaseUnaryEdge<1, double, VertexSE3>::computeError;

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

  EdgeObstacle(Eigen::Vector3d obstacle, double min_obstacle_dist, double penalty_epsilon):
    obstacle(obstacle), min_obstacle_dist(min_obstacle_dist), penalty_epsilon(penalty_epsilon) 
  {
    this->setMeasurement(0.0);
  }

  void computeError()
  {
    const VertexSE3* bandpt = static_cast<const VertexSE3*>(_vertices[0]);

    double dist = (bandpt->position() - obstacle).norm();

    _error[0] = penaltyBoundFromBelow(dist, min_obstacle_dist, penalty_epsilon);

    // if (obstacle_cost_exponent != 1.0 && min_obstacle_dist > 0.0)
    // {
    //   // Optional non-linear cost. Note the max cost (before weighting) is
    //   // the same as the straight line version and that all other costs are
    //   // below the straight line (for positive exponent), so it may be
    //   // necessary to increase weight_obstacle and/or the inflation_weight
    //   // when using larger exponents.
    //   _error[0] = min_obstacle_dist * std::pow(_error[0] / min_obstacle_dist, obstacle_cost_exponent);
    // }

  }

private:
  double min_obstacle_dist;
  double penalty_epsilon;
  Eigen::Vector3d obstacle;

  inline double penaltyBoundFromBelow(const double& var, const double& a,const double& epsilon)
  {
    if (var >= a + epsilon)
    {
      return 0.;
    }
    else{
      return (-var + (a + epsilon));
    }
  }

protected:
    
  using g2o::BaseUnaryEdge<1, double, VertexSE3>::_error;
  using g2o::BaseUnaryEdge<1, double, VertexSE3>::_vertices;
    
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  

};
*/

class EdgeXYZ_Obstacle:public g2o::BaseUnaryEdge<1, double, VertexXYZ>{
public:
  
  using typename g2o::BaseUnaryEdge<1, double, VertexXYZ>::ErrorVector;
  using g2o::BaseUnaryEdge<1, double, VertexXYZ>::computeError;

  ErrorVector& getError(){
    computeError();
    return _error;
  }
  
  virtual bool read(std::istream& is){
    return true;
  }
 
  virtual bool write(std::ostream& os) const{
    return os.good();
  }

  EdgeXYZ_Obstacle(Eigen::Vector3d obstacle, double min_obstacle_dist):
    obstacle(obstacle), min_obstacle_dist(min_obstacle_dist) {
    this->setMeasurement(0.0);
  }

  void computeError()
  {
    const VertexXYZ* conf1 = static_cast<const VertexXYZ*>(_vertices[0]);

    double dist = (conf1->position() - obstacle).norm();

    _error[0] = penaltyBoundFromBelow(dist, min_obstacle_dist);

    // if (obstacle_cost_exponent != 1.0 && min_obstacle_dist > 0.0)
    // {
    //   // Optional non-linear cost. Note the max cost (before weighting) is
    //   // the same as the straight line version and that all other costs are
    //   // below the straight line (for positive exponent), so it may be
    //   // necessary to increase weight_obstacle and/or the inflation_weight
    //   // when using larger exponents.
    //   _error[0] = min_obstacle_dist * std::pow(_error[0] / min_obstacle_dist, obstacle_cost_exponent);
    // }
  }

  static double lost_calc(VertexXYZ* conf1, Eigen::Vector3d obstacle, double min_obstacle_dist){
    double dist = (conf1->position() - obstacle).norm();
    double loss;
    if (dist >= min_obstacle_dist){
      loss = 0.;
    }
    else{
      loss = -dist + min_obstacle_dist;
    }

    return loss;
  }

private:
  double min_obstacle_dist;
  Eigen::Vector3d obstacle;

  inline double penaltyBoundFromBelow(const double& var, const double& a){
    if (var >= a){
      return 0.0;
    }
    else{
      return -var + a;
    }
  }

protected:
  using g2o::BaseUnaryEdge<1, double, VertexXYZ>::_error;
  using g2o::BaseUnaryEdge<1, double, VertexXYZ>::_vertices;
    
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  

};

}

#endif