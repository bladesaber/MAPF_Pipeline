#ifndef MAPF_PIPELINE_EDGE_BOUNDARY_H
#define MAPF_PIPELINE_EDGE_BOUNDARY_H

#include "eigen3/Eigen/Core"
#include <g2o/core/base_unary_edge.h>

#include "vertex_XYZ.h"

namespace SmootherNameSpace{

class EdgeXYZ_Boundary:public g2o::BaseUnaryEdge<1, double, VertexXYZ>{
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

  EdgeXYZ_Boundary(
    double min_x, double min_y, double min_z,
    double max_x, double max_y, double max_z
  ):min_x(min_x), min_y(min_y), min_z(min_z), max_x(max_x), max_y(max_y), max_z(max_z){
    this->setMeasurement(0.0);
  }

  void computeError()
  {
    const VertexXYZ* conf1 = static_cast<const VertexXYZ*>(_vertices[0]);

    double cost = 0.0;

    cost += penaltyBoundFromBelow(conf1->x(), min_x);
    cost += penaltyBoundFromBelow(conf1->y(), min_y);
    cost += penaltyBoundFromBelow(conf1->z(), min_z);

    cost += penaltyBoundFromAbove(conf1->x(), max_x);
    cost += penaltyBoundFromAbove(conf1->y(), max_y);
    cost += penaltyBoundFromAbove(conf1->z(), max_z);

    _error[0] = cost;

  }

  static double lost_calc(VertexXYZ* conf1){
    return 0.0;
  }

private:
  double min_x, min_y, min_z;
  double max_x, max_y, max_z;

  inline double penaltyBoundFromBelow(const double& var, const double& a){
    if (var >= a){
      return 0.0;
    }
    else{
      return -var + a;
    }
  }

  inline double penaltyBoundFromAbove(const double& var, const double& a){
    if (var <= a){
      return 0.0;
    }
    else{
      return var - a;
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