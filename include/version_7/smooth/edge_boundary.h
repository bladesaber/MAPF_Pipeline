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
    double max_x, double max_y, double max_z,
    double radius
  ):min_x(min_x), min_y(min_y), min_z(min_z), max_x(max_x), max_y(max_y), max_z(max_z), radius(radius)
  {
    this->setMeasurement(0.0);
  }

  void computeError()
  {
    const VertexXYZ* conf1 = static_cast<const VertexXYZ*>(_vertices[0]);

    double cost = 0.0;

    cost += penaltyBoundFromBelow(conf1->x(), min_x + radius);
    cost += penaltyBoundFromBelow(conf1->y(), min_y + radius);
    cost += penaltyBoundFromBelow(conf1->z(), min_z + radius);

    cost += penaltyBoundFromAbove(conf1->x(), max_x - radius);
    cost += penaltyBoundFromAbove(conf1->y(), max_y - radius);
    cost += penaltyBoundFromAbove(conf1->z(), max_z - radius);

    _error[0] = cost;

  }

  static double lost_calc(
    VertexXYZ* conf1, double radius,
    double min_x, double min_y, double min_z,
    double max_x, double max_y, double max_z
  ){
    double cost_xBelow = penaltyBoundFromBelow(conf1->x(), min_x + radius);
    double cost_yBelow = penaltyBoundFromBelow(conf1->y(), min_y + radius);
    double cost_zBelow = penaltyBoundFromBelow(conf1->z(), min_z + radius);

    double cost_xUp = penaltyBoundFromAbove(conf1->x(), max_x - radius);
    double cost_yUp = penaltyBoundFromAbove(conf1->y(), max_y - radius);
    double cost_zUp = penaltyBoundFromAbove(conf1->z(), max_z - radius);

    std::cout << "cost_xBelow:" << cost_xBelow;
    std::cout << " cost_yBelow:" << cost_yBelow;
    std::cout << " cost_zBelow:" << cost_zBelow;
    std::cout << " cost_xUp:" << cost_xUp;
    std::cout << " cost_yUp:" << cost_yUp;
    std::cout << " cost_zUp:" << cost_zUp << std::endl;

    return cost_xBelow + cost_yBelow + cost_zBelow + cost_xUp + cost_yUp + cost_zUp;
  }

private:
  double min_x, min_y, min_z;
  double max_x, max_y, max_z;
  double radius;

  static inline double penaltyBoundFromBelow(const double& var, const double& a){
    if (var >= a){
      return 0.0;
    }
    else{
      return -var + a;
    }
  }

  static inline double penaltyBoundFromAbove(const double& var, const double& a){
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