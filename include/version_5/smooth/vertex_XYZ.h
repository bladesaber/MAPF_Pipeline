#ifndef MAPF_PIPELINE_VERTEX_XYZ_H
#define MAPF_PIPELINE_VERTEX_XYZ_H

#include "eigen3/Eigen/Core"
#include "g2o/config.h"
#include <g2o/core/base_vertex.h>
#include <g2o/core/hyper_graph_action.h>
#include <g2o/stuff/misc.h>
#include "math.h"

namespace SmootherNameSpace{

class VertexXYZ: public g2o::BaseVertex<3, Eigen::Vector3d>{
public:
  VertexXYZ(bool fixed = false)
  {
    setToOriginImpl();
    setFixed(fixed);
  }

  VertexXYZ(const Eigen::Vector3d& pose_xyz, bool fixed = false)
  {
    _estimate = pose_xyz;
    setFixed(fixed);
  }

  VertexXYZ(const Eigen::Ref<const Eigen::Vector3d>& pose_xyz, bool fixed = false)
  {
    _estimate = pose_xyz;
    setFixed(fixed);
  }

  VertexXYZ(double x, double y, double z, bool fixed = false)
  {
    _estimate.coeffRef(0) = x;
    _estimate.coeffRef(1) = y;
    _estimate.coeffRef(2) = z;
    setFixed(fixed);
  }

  ~VertexXYZ(){}

  virtual void setToOriginImpl() override
  {
    _estimate.setZero();
  }

  inline double& x() {
    return _estimate.coeffRef(0);
  }
  
  inline const double& x() const {
    return _estimate.coeffRef(0);
  }
  
  inline double& y() {
    return _estimate.coeffRef(1);
  }
  
  inline const double& y() const {
    return _estimate.coeffRef(1);
  }

  inline double& z() {
    return _estimate.coeffRef(2);
  }
  
  inline const double& z() const {
    return _estimate.coeffRef(2);
  }

  inline Eigen::Vector3d& position() {
    return _estimate;
  }

  inline const Eigen::Vector3d& position() const{
    return _estimate;
  }

  virtual void oplusImpl(const double* update) override
  {
    _estimate.coeffRef(0) += update[0];
    _estimate.coeffRef(1) += update[1];
    _estimate.coeffRef(2) += update[2];
  }

  virtual bool read(std::istream& is) override
  {
    is >> _estimate.coeffRef(0) >> _estimate.coeffRef(1) >> _estimate.coeffRef(2);
    return true;
  }

  virtual bool write(std::ostream& os) const override
  {
    os << _estimate.coeffRef(0) << " " << _estimate.coeffRef(1) << " " << _estimate.coeffRef(2);
    return os.good();
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif