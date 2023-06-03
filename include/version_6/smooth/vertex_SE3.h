#ifndef MAPF_PIPELINE_VERTEX_SE3_H
#define MAPF_PIPELINE_VERTEX_SE3_H

#include "eigen3/Eigen/Core"
#include "g2o/config.h"
#include <g2o/core/base_vertex.h>
#include <g2o/core/hyper_graph_action.h>
#include <g2o/stuff/misc.h>
#include "math.h"

/*
namespace SmootherNameSpace{

class PoseSE3
{
public:
  PoseSE3(){
    setZero();
  };
  ~PoseSE3(){};

  void setZero()
  {
    _position.setZero();
    _alpha = 0.0;
    _theta = 0.0;
  }

  PoseSE3(const Eigen::Ref<const Eigen::Vector3d>& position, double alpha, double theta)
  {
      _position = position;
      _alpha = alpha;
      _theta = theta;
  }

  PoseSE3(double x, double y, double z, double alpha, double theta)
  {
      _position.coeffRef(0) = x;
      _position.coeffRef(1) = y;
      _position.coeffRef(2) = z;
      _alpha = alpha;
      _theta = theta;
  }

  PoseSE3(const PoseSE3& pose)
  {
      _position = pose._position;
      _alpha = pose._alpha;
      _theta = pose._theta;
  }

  Eigen::Vector3d& position(){
    return _position;
  }

  const Eigen::Vector3d& position() const {
    return _position;
  }

  double& x() {
    return _position.coeffRef(0);
  }

  const double& x() const {
    return _position.coeffRef(0);
  }

  double& y() {
    return _position.coeffRef(1);
  }

  const double& y() const {
    return _position.coeffRef(1);
  }

  double& z() {
    return _position.coeffRef(2);
  }

  const double& z() const {
    return _position.coeffRef(2);
  }

  double& alpha(){
    return _alpha;
  } 

  const double& alpha() const {
    return _alpha;
  }

  double& theta(){
    return _theta;
  } 

  const double& theta() const {
    return _theta;
  }

  Eigen::Vector3d orientation2UnitVec() const{
    double dz = sin(_theta);
    double dl = cos(_theta);
    double dx = dl * cos(_alpha);
    double dy = dl * sin(_alpha);
    return Eigen::Vector3d(dx, dy, dz);
  }

  // double fmodr( double x, double y)
  // {
  //   return x - y * floor(x / y);
  // }

  // double mod2pi(double theta)
  // {
  //   return fmodr( theta, 2 * M_PI );
  // }

  // void unitVec2Orientation(Eigen::Vector3d vec) const{
  //   _alpha = mod2pi(std::atan2(vec[1], vec[0]));
  //   double length = std::sqrt( std::pow(vec[0], 2) + std::pow(vec[1], 2));
  //   _theta = std::atan2(vec[2], length);
  // }

  void plus(const double* pose_as_array)
  {
    _position.coeffRef(0) += pose_as_array[0];
    _position.coeffRef(1) += pose_as_array[1];
    _position.coeffRef(2) += pose_as_array[2];
    _alpha = g2o::normalize_theta( _alpha + pose_as_array[3] );
    _theta = g2o::normalize_theta( _theta + pose_as_array[4] );
  }

  PoseSE3& operator=(const PoseSE3& rhs) 
  {
    if (&rhs != this)
    {
    _position = rhs._position;
    _alpha = rhs._alpha;
    _theta = rhs._theta;
    }
    return *this;
  }

  PoseSE3& operator+=(const PoseSE3& rhs)
  {
    _position += rhs._position;
    _alpha = g2o::normalize_theta(_alpha + rhs._alpha);
    _theta = g2o::normalize_theta(_theta + rhs._theta);
    return *this;
  }

  friend PoseSE3 operator+(PoseSE3 lhs, const PoseSE3& rhs) 
  {
    return lhs += rhs;
  }

  PoseSE3& operator-=(const PoseSE3& rhs)
  {
    _position -= rhs._position;
    _alpha = g2o::normalize_theta(_alpha - rhs._alpha);
    _theta = g2o::normalize_theta(_theta - rhs._theta);
    return *this;
  }

  friend PoseSE3 operator-(PoseSE3 lhs, const PoseSE3& rhs) 
  {
    return lhs -= rhs;
  }

  friend std::ostream& operator<< (std::ostream& stream, const PoseSE3& pose)
	{
		stream << "x:" << pose._position[0] << " y:" << pose._position[1] << " z:" << pose._position[2];
    stream << " alpha:" << pose._alpha << " theta:" << pose._theta;
    return stream;
	}

private:
  Eigen::Vector3d _position; 
  double _alpha;
  double _theta;
      
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  

};

class VertexSE3: public g2o::BaseVertex<5, PoseSE3>
{
public:
  VertexSE3(bool fixed = false)
  {
    setToOriginImpl();
    setFixed(fixed);
  }

  VertexSE3(const PoseSE3& pose, bool fixed = false)
  {
    _estimate = pose;
    setFixed(fixed);
  }

  VertexSE3(
    const Eigen::Ref<const Eigen::Vector3d>& position, 
    double alpha, double theta, bool fixed = false
  )
  {
    _estimate.position() = position;
    _estimate.alpha() = alpha;
    _estimate.theta() = theta;
    setFixed(fixed);
  }

  VertexSE3(
    double x, double y, double z, 
    double alpha, double theta, bool fixed = false
  )
  {
    _estimate.x() = x;
    _estimate.y() = y;
    _estimate.z() = z;
    _estimate.alpha() = alpha;
    _estimate.theta() = theta;
    setFixed(fixed);
  }

  inline PoseSE3& pose(){
    return _estimate;
  }

  inline const PoseSE3& pose() const {
    return _estimate;
  }

  virtual void setToOriginImpl() override
  {
    _estimate.setZero();
  }

  inline Eigen::Vector3d& position() {
    return _estimate.position();
  }

  inline const Eigen::Vector3d& position() const {
    return _estimate.position();
  }

  inline double& x() {
    return _estimate.x();
  }
  
  inline const double& x() const {
    return _estimate.x();
  }
  
  inline double& y() {
    return _estimate.y();
  }
  
  inline const double& y() const {
    return _estimate.y();
  }

  inline double& z() {
    return _estimate.z();
  }
  
  inline const double& z() const {
    return _estimate.z();
  }

  inline double& alpha(){
    return _estimate.alpha();
  }
  
  inline const double& alpha() const {
    return _estimate.alpha();
  }

  inline double& theta(){
    return _estimate.theta();
  }
  
  inline const double& theta() const {
    return _estimate.theta();
  }

  virtual void oplusImpl(const double* update) override
  {
    _estimate.plus(update);
  }

  virtual bool read(std::istream& is) override
  {
    is >> _estimate.x() >> _estimate.y() >> _estimate.z() >> _estimate.alpha() >> _estimate.theta();
    return true;
  }

  virtual bool write(std::ostream& os) const override
  {
    os << _estimate.x() << " " << _estimate.y() << " " << _estimate.z();
    os << " " << _estimate.alpha() << " " << _estimate.theta();
    return os.good();
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

*/

#endif