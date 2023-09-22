//
// Created by admin123456 on 2023/8/29.
//

#ifndef MAPF_PIPELINE_SPRINGVERTEX_XYZR_H
#define MAPF_PIPELINE_SPRINGVERTEX_XYZR_H

#include "assert.h"
#include "eigen3/Eigen/Core"
#include "g2o/config.h"
#include <g2o/core/base_vertex.h>
#include <g2o/core/hyper_graph_action.h>
#include <g2o/stuff/misc.h>
#include "math.h"

using namespace std;

/*
namespace TightSpringNameSpace {

    class SpringVertexStructor : public g2o::BaseVertex<4, Eigen::Vector4d> {
    private:
        std::string name;

    public:
        std::string xyzTag;
        bool fixRadian;

        SpringVertexStructor(bool fixed = false, bool fixRadian = true) : fixRadian(fixRadian) {
            setToOriginImpl();
            setFixed(fixed);
        }

        SpringVertexStructor(
                std::string xyzTag, double x, double y, double z, double radian, bool fixRadian, bool fixed = false
        ): xyzTag(xyzTag), fixRadian(fixRadian) {
            _estimate.coeffRef(0) = x;
            _estimate.coeffRef(1) = y;
            _estimate.coeffRef(2) = z;
            _estimate.coeffRef(3) = radian;
            setFixed(fixed);
        }

        ~SpringVertexStructor() {}

        virtual void setToOriginImpl() override {
            _estimate.setZero();
        }

        inline double &x() {
            return _estimate.coeffRef(0);
        }

        inline const double &x() const {
            return _estimate.coeffRef(0);
        }

        inline double &y() {
            return _estimate.coeffRef(1);
        }

        inline const double &y() const {
            return _estimate.coeffRef(1);
        }

        inline double &z() {
            return _estimate.coeffRef(2);
        }

        inline const double &z() const {
            return _estimate.coeffRef(2);
        }

        inline double &radian() {
            return _estimate.coeffRef(3);
        }

        inline const double &radian() const {
            return _estimate.coeffRef(3);
        }

        inline const double& compute_shapeX(double locX, double locY, double locZ, double& res) const{
            if (fixRadian){
                res = x() + locX;
            } else{
                if (xyzTag == "X"){
                    res = locX + x();
                } else if (xyzTag == "Y"){
                    res = cos(radian()) * locX + sin(radian()) * locZ + x();
                } else{
                    res = cos(radian()) * locX - sin(radian()) * locY + x();
                }
            }
            return res;
        }

        inline const double& compute_shapeY(double locX, double locY, double locZ, double& res) const{
            if (fixRadian){
                res = y() + locY;
            } else{
                if (xyzTag == "X"){
                    res = cos(radian()) * locY - sin(radian()) * locZ + y();
                } else if (xyzTag == "Y"){
                    res = y() + locY;
                } else{
                    res = sin(radian()) * locX + cos(radian()) * locY + y();
                }
            }
            return res;
        }

        inline const double& compute_shapeZ(double locX, double locY, double locZ, double& res) const{
            if (fixRadian){
                res = z() + locZ;
            } else{
                if (xyzTag == "X"){
                    res = sin(radian()) * locY + cos(radian()) * locZ + z();
                } else if (xyzTag == "Y"){
                    res = -sin(radian()) * locX + cos(radian()) * locZ + z();
                } else{
                    res = z() + locZ;
                }
            }
            return res;
        }

        virtual void oplusImpl(const double *update) override {
            Eigen::Vector4d update_vec(update);

            Eigen::Vector3d xyz_vec = update_vec.segment(0, 3);
            double length = xyz_vec.norm();
            if (length > 0.1){
                xyz_vec = xyz_vec / length * min(length, 1.0);
                _estimate.coeffRef(0) += xyz_vec[0];
                _estimate.coeffRef(1) += xyz_vec[1];
                _estimate.coeffRef(2) += xyz_vec[2];
            }

            if (!fixRadian) {
                double radian_vec = min(max(update[3], -0.35), 0.35);
                _estimate.coeffRef(3) += radian_vec;
            }
        }

        virtual bool read(std::istream &is) override {
            is >> _estimate.coeffRef(0) >> _estimate.coeffRef(1) >> _estimate.coeffRef(2) >> _estimate.coeffRef(3);
            return true;
        }

        virtual bool write(std::ostream &os) const override {
            os << _estimate.coeffRef(0) << " " << _estimate.coeffRef(1) << " " << _estimate.coeffRef(2) << " "
               << _estimate.coeffRef(3);
            return os.good();
        }

        void setName(std::string tag){
            this->name = tag;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}
*/

#endif //MAPF_PIPELINE_SPRINGVERTEX_XYZR_H
