//
// Created by admin123456 on 2023/8/25.
//

#ifndef MAPF_PIPELINE_EDGE_PLANE_H
#define MAPF_PIPELINE_EDGE_PLANE_H

#include "assert.h"
#include "smooth/vertex_XYZ.h"
#include "smooth/vertex_XYZR.h"
#include <g2o/core/base_binary_edge.h>

namespace SmootherNameSpace{

    class EdgeXYZ_Plane : public g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZ> {
    public:
        using typename g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZ>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZ>::computeError;

        virtual bool read(std::istream &is) {
            return true;
        }

        virtual bool write(std::ostream &os) const {
            return os.good();
        }

        ErrorVector &getError() {
            computeError();
            return _error;
        }

        EdgeXYZ_Plane(std::string plane, double kSpring = 1.0) : plane(plane), kSpring(kSpring){
            this->setMeasurement(0.);
        }

        void computeError() {
            const VertexXYZ *pose1 = static_cast<const VertexXYZ *>(_vertices[0]);
            const VertexXYZ *pose2 = static_cast<const VertexXYZ *>(_vertices[1]);

            if (plane == "X"){
                _error[0] = (pose1->x() - pose2->x()) * kSpring;
            }else if (plane == "Y"){
                _error[0] = (pose1->y() - pose2->y()) * kSpring;
            }else if (plane == "Z"){
                _error[0] = (pose1->z() - pose2->z()) * kSpring;
            } else{
                assert(false);
            }

            _error[0] = 0;
        }

        static double lost_calc(VertexXYZ *pose1, VertexXYZ *pose2, double kSpring = 1.0) {
            return 0;
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZ>::_error;
        using g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZ>::_vertices;

    private:
        double kSpring;
        std::string plane;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };

    class EdgeXYZR_Plane : public g2o::BaseBinaryEdge<1, double, VertexXYZR, VertexXYZR> {
    public:
        using typename g2o::BaseBinaryEdge<1, double, VertexXYZR, VertexXYZR>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, VertexXYZR, VertexXYZR>::computeError;

        virtual bool read(std::istream &is) {
            return true;
        }

        virtual bool write(std::ostream &os) const {
            return os.good();
        }

        ErrorVector &getError() {
            computeError();
            return _error;
        }

        EdgeXYZR_Plane(std::string plane, double kSpring = 1.0) : plane(plane), kSpring(kSpring){
            this->setMeasurement(0.);
        }

        void computeError() {
            const VertexXYZR *pose1 = static_cast<const VertexXYZR *>(_vertices[0]);
            const VertexXYZR *pose2 = static_cast<const VertexXYZR *>(_vertices[1]);

            if (plane == "X"){
                _error[0] = (pose1->x() - pose2->x()) * kSpring;
            }else if (plane == "Y"){
                _error[0] = (pose1->y() - pose2->y()) * kSpring;
            }else if (plane == "Z"){
                _error[0] = (pose1->z() - pose2->z()) * kSpring;
            } else{
                assert(false);
            }

            _error[0] = 0;
        }

        static double lost_calc(VertexXYZR *pose1, VertexXYZR *pose2, double kSpring = 1.0) {
            return 0;
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, VertexXYZR, VertexXYZR>::_error;
        using g2o::BaseBinaryEdge<1, double, VertexXYZR, VertexXYZR>::_vertices;

    private:
        double kSpring;
        std::string plane;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };

    class EdgeCom_Plane : public g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZR> {
    public:
        using typename g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZR>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZR>::computeError;

        virtual bool read(std::istream &is) {
            return true;
        }

        virtual bool write(std::ostream &os) const {
            return os.good();
        }

        ErrorVector &getError() {
            computeError();
            return _error;
        }

        EdgeCom_Plane(std::string plane, double kSpring = 1.0) : plane(plane), kSpring(kSpring){
            this->setMeasurement(0.);
        }

        void computeError() {
            const VertexXYZ *pose1 = static_cast<const VertexXYZ *>(_vertices[0]);
            const VertexXYZR *pose2 = static_cast<const VertexXYZR *>(_vertices[1]);

            if (plane == "X"){
                _error[0] = (pose1->x() - pose2->x()) * kSpring;
            }else if (plane == "Y"){
                _error[0] = (pose1->y() - pose2->y()) * kSpring;
            }else if (plane == "Z"){
                _error[0] = (pose1->z() - pose2->z()) * kSpring;
            } else{
                assert(false);
            }

            _error[0] = 0;
        }

        static double lost_calc(VertexXYZ *pose1, VertexXYZR *pose2, double kSpring = 1.0) {
            return 0;
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZR>::_error;
        using g2o::BaseBinaryEdge<1, double, VertexXYZ, VertexXYZR>::_vertices;

    private:
        double kSpring;
        std::string plane;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };
}

#endif //MAPF_PIPELINE_EDGE_PLANE_H
