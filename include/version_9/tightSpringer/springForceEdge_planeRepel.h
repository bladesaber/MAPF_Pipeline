//
// Created by admin123456 on 2023/8/31.
//

#ifndef MAPF_PIPELINE_SPRINGEDGE_RANGEFIXED_H
#define MAPF_PIPELINE_SPRINGEDGE_RANGEFIXED_H

#include "assert.h"
#include "eigen3/Eigen/Core"
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>

#include "tightSpringer/springVertex_cell.h"
#include "tightSpringer/springVertex_structor.h"
#include "tightSpringer/springVertex_plane.h"

/*
namespace TightSpringNameSpace {

    // cell <---> plane
    // structor <---> plane

    class SpringForceEdge_CellToPlaneRepel : public g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexPlane> {
    public:
        using typename g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexPlane>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexPlane>::computeError;

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

        SpringForceEdge_CellToPlaneRepel(
                std::string planeTag, std::string compareTag, double bound_shift, double kSpring = 1.0
        ) : bound_shift(bound_shift), kSpring(kSpring) {

            assert(planeTag == "X" || planeTag == "Y" || planeTag == "Z");
            this->planeTag = planeTag;

            assert(compareTag == "larger" || compareTag == "less");
            this->compareTag = compareTag;

            this->setMeasurement(0.);
        }

        void computeError() {
            const SpringVertexCell *vertex0 = static_cast<const SpringVertexCell *>(_vertices[0]);
            const SpringVertexPlane *vertex1 = static_cast<const SpringVertexPlane *>(_vertices[1]);

            double cost;
            if (planeTag == "X") {
                if (compareTag == "larger"){
                    cost = penaltyDownBound(vertex0->x(), vertex1->x(), bound_shift);
                } else{
                    cost = penaltyUpBound(vertex0->x(), vertex1->x(), bound_shift);
                }

            } else if (planeTag == "Y") {
                if (compareTag == "larger"){
                    cost = penaltyDownBound(vertex0->y(), vertex1->y(), bound_shift);
                } else{
                    cost = penaltyUpBound(vertex0->y(), vertex1->y(), bound_shift);
                }

            } else {
                if (compareTag == "larger"){
                    cost = penaltyDownBound(vertex0->z(), vertex1->z(), bound_shift);
                } else{
                    cost = penaltyUpBound(vertex0->z(), vertex1->z(), bound_shift);
                }
            }

            _error[0] = cost * kSpring;
        }

        double lost_calc() {
            throw "Not Implementation";
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexPlane>::_error;
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexPlane>::_vertices;

    private:
        std::string planeTag;
        std::string compareTag;
        double bound_shift;
        double kSpring;

        inline double penaltyDownBound(const double &var, const double &a, double eplision) {
            if (var >= a + eplision) {
                return 0.0;
            } else {
                return a + eplision - var;
            }
        }

        inline double penaltyUpBound(const double &var, const double &a, double eplision) {
            if (var <= a - eplision) {
                return 0.0;
            } else {
                return var - (a - eplision);
            }
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    class SpringForceEdge_structorToPlaneRepel : public g2o::BaseBinaryEdge<1, double, SpringVertexStructor, SpringVertexPlane> {
    public:
        using typename g2o::BaseBinaryEdge<1, double, SpringVertexStructor, SpringVertexPlane>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, SpringVertexStructor, SpringVertexPlane>::computeError;

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

        SpringForceEdge_structorToPlaneRepel(
                std::string planeTag, std::string compareTag,
                std::vector<std::tuple<double, double, double>> conflict_xyzs,
                double bound_shift, double kSpring = 1.0
        ) : planeTag(planeTag), compareTag(compareTag), conflict_xyzs(conflict_xyzs),
            bound_shift(bound_shift), kSpring(kSpring) {

            assert(planeTag == "X" || planeTag == "Y" || planeTag == "Z");
            this->planeTag = planeTag;

            assert(compareTag == "larger" || compareTag == "less");
            this->compareTag = compareTag;

            this->setMeasurement(0.);
        }

        void computeError() {
            const SpringVertexStructor *vertex0 = static_cast<const SpringVertexStructor *>(_vertices[0]);
            const SpringVertexPlane *vertex1 = static_cast<const SpringVertexPlane *>(_vertices[1]);

            double shapeX, shapeY, shapeZ;
            double cost;
            for (auto xyz: conflict_xyzs){
                std::tie(shapeX, shapeY, shapeZ) = xyz;

                if (planeTag == "X"){
                    double x0;
                    vertex0->compute_shapeX(shapeX, shapeY, shapeZ, x0);

                    if (compareTag == "larger") {
                        cost += penaltyDownBound(x0, vertex1->x(), bound_shift);
                    } else {
                        cost += penaltyUpBound(x0, vertex1->x(), bound_shift);
                    }

                } else if (planeTag == "Y"){
                    double y0;
                    vertex0->compute_shapeY(shapeX, shapeY, shapeZ, y0);

                    if (compareTag == "larger") {
                        cost += penaltyDownBound(y0, vertex1->y(), bound_shift);
                    } else {
                        cost += penaltyUpBound(y0, vertex1->y(), bound_shift);
                    }

                } else {
                    double z0;
                    vertex0->compute_shapeZ(shapeX, shapeY, shapeZ, z0);

                    if (compareTag == "larger") {
                        cost += penaltyDownBound(z0, vertex1->z(), bound_shift);
                    } else {
                        cost += penaltyUpBound(z0, vertex1->z(), bound_shift);
                    }
                }
            }

            _error[0] = cost * kSpring;
        }

        virtual void linearizeOplus() override{
            const SpringVertexStructor *vertex0 = static_cast<const SpringVertexStructor *>(_vertices[0]);
            const SpringVertexPlane *vertex1 = static_cast<const SpringVertexPlane *>(_vertices[1]);

            if (planeTag == "X"){
                _jacobianOplusXj(0, 0) = 1.0;
            } else if (planeTag == "Y"){
                _jacobianOplusXj(0, 1) = 1.0;
            }else {
                _jacobianOplusXj(0, 2) = 1.0;
            }

            double shapeX, shapeY, shapeZ;
            double scale = 1.0 / (conflict_xyzs.size() + 1.0);
            for (auto xyz: conflict_xyzs) {
                std::tie(shapeX, shapeY, shapeZ) = xyz;

                double grad_cell = compute_gradRadian(vertex0, shapeX, shapeY, shapeZ);
                _jacobianOplusXj(0, 3) += grad_cell * scale;
            }
        }

        double compute_gradRadian(const SpringVertexStructor *vertex, double locX, double locY, double locZ) {
            if (vertex->fixRadian){
                return 0.0;
            } else{
                if (planeTag == "X"){
                    if (vertex->xyzTag == "X"){
                        return 0.0;
                    } else if (vertex->xyzTag == "Y"){
                        return -locX * sin(vertex->radian()) + locZ * cos(vertex->radian());
                    } else{
                        return -locX * sin(vertex->radian()) - locY * cos(vertex->radian());
                    }

                } else if (planeTag == "Y"){
                    if (vertex->xyzTag == "X"){
                        return -locY * sin(vertex->radian()) - locZ * cos(vertex->radian());
                    } else if (vertex->xyzTag == "Y"){
                        return 0.0;
                    } else{
                        return locX * cos(vertex->radian()) - locY * sin(vertex->radian());
                    }

                } else{
                    if (vertex->xyzTag == "X"){
                        return locY * cos(vertex->radian()) - locZ * sin(vertex->radian());
                    } else if (vertex->xyzTag == "Y"){
                        return -locX * cos(vertex->radian()) - locZ * sin(vertex->radian());
                    } else{
                        return 0.0;
                    }
                }
            }
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, SpringVertexStructor, SpringVertexPlane>::_error;
        using g2o::BaseBinaryEdge<1, double, SpringVertexStructor, SpringVertexPlane>::_vertices;

    private:
        std::vector<std::tuple<double, double, double>> conflict_xyzs;
        std::string planeTag;
        std::string compareTag;
        double bound_shift;
        double kSpring;

        inline double penaltyDownBound(const double &var, const double &a, double eplision) {
            if (var >= a + eplision) {
                return 0.0;
            } else {
                return a + eplision - var;
            }
        }

        inline double penaltyUpBound(const double &var, const double &a, double eplision) {
            if (var <= a - eplision) {
                return 0.0;
            } else {
                return var - (a - eplision);
            }
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}
*/
#endif //MAPF_PIPELINE_SPRINGEDGE_RANGEFIXED_H
