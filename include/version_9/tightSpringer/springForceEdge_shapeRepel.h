//
// Created by admin123456 on 2023/8/31.
//

#ifndef MAPF_PIPELINE_SPRINGEDGE_SHAPEREPEL_H
#define MAPF_PIPELINE_SPRINGEDGE_SHAPEREPEL_H

#include "assert.h"
#include "eigen3/Eigen/Core"
#include <g2o/core/base_binary_edge.h>

#include "tightSpringer/springVertex_cell.h"
#include "tightSpringer/springVertex_structor.h"

using namespace std;

/*
namespace TightSpringNameSpace {

    // cell <---> cell
    // structor <---> cell
    // structor <---> structor

    class SpringForceEdge_CellToCell_ShapeRepel : public g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell> {
    public:
        using typename g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell>::computeError;

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

        SpringForceEdge_CellToCell_ShapeRepel(
                double conflict_thre, double kSpring = 1.0
        ) :conflict_thre(conflict_thre), kSpring(kSpring) {
            this->setMeasurement(0.);
        }

        void computeError() {
            const SpringVertexCell *vertex0 = static_cast<const SpringVertexCell *>(_vertices[0]);
            const SpringVertexCell *vertex1 = static_cast<const SpringVertexCell *>(_vertices[1]);

            double dist = (vertex0->position() - vertex1->position()).norm();

            // TODO 这里限制为球形假定
            _error[0] = penaltyDownBound(dist, conflict_thre) * kSpring;
        }

        double lost_calc() {
            throw "Not Implementation";
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell>::_error;
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexCell>::_vertices;

    private:
        double conflict_thre;
        double kSpring;

        inline double penaltyDownBound(const double &var, const double &a) {
            if (var >= a) {
                return 0.0;
            } else {
                return -var + a;
            }
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    class SpringForceEdge_CellToStructor_ShapeRepel : public g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexStructor> {
    public:
        using typename g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexStructor>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexStructor>::computeError;

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

        SpringForceEdge_CellToStructor_ShapeRepel(
                double shapeX, double shapeY, double shapeZ,
                double conflict_thre, double kSpring = 1.0
        ) : shapeX(shapeX), shapeY(shapeY), shapeZ(shapeZ), conflict_thre(conflict_thre), kSpring(kSpring) {
            this->setMeasurement(0.);
        }

        void computeError() {
            const SpringVertexCell *vertex0 = static_cast<const SpringVertexCell *>(_vertices[0]);
            const SpringVertexStructor *vertex1 = static_cast<const SpringVertexStructor *>(_vertices[1]);

            assert(vertex1->xyzTag == "X" || vertex1->xyzTag == "Y" || vertex1->xyzTag == "Z");

            double x1, y1, z1;
            vertex1->compute_shapeX(shapeX, shapeY, shapeZ, x1);
            vertex1->compute_shapeY(shapeX, shapeY, shapeZ, y1);
            vertex1->compute_shapeZ(shapeX, shapeY, shapeZ, z1);

            // TODO 在只知道碰撞点与优化点的信息下，理论上无法确定优化方向，这里只能通过良好的初始化以及预距离解决
            double dist = sqrt(
                    pow(vertex0->x() - x1, 2.0) + pow(vertex0->y() - y1, 2.0) + pow(vertex0->z() - z1, 2.0)
            );

            _error[0] = penaltyDownBound(dist, conflict_thre) * kSpring;
        }

        double lost_calc() {
            throw "Not Implementation";
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexStructor>::_error;
        using g2o::BaseBinaryEdge<1, double, SpringVertexCell, SpringVertexStructor>::_vertices;

    private:
        double conflict_thre;
        double shapeX, shapeY, shapeZ;
        double kSpring;

        inline double penaltyDownBound(const double &var, const double &a) {
            if (var >= a) {
                return 0.0;
            } else {
                return -var + a;
            }
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    class SpringForceEdge_StructorToStructor_ShapeRepel : public g2o::BaseBinaryEdge<1, double, SpringVertexStructor, SpringVertexStructor> {
    public:
        using typename g2o::BaseBinaryEdge<1, double, SpringVertexStructor, SpringVertexStructor>::ErrorVector;
        using g2o::BaseBinaryEdge<1, double, SpringVertexStructor, SpringVertexStructor>::computeError;

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

        SpringForceEdge_StructorToStructor_ShapeRepel(
                double shapeX_0, double shapeY_0, double shapeZ_0,
                double shapeX_1, double shapeY_1, double shapeZ_1,
                double conflict_thre, double kSpring = 1.0
        ) : shapeX_0(shapeX_0), shapeY_0(shapeY_0), shapeZ_0(shapeZ_0),
            shapeX_1(shapeX_1), shapeY_1(shapeY_1), shapeZ_1(shapeZ_1),
            conflict_thre(conflict_thre), kSpring(kSpring) {
            this->setMeasurement(0.);
        }

        void computeError() {
            const SpringVertexStructor *vertex0 = static_cast<const SpringVertexStructor *>(_vertices[0]);
            const SpringVertexStructor *vertex1 = static_cast<const SpringVertexStructor *>(_vertices[1]);

            assert(vertex0->xyzTag == "X" || vertex0->xyzTag == "Y" || vertex0->xyzTag == "Z");
            assert(vertex1->xyzTag == "X" || vertex1->xyzTag == "Y" || vertex1->xyzTag == "Z");

            double x0, y0, z0;
            vertex0->compute_shapeX(shapeX_0, shapeY_0, shapeZ_0, x0);
            vertex0->compute_shapeY(shapeX_0, shapeY_0, shapeZ_0, y0);
            vertex0->compute_shapeZ(shapeX_0, shapeY_0, shapeZ_0, z0);

            double x1, y1, z1;
            vertex1->compute_shapeX(shapeX_1, shapeY_1, shapeZ_1, x1);
            vertex1->compute_shapeY(shapeX_1, shapeY_1, shapeZ_1, y1);
            vertex1->compute_shapeZ(shapeX_1, shapeY_1, shapeZ_1, z1);

            // TODO 在只知道碰撞点与优化点的信息下，理论上无法确定优化方向，这里只能通过良好的初始化以及预距离解决
            double dist = sqrt(pow(x0 - x1, 2.0) + pow(y0 - y1, 2.0) + pow(z0 - z1, 2.0));
            _error[0] = penaltyDownBound(dist, conflict_thre) * kSpring;
        }

        double lost_calc() {
            throw "Not Implementation";
        }

    protected:
        using g2o::BaseBinaryEdge<1, double, SpringVertexStructor, SpringVertexStructor>::_error;
        using g2o::BaseBinaryEdge<1, double, SpringVertexStructor, SpringVertexStructor>::_vertices;

    private:
        double conflict_thre;
        double shapeX_0, shapeY_0, shapeZ_0;
        double shapeX_1, shapeY_1, shapeZ_1;
        double kSpring;

        inline double penaltyDownBound(const double &var, const double &a) {
            if (var >= a) {
                return 0.0;
            } else {
                return -var + a;
            }
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}
*/
#endif //MAPF_PIPELINE_SPRINGEDGE_SHAPEREPEL_H
