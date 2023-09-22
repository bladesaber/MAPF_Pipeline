//
// Created by admin123456 on 2023/8/31.
//

#ifndef MAPF_PIPELINE_SPRINGSMOOTHER_H
#define MAPF_PIPELINE_SPRINGSMOOTHER_H

#include "assert.h"
#include "string"
#include "assert.h"
#include "eigen3/Eigen/Core"

#include "g2o/core/block_solver.h"
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"

#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

#include "tightSpringer/springVertex_structor.h"
#include "tightSpringer/springVertex_cell.h"
#include "tightSpringer/springVertex_plane.h"

#include "tightSpringer/springConEdge_ElasticBand.h"
#include "tightSpringer/springConEdge_kinematicSegment.h"
#include "tightSpringer/springConEdge_kinematicPoint.h"
#include "tightSpringer/springPoseEdge_poseFixed.h"
#include "tightSpringer/springPoseEdge_radiusFixed.h"
#include "tightSpringer/springForceEdge_planeRepel.h"
#include "tightSpringer/springForceEdge_shapeRepel.h"
#include "tightSpringer/springPoseEdge_valueShift.h"
#include "tightSpringer/springTarEdge_minVolume.h"
#include "tightSpringer/springTarEdge_poseCluster.h"
#include "tightSpringer/springTarEdge_minAxes.h"

/*
namespace TightSpringNameSpace {

    class Springer_Plane {
    public:
        std::string name;
        size_t nodeIdx;
        double x, y, z;
        bool fixed = false;

        Springer_Plane(std::string name, size_t nodeIdx, double x, double y, double z, bool fixed) :
                name(name), nodeIdx(nodeIdx), x(x), y(y), z(z), fixed(fixed) {};

        ~Springer_Plane() {
            release_veterx();
        };

        SpringVertexPlane *vertex;
        bool has_vertex = false;

        double vertex_x() { return vertex->x(); }

        double vertex_y() { return vertex->y(); }

        double vertex_z() { return vertex->z(); }

        void updateVertex() {
            x = vertex_x();
            y = vertex_y();
            z = vertex_z();
        }

        void release_veterx() {
            if (has_vertex) {
                delete vertex;
                has_vertex = false;
            }
        }
    };

    class Springer_Cell {
    public:
        size_t nodeIdx;
        bool fixed = false;

        double x, y, z, radius;

        Springer_Cell(size_t nodeIdx, double x, double y, double z, double radius, bool fixed) :
                nodeIdx(nodeIdx), x(x), y(y), z(z), radius(radius), fixed(fixed) {};

        ~Springer_Cell() {
            release_veterx();
        };

        SpringVertexCell *vertex;
        bool has_vertex = false;

        double vertex_x() { return vertex->x(); }

        double vertex_y() { return vertex->y(); }

        double vertex_z() { return vertex->z(); }

        void updateVertex() {
            x = vertex_x();
            y = vertex_y();
            z = vertex_z();
        }

        void release_veterx() {
            if (has_vertex) {
                delete vertex;
                has_vertex = false;
            }
        }
    };

    class Springer_Connector {
    public:
        std::string name;
        size_t nodeIdx;
        bool fixed = false;

        double x, y, z;

        Springer_Connector(std::string name, size_t nodeIdx, double x, double y, double z, bool fixed) :
                name(name), nodeIdx(nodeIdx), x(x), y(y), z(z), fixed(fixed) {};

        ~Springer_Connector() {
            release_veterx();
        };

        SpringVertexCell *vertex;
        bool has_vertex = false;

        double vertex_x() { return vertex->x(); }

        double vertex_y() { return vertex->y(); }

        double vertex_z() { return vertex->z(); }

        void updateVertex() {
            x = vertex_x();
            y = vertex_y();
            z = vertex_z();
        }

        void release_veterx() {
            if (has_vertex) {
                delete vertex;
                has_vertex = false;
            }
        }
    };

    class Springer_Structor {
    public:
        std::string name;
        size_t nodeIdx;
        bool fixed = false;

        std::string xyzTag;
        double x, y, z, radian;
        bool fixRadian;
        double shell_radius;

        Springer_Structor(
                std::string name, size_t nodeIdx, std::string xyzTag,
                double x, double y, double z, double radian, double shell_radius, bool fixed
        ) : name(name), nodeIdx(nodeIdx), xyzTag(xyzTag), x(x), y(y), z(z),
            radian(radian), shell_radius(shell_radius), fixed(fixed) {
            assert(xyzTag == "X" || xyzTag == "Y" || xyzTag == "Z" || xyzTag == "None");
            this->xyzTag = xyzTag;
            if (xyzTag == "None") {
                fixRadian = true;
            } else {
                fixRadian = false;
            }
        };

        ~Springer_Structor() {
            release_veterx();
        };

        SpringVertexStructor *vertex;
        bool has_vertex = false;

        double vertex_x() { return vertex->x(); }

        double vertex_y() { return vertex->y(); }

        double vertex_z() { return vertex->z(); }

        double vertex_radian() { return vertex->radian(); }

        void updateVertex() {
            x = vertex_x();
            y = vertex_y();
            z = vertex_z();
            radian = vertex_radian();
        }

        void release_veterx() {
            if (has_vertex) {
                delete vertex;
                has_vertex = false;
            }
        }
    };

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>> BlockSolver;
    typedef g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType> LinearSolver;

    class SpringerSmooth_Runner {
    public:
        SpringerSmooth_Runner() {};

        ~SpringerSmooth_Runner() {
            clear_graph();
            clear_graphNodes();
            release();
        };

        std::map<size_t, Springer_Plane *> plane_NodeMap;
        std::map<size_t, Springer_Cell *> cell_NodeMap;
        std::map<size_t, Springer_Connector *> connector_NodeMap;
        std::map<size_t, Springer_Structor *> structor_NodeMap;

        // ------ add objects
        bool add_Plane(std::string name, size_t nodeIdx, double x, double y, double z, bool fixed) {
            if (plane_NodeMap.find(nodeIdx) == plane_NodeMap.end()) {
                Springer_Plane *node = new Springer_Plane(name, nodeIdx, x, y, z, fixed);
                plane_NodeMap[nodeIdx] = node;
                return true;
            }
            return false;
        }

        bool add_Cell(size_t nodeIdx, double x, double y, double z, double radius, bool fixed) {
            if (cell_NodeMap.find(nodeIdx) == cell_NodeMap.end()) {
                Springer_Cell *node = new Springer_Cell(nodeIdx, x, y, z, radius, fixed);
                cell_NodeMap[nodeIdx] = node;
                return true;
            }
            return false;
        }

        bool add_Connector(std::string name, size_t nodeIdx, double x, double y, double z, bool fixed) {
            if (connector_NodeMap.find(nodeIdx) == connector_NodeMap.end()) {
                Springer_Connector *node = new Springer_Connector(name, nodeIdx, x, y, z, fixed);
                connector_NodeMap[nodeIdx] = node;
                return true;
            }
            return false;
        }

        bool add_Structor(
                std::string name, size_t nodeIdx, std::string xyzTag,
                double x, double y, double z, double radian, double shell_radius, bool fixed
        ) {
            if (structor_NodeMap.find(nodeIdx) == structor_NodeMap.end()) {
                Springer_Structor *node = new Springer_Structor(
                        name, nodeIdx, xyzTag, x, y, z, radian, shell_radius, fixed
                );
                structor_NodeMap[nodeIdx] = node;
                return true;
            }
            return false;
        }

        // ------ Position Edge Type ------
        // structor <---> plane
        bool addEdge_structorToPlane_valueShift(
                size_t planeIdx, size_t structorIdx, std::string xyzTag, double shiftValue,
                double kSpring, double weight
        );

        // connector <---> structor
        bool addEdge_connectorToStruct_valueShift(
                size_t connectorIdx, size_t structorIdx, std::string xyzTag, double shiftValue,
                double kSpring, double weight
        );

        bool addEdge_connectorToStruct_radiusFixed(
                size_t connectorIdx, size_t structorIdx, std::string xyzTag, double radius,
                double kSpring, double weight
        );

        bool addEdge_connectorToStruct_poseFixed(
                size_t connectorIdx, size_t structorIdx, std::string xyzTag,
                double shapeX, double shapeY, double shapeZ,
                double kSpring, double weight
        );

        // ------ Connect Edge Type ------
        // Cell <---> Connector
        bool addEdge_cellToConnector_elasticBand(size_t cellIdx, size_t connectorIdx, double kSpring, double weight);

        bool addEdge_cellToCell_elasticBand(size_t cellIdx0, size_t cellIdx1, double kSpring, double weight);

        bool addEdge_cellToConnector_kinematicPoint(
                size_t cellIdx, size_t connectorIdx, double vecX, double vecY, double vecZ, double targetValue,
                bool fromCell, double kSpring, double weight
        );

        bool addEdge_cellToConnector_kinematicSegment(
                size_t connectorIdx, size_t cellIdx0, size_t cellIdx1,
                double targetValue, bool fromConnector, double kSpring, double weight
        );

        bool addEdge_cellToCell_kinematicSegment(
                size_t cellIdx0, size_t cellIdx1, size_t cellIdx2, double targetValue, double kSpring, double weight
        );

        // ------ Force Edge Type ------
        bool addEdge_structorToPlane_planeRepel(
                size_t structorIdx, size_t planeIdx,
                std::string planeTag, std::string compareTag,
                std::vector<std::tuple<double, double, double>> conflict_xyzs,
                double bound_shift, double kSpring, double weight
        );

        bool addEdge_cellToPlane_planeRepel(
                size_t cellIdx, size_t planeIdx,
                std::string planeTag, std::string compareTag,
                double bound_shift, double kSpring, double weight
        );

        bool addEdge_cellToCell_shapeRepel(
                size_t cellIdx0, size_t cellIdx1, double bound_shift, double kSpring, double weight
        );

        bool addEdge_cellToStructor_shapeRepel(
                size_t cellIdx, size_t structorIdx,
                double shapeX, double shapeY, double shapeZ,
                double bound_shift, double kSpring, double weight
        );

        bool addEdge_structorToStructor_shapeRepel(
                size_t structorIdx0, size_t structorIdx1,
                double shapeX_0, double shapeY_0, double shapeZ_0,
                double shapeX_1, double shapeY_1, double shapeZ_1,
                double bound_shift, double kSpring, double weight
        );

        // ------ Target Edge Type ------
        bool addEdge_minVolume(size_t minPlaneIdx, size_t maxPlaneIdx, double scale, double kSpring, double weight);

        bool addEdge_structor_poseCluster(size_t structorIdx, double scale, double kSpring, double weight);

        bool addEdge_connector_poseCluster(size_t connectorIdx, double scale, double kSpring, double weight);

        bool addEdge_minAxes(
                size_t minPlaneIdx, size_t maxPlaneIdx, std::string xyzTag, double scale,
                double kSpring, double weight
        );

        // --------------------------------------------------------
        void initOptimizer(std::string method="Levenberg") {
            assert(method=="Levenberg" || method=="GaussNewton" || method == "Dogleg");

            optimizer = std::make_shared<g2o::SparseOptimizer>();

            // 选择不同的矩阵稀疏方法
            // g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>* linear_solver = new g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>();
            // std::unique_ptr<g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>> linear_solver(
            //         new g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>()
            // );
             std::unique_ptr<LinearSolver> linear_solver(new LinearSolver());

            linear_solver->setBlockOrdering(true);

            // BlockSolver* block_solver = new BlockSolver(linear_solver);
            std::unique_ptr<BlockSolver> block_solver(new BlockSolver(std::move(linear_solver)));

            // 由于前面用到 unique_ptr 所以使用 std::move
            if (method == "Levenberg"){
                g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
                optimizer->setAlgorithm(solver);

            } else if (method == "GaussNewton"){
                g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(std::move(block_solver));
                optimizer->setAlgorithm(solver);

            } else{
                g2o::OptimizationAlgorithmDogleg *solver = new g2o::OptimizationAlgorithmDogleg(std::move(block_solver));
                optimizer->setAlgorithm(solver);
            }

            // set optimizer
            optimizer->initMultiThreading(); // required for >Eigen 3.1
            optimizer->setComputeBatchStatistics(true);
        }

        void clear_graph() {
            if (optimizer) {
                // we will delete all edges but keep the vertices.
                // before doing so, we will delete the link from the vertices to the edges.
                auto &vertices = optimizer->vertices();
                for (auto &v: vertices) {
                    v.second->edges().clear();
                }

                // necessary, because optimizer->clear deletes pointer-targets (therefore it deletes TEB states!)
                optimizer->vertices().clear();
                optimizer->clear();
            }
        }

        bool is_g2o_graph_empty() {
            return optimizer->edges().empty() && optimizer->vertices().empty();
        }

        void update_nodeMapVertex() {
            for (auto iter: plane_NodeMap) {
                iter.second->updateVertex();
            }
            for (auto iter: connector_NodeMap) {
                iter.second->updateVertex();
            }
            for (auto iter: cell_NodeMap) {
                iter.second->updateVertex();
            }
            for (auto iter: structor_NodeMap) {
                iter.second->updateVertex();
            }
        }

        bool add_vertexes() {
            for (auto iter: plane_NodeMap) {
                Springer_Plane *node = iter.second;
                node->vertex = new SpringVertexPlane(node->x, node->y, node->z, node->fixed);
                node->has_vertex = true;
                node->vertex->setId(node->nodeIdx);
                node->vertex->setName(node->name);

                bool success = optimizer->addVertex(node->vertex);
                if (!success) {
                    std::printf("[WARNING]: Plane Vertex %zu Init Fail", iter.second->nodeIdx);
                    return false;
                }
            }

            for (auto iter: connector_NodeMap) {
                Springer_Connector *node = iter.second;
                node->vertex = new SpringVertexCell(node->x, node->y, node->z, node->fixed);
                node->has_vertex = true;
                node->vertex->setId(node->nodeIdx);
                node->vertex->setName(node->name);

                bool success = optimizer->addVertex(node->vertex);
                if (!success) {
                    std::printf("[WARNING]: Connector Vertex %zu Init Fail", iter.second->nodeIdx);
                    return false;
                }
            }

            for (auto iter: cell_NodeMap) {
                Springer_Cell *node = iter.second;
                node->vertex = new SpringVertexCell(node->x, node->y, node->z, node->fixed);
                node->has_vertex = true;
                node->vertex->setId(node->nodeIdx);

                bool success = optimizer->addVertex(node->vertex);
                if (!success) {
                    std::printf("[WARNING]: Cell Vertex %zu Init Fail", iter.second->nodeIdx);
                    return false;
                }
            }

            for (auto iter: structor_NodeMap) {
                Springer_Structor *node = iter.second;
                node->vertex = new SpringVertexStructor(
                        node->xyzTag, node->x, node->y, node->z, node->radian, node->fixRadian, node->fixed
                );
                node->has_vertex = true;
                node->vertex->setId(node->nodeIdx);
                node->vertex->setName(node->name);

                bool success = optimizer->addVertex(node->vertex);
                if (!success) {
                    std::printf("[WARNING]: Cell Vertex %zu Init Fail", iter.second->nodeIdx);
                    return false;
                }
            }

            return true;
        }

        void clear_vertexes() {
            for (auto iter: plane_NodeMap) {
                iter.second->release_veterx();
            }

            for (auto iter: connector_NodeMap) {
                iter.second->release_veterx();
            }

            for (auto iter: structor_NodeMap) {
                iter.second->release_veterx();
            }

            for (auto iter: cell_NodeMap) {
                iter.second->release_veterx();
            }
        }

        void clear_graphNodes() {
            for (auto iter: plane_NodeMap) {
                delete iter.second;
            }
            plane_NodeMap.clear();

            for (auto iter: connector_NodeMap) {
                delete iter.second;
            }
            connector_NodeMap.clear();

            for (auto iter: structor_NodeMap) {
                delete iter.second;
            }
            structor_NodeMap.clear();

            for (auto iter: cell_NodeMap) {
                delete iter.second;
            }
            cell_NodeMap.clear();
        }

        void optimizeGraph(int no_iterations, bool verbose) {
            optimizer->setVerbose(verbose);
            optimizer->initializeOptimization();

            int iter = optimizer->optimize(no_iterations);
            if (!iter) {
                std::cout << "optimizeGraph(): Optimization failed! iter=" << iter << std::endl;
                assert(false);
            }
        }

        void info() {
            std::cout << "Graph Vertex Size:" << optimizer->vertices().size();
            std::cout << " Edge Size:" << optimizer->edges().size() << std::endl;
        }

    private:
          std::shared_ptr<g2o::SparseOptimizer> optimizer;

        void release(){
             optimizer = nullptr;
        }
    };

}
*/
#endif //MAPF_PIPELINE_SPRINGSMOOTHER_H
