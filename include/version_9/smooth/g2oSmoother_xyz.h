//
// Created by admin123456 on 2023/8/9.
//

#ifndef MAPF_PIPELINE_G2OSMOOTHER_XYZ_H
#define MAPF_PIPELINE_G2OSMOOTHER_XYZ_H

#include "string"
#include "assert.h"
#include "eigen3/Eigen/Core"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"

#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

#include "vertex_XYZ.h"
#include "edge_elastic_band.h"
#include "edge_kinematics.h"
#include "edge_obstacle.h"
#include "edge_pipe_conflict.h"

#include "kdtree_xyzra.h"
#include "utils.h"

using namespace PathNameSpace;

namespace SmootherNameSpace {
    class NxGraphNode {
    public:
        size_t nodeIdx;
        bool fixed = false;

        double x, y, z, radius;
        double vec_i = -1;
        double vec_j = -1;
        double vec_k = -1;

        NxGraphNode(size_t nodeIdx, double x, double y, double z, double radius, bool fixed) :
                nodeIdx(nodeIdx), x(x), y(y), z(z), radius(radius), fixed(fixed) {};

        ~NxGraphNode() {
            release();
        };

        VertexXYZ *vertex;
        bool has_vertex = false;

        double vertex_x() { return vertex->x(); }

        double vertex_y() { return vertex->y(); }

        double vertex_z() { return vertex->z(); }

        void updateVertex() {
            x = vertex_x();
            y = vertex_y();
            z = vertex_z();
        }

    private:
        void release() {
            if (has_vertex) {
                delete vertex;
            }
        }
    };

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>> BlockSolver;
    typedef g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType> LinearSolver;

    class FlexSmootherXYZ_Runner {
    public:
        FlexSmootherXYZ_Runner() {
            obsTree = new PathNameSpace::KDTree_XYZRA();
        };

        ~FlexSmootherXYZ_Runner() {
            release();
        };

        std::map<size_t, NxGraphNode *> graphNode_map;

        void initOptimizer() {
            optimizer = std::make_shared<g2o::SparseOptimizer>();

            // 选择不同的矩阵稀疏方法
            // g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>* linear_solver = new g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>();
            // g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>* linear_solver = new g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>();

            std::unique_ptr<LinearSolver> linear_solver(new LinearSolver());
            linear_solver->setBlockOrdering(true);

            // BlockSolver* block_solver = new BlockSolver(linear_solver);
            std::unique_ptr<BlockSolver> block_solver(new BlockSolver(std::move(linear_solver)));

            // 选择不同的梯度方法
            // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(block_solver);
            // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(block_solver);

            // 由于前面用到 unique_ptr 所以使用 std::move
            g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
                    std::move(block_solver));

            // set optimizer
            optimizer->setAlgorithm(solver);
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

        bool add_graphNode(size_t nodeIdx, double x, double y, double z, double radius, bool fixed) {
            if (graphNode_map.find(nodeIdx) == graphNode_map.end()) {
                NxGraphNode *node = new NxGraphNode(nodeIdx, x, y, z, radius, fixed);
                graphNode_map[nodeIdx] = node;
                return true;
            }
            return false;
        }

        void clear_graphNodeMap() {
            for (auto iter: graphNode_map) {
                delete iter.second;
            }
            graphNode_map.clear();
            // graphNode_map.erase(nodeIdx);
        }

        bool add_vertex(size_t nodeIdx) {
            NxGraphNode *node = graphNode_map[nodeIdx];
            if (node->has_vertex) {
                return true;

            } else {
                node->vertex = new VertexXYZ(node->x, node->y, node->z, node->fixed);
                node->has_vertex = true;
                node->vertex->setId(nodeIdx);
                bool success = optimizer->addVertex(node->vertex);
                return success;
            }
        }

        bool is_g2o_graph_empty() {
            return optimizer->edges().empty() && optimizer->vertices().empty();
        }

        bool is_g2o_graph_edges_empty() {
            return optimizer->edges().empty();
        }

        bool is_g2o_graph_vertices_empty() {
            return optimizer->vertices().empty();
        }

        void add_obstacle(double x, double y, double z, double radius) {
            obsTree->insertNode(0, x, y, z, radius, 0, 0);
        }

        bool add_elasticBand(size_t nodeIdx0, size_t nodeIdx1, double kSpring, double weight) {
            NxGraphNode *node0 = graphNode_map[nodeIdx0];
            NxGraphNode *node1 = graphNode_map[nodeIdx1];

            EdgeXYZ_ElasticBand *edge = new EdgeXYZ_ElasticBand(kSpring);
            edge->setVertex(0, node0->vertex);
            edge->setVertex(1, node1->vertex);

            Eigen::Matrix<double, 1, 1> information;
            // information.fill( elasticBand_weight * current_loss );
            information.fill(weight);
            edge->setInformation(information);

            bool success = optimizer->addEdge(edge);
            return success;
        }

        bool add_kinematicEdge(size_t nodeIdx0, size_t nodeIdx1, size_t nodeIdx2, double kSpring, double weight) {
            NxGraphNode *node0 = graphNode_map[nodeIdx0];
            NxGraphNode *node1 = graphNode_map[nodeIdx1];
            NxGraphNode *node2 = graphNode_map[nodeIdx2];

            double length = std::min(
                    norm2_distance(node0->x, node0->y, node0->z, node1->x, node1->y, node1->z),
                    norm2_distance(node1->x, node1->y, node1->z, node2->x, node2->y, node2->z)
            );
            double cosTheta_target = 1.0 - std::cos(1.0 / (3.0 * std::min(node1->radius, node2->radius)) * length);
//            double cosTheta_target = 0.0;

            EdgeXYZ_Kinematics *edge = new EdgeXYZ_Kinematics(cosTheta_target, kSpring);
            edge->setVertex(0, node0->vertex);
            edge->setVertex(1, node1->vertex);
            edge->setVertex(2, node2->vertex);

            Eigen::Matrix<double, 1, 1> information;
            information.fill(weight);
            edge->setInformation(information);

            bool success = optimizer->addEdge(edge);
            return success;
        }

        bool add_kinematicVertexEdge(
                size_t nodeIdx0, size_t nodeIdx1, double vec_i, double vec_j, double vec_k,
                double kSpring, double weight) {
            NxGraphNode *node0 = graphNode_map[nodeIdx0];
            NxGraphNode *node1 = graphNode_map[nodeIdx1];

            Eigen::Vector3d orientation = Eigen::Vector3d(vec_i, vec_j, vec_k);
            double length = norm2_distance(node0->x, node0->y, node0->z, node1->x, node1->y, node1->z);
            double cosTheta_target = 1.0 - std::cos(1.0 / (3.0 * std::min(node0->radius, node1->radius)) * length);
//            double cosTheta_target = 0.0;

            EdgeXYZ_VertexKinematics *edge = new EdgeXYZ_VertexKinematics(orientation, cosTheta_target, kSpring);
            edge->setVertex(0, node0->vertex);
            edge->setVertex(1, node1->vertex);

            Eigen::Matrix<double, 1, 1> information;
            information.fill(weight);
            edge->setInformation(information);

            bool success = optimizer->addEdge(edge);
            return success;
        }

        bool add_obstacleEdge(size_t nodeIdx, double searchScale, double repleScale, double kSpring, double weight) {
            if (obsTree->getTreeCount() == 0) {
                return true;
            }

            NxGraphNode *node = graphNode_map[nodeIdx];

            std::vector<PathNameSpace::KDTree_XYZRA_Res *> resList;
            obsTree->nearest_range(node->x, node->y, node->z, node->radius * searchScale, resList);

            bool success = true;
            for (PathNameSpace::KDTree_XYZRA_Res *res: resList) {
                EdgeXYZ_Obstacle *edge = new EdgeXYZ_Obstacle(
                        Eigen::Vector3d(res->x, res->y, res->z),
                        (node->radius + res->data->radius) * repleScale, kSpring
                );
                edge->setVertex(0, node->vertex);

                Eigen::Matrix<double, 1, 1> information;
                information.fill(weight);
                edge->setInformation(information);

                delete res;
                success = success && optimizer->addEdge(edge);
            }
            return success;
        }

        bool add_pipeConflictEdge(size_t nodeIdx, size_t groupIdx, double searchScale, double repleScale,
                                  double kSpring, double weight) {
            NxGraphNode *node = graphNode_map[nodeIdx];

            bool success = true;
            std::vector<KDTree_XYZRA_Res *> resList;
            for (auto iter: groupPathTree_map) {
                if (groupIdx == iter.first) {
                    continue;
                }

                resList.clear();
                double max_radius = groupPathMaxRadius_map[iter.first];
                iter.second->nearest_range(node->x, node->y, node->z, (max_radius + node->radius) * searchScale,resList);

                for (KDTree_XYZRA_Res *res: resList) {
                    double dist = norm2_distance(node->x, node->y, node->z, res->x, res->y, res->z);
                    if (dist <= (node->radius + res->data->radius) * searchScale) {
                        EdgeXYZ_PipeConflict *edge = new EdgeXYZ_PipeConflict(
                                (node->radius + res->data->radius) * repleScale, kSpring
                        );
                        edge->setVertex(0, node->vertex);
                        edge->setVertex(1, graphNode_map[res->data->idx]->vertex);

                        Eigen::Matrix<double, 1, 1> information;
                        information.fill(weight);
                        edge->setInformation(information);

                        success = success && optimizer->addEdge(edge);
                    }
                    delete res;
                }
            }
            return success;
        }

        void updateNodeMap_Vertex() {
            for (auto iter: graphNode_map) {
                iter.second->updateVertex();
            }
        }

        void updateGroupTrees(size_t groupIdx, std::vector<size_t> nodeIdxs) {
            bool isInit = groupPathTree_map.find(groupIdx) == groupPathTree_map.end();
            if (!isInit) {
                delete groupPathTree_map[groupIdx];
            }

            groupPathMaxRadius_map[groupIdx] = DBL_MIN;
            KDTree_XYZRA *tree = new KDTree_XYZRA();

            for (size_t nodeIdx: nodeIdxs) {
                NxGraphNode *node = graphNode_map[nodeIdx];
                tree->insertNode(node->nodeIdx, node->x, node->y, node->z, node->radius, 0.0, 0.0);

                groupPathMaxRadius_map[groupIdx] = std::max(groupPathMaxRadius_map[groupIdx], node->radius);
            }
            groupPathTree_map[groupIdx] = tree;
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
//            for (auto iter: groupPathTree_map) {
//                std::cout << "groupIdx:" << iter.first << " TreeCount:" << iter.second->getTreeCount() << std::endl;
//            }
            std::cout << "obstacleTree Size:" << obsTree->getTreeCount() << std::endl;
            std::cout << "Graph Vertex Size:" << optimizer->vertices().size();
            std::cout << " Edge Size:" << optimizer->edges().size() << std::endl;
        }

    private:
        std::shared_ptr<g2o::SparseOptimizer> optimizer;

        PathNameSpace::KDTree_XYZRA *obsTree;

        std::map<size_t, KDTree_XYZRA *> groupPathTree_map;
        std::map<size_t, double> groupPathMaxRadius_map;

        void release() {
            delete obsTree;
        }
    };

};

#endif //MAPF_PIPELINE_G2OSMOOTHER_XYZ_H
