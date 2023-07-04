#ifndef MAPF_PIPELINE_SMOOTHER_XYZ_H
#define MAPF_PIPELINE_SMOOTHER_XYZ_H

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
#include "edge_boundary.h"

#include "groupPath.h"

using namespace PathNameSpace;

namespace SmootherNameSpace{

// 5 指 pose 是5纬度，-1指观测纬度未确认
// typedef g2o::BlockSolver<g2o::BlockSolverTraits<5, -1>>  BlockSolver;
// -1, -1 指未确认
typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>>  BlockSolver;

typedef g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType> LinearSolver;

class SmootherXYZG2O{
public:
    SmootherXYZG2O(){
        obsTree = new KDTree_XYZRA();
    };
    ~SmootherXYZG2O(){
        release();
    };

    // ------ G2O Edge Param
    double elasticBand_minLength = 0.3;
    double elasticBand_targetLength = -1.0;
    double elasticBand_kSpring = 1.0;
    double vertexKinematic_kSpring = 3.0;
    double edgeKinematic_kSpring = 10.0;
    // ---------------------

    std::map<size_t, GroupPath*> groupMap;
    double obstacle_detection_scale = 3.0;
    double pipeConflict_detection_scale = 1.5;

    void initOptimizer(){
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
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

        // set optimizer
        optimizer->setAlgorithm(solver);
        optimizer->initMultiThreading(); // required for >Eigen 3.1

        optimizer->setComputeBatchStatistics(true);
    }

    void add_Path(size_t groupIdx, Path_XYZR& path_xyzr){
        auto iter = groupMap.find(groupIdx);
        if (iter == groupMap.end()){
            GroupPath* new_groupPath = new GroupPath(groupIdx);
            new_groupPath->insertPath(path_xyzr);
            groupMap[new_groupPath->groupIdx] = new_groupPath;
        }
        else{
            groupMap[groupIdx]->insertPath(path_xyzr);
        }
    }

    bool add_OptimizePath(
        size_t groupIdx, size_t pathIdx, 
        double start_x, double start_y, double start_z,
        double end_x, double end_y, double end_z,
        std::pair<double, double> startDire, 
        std::pair<double, double> endDire,
        double startFlexRatio, double endFlexRatio
    ){
        return groupMap[groupIdx]->insert_OptimizePath(
            pathIdx, 
            start_x, start_y, start_z,
            end_x, end_y, end_z,
            startDire, endDire,
            startFlexRatio, endFlexRatio
        );
    }

    void setMaxRadius(size_t groupIdx, double radius){
        groupMap[groupIdx]->setMaxRadius(radius);
    }
    void setBoundary(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax){
        min_x = xmin;
        min_y = ymin;
        min_z = zmin;
        max_x = xmax;
        max_y = ymax;
        max_z = zmax;
    }
    std::vector<size_t> extractPath(
        size_t groupIdx,
        double start_x, double start_y, double start_z,
        double end_x, double end_y, double end_z
    ){
        return groupMap[groupIdx]->extractPath(start_x, start_y, start_z, end_x, end_y, end_z);
    }

    bool add_vertexs();
    bool add_elasticBand(double elasticBand_weight);
    bool add_kinematicEdge(double kinematic_weight);

    void insertStaticObs(double x, double y, double z, double radius, double alpha, double theta){
        obsTree->insertNode(0, x, y, z, radius, alpha, theta);
    }
    bool add_obstacleEdge(double obstacle_weight);

    bool add_pipeConflictEdge(double pipeConflict_weight);

    bool add_boundaryEdge(double boundary_weight);

    bool build_graph(
        double elasticBand_weight = 0.02, 
        double kinematic_weight = 1.0,
        double obstacle_weight=1.0,
        double pipeConflict_weight=1.0,
        double boundary_weight=5.0
    );

    void loss_report(
        size_t groupIdx,
        size_t pathIdx,
        double elasticBand_weight, 
        double kinematic_weight,
        double obstacle_weight,
        double pipeConflict_weight,
        double boundary_weight
    );

    void optimizeGraph(int no_iterations, bool verbose){
        optimizer->setVerbose(verbose);
        optimizer->initializeOptimization();

        int iter = optimizer->optimize(no_iterations);
        if(!iter)
        {
            std::cout << "optimizeGraph(): Optimization failed! iter=" << iter << std::endl;
            assert(false);
        }
    }

    void clear_graph(){
        if (optimizer){
            // we will delete all edges but keep the vertices.
            // before doing so, we will delete the link from the vertices to the edges.
            auto& vertices = optimizer->vertices();
            for(auto& v : vertices){
                v.second->edges().clear();
            }

            // necessary, because optimizer->clear deletes pointer-targets (therefore it deletes TEB states!)
            optimizer->vertices().clear();
            optimizer->clear();
        }
    }

    void update2groupVertex(){
        for (auto group_iter : groupMap)
        {
            GroupPath* group = group_iter.second;
            for (auto node_iter : group->graphNodeMap){
                (node_iter.second)->updateVertex();
            }
        }
    }

    void info(){
        std::cout << "Graph Vertex Size:" << optimizer->vertices().size();
        std::cout << " Edge Size:" << optimizer->edges().size() << std::endl;
    }

private:
  double min_x, min_y, min_z;
  double max_x, max_y, max_z;

    KDTree_XYZRA* obsTree;

    std::shared_ptr<g2o::SparseOptimizer> optimizer;

    void release(){
        delete obsTree;
    }
};

}

#endif