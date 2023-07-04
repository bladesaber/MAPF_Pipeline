#ifndef MAPF_PIPELINE_SPANTree_GROUPSOLVER_H
#define MAPF_PIPELINE_SPANTree_GROUPSOLVER_H

#include "eigen3/Eigen/Core"
#include "math.h"

#include "common.h"
#include "AstarSolver.h"
#include "kdtree_xyzrl.h"

namespace PlannerNameSpace{

Path_XYZRL sampleDetailPath(Path_XYZR& path_xyzr, double stepLength);

class TaskInfo{
public:
    size_t link_sign0;
    size_t link_sign1;
    double radius0;
    double radius1;

    Path_XYZRL res_path;

    TaskInfo(){};
    ~TaskInfo(){};
};

class SpanningTree_GroupSolver
{
private:

struct TreeLeaf {
    TreeLeaf(){}
    TreeLeaf(size_t sign){
        relative_set.insert(sign);
    }
    ~TreeLeaf(){}

    std::set<size_t> relative_set;
    std::set<size_t> path;

    bool isSameSet(TreeLeaf* rhs){
        for (size_t sign: rhs->relative_set){
            if (relative_set.find(sign) != relative_set.end()){
                return true;
            }
        }
        return false;
    }

    void mergeBranch(TreeLeaf* rhs0, TreeLeaf* rhs1){
        for (size_t sign: rhs0->relative_set){
            relative_set.insert(sign);
        }
        for (size_t sign: rhs1->relative_set){
            relative_set.insert(sign);
        }
    }

    void mergePath(std::vector<size_t>& path0){
        for (size_t loc: path0){
            path.insert(loc);
        }
    }

    void mergePath(std::set<size_t>& path0){
        for (size_t loc: path0){
            path.insert(loc);
        }
    }
};

public:
    SpanningTree_GroupSolver(){};
    ~SpanningTree_GroupSolver(){
        // release();
    };

    std::vector<TaskInfo*> task_seq;
    std::vector<size_t> locations;

    std::vector<std::pair<size_t, size_t>> getSequence_miniumSpanningTree(Instance& instance, std::vector<size_t> locs);

    void insertPipe(std::map<size_t, double>& pipeMap, Instance& instance){
        locations.clear();
        for (auto iter: pipeMap){
            locations.emplace_back(iter.first);
        }

        std::vector<std::pair<size_t, size_t>> seq = getSequence_miniumSpanningTree(instance, locations);

        task_seq.clear();
        for (auto iter: seq){
            TaskInfo* task = new TaskInfo();
            task->link_sign0 = iter.first;
            task->link_sign1 = iter.second;
            task->radius0 = pipeMap[iter.first];
            task->radius1 = pipeMap[iter.second];
            task_seq.emplace_back(task);
        }
    }

    bool findPath(AStarSolver* solver, std::vector<ConstrainType> constraints, Instance& instance, double stepLength){
        std::map<size_t, TreeLeaf*> branchMap;
        for (size_t loc: locations){
            branchMap[loc] = new TreeLeaf(loc);
            branchMap[loc]->path.insert(loc);
            // std::cout << "branchMap loc:" << loc << std::endl;
        }

        // ------ Create Constrain Table
        ConstraintTable constrain_table = ConstraintTable();
        for (auto constraint : constraints){
            constrain_table.insert2CT(constraint);
        }

        Path path;
        Path_XYZR path_xyzr;
        size_t sign0, sign1;
        bool success = false;
        double x, y, z, radius;

        for (size_t i=0; i<task_seq.size(); i++){
            TaskInfo* task = task_seq[i];
            sign0 = task->link_sign0;
            sign1 = task->link_sign1;
            radius = task->radius0;
            
            std::vector<size_t> start_locs;
            start_locs.assign(branchMap[sign0]->path.begin(), branchMap[sign0]->path.end());
            std::vector<size_t> goal_locs;
            goal_locs.assign(branchMap[sign1]->path.begin(), branchMap[sign1]->path.end());

            // std::cout << "From:" << sign0 << " ->" << sign1 << std::endl;
            // std::cout << "start_locs size:" << start_locs.size() << " goal_locs:" << goal_locs.size() << std::endl;

            path.clear();
            path = solver->findPath(
                radius, // radius
                constrain_table, // constrain_table
                instance, // instance
                start_locs, // start_locs
                goal_locs // goal_locs
            );

            if (path.size() == 0){
                break;
                
            }else{
                path_xyzr.clear();
                for (size_t loc : path){
                    std::tie(x, y, z) = instance.getCoordinate(loc);
                    path_xyzr.emplace_back(std::make_tuple(x, y, z, radius));
                }
                // task->res_path.clear();
                task->res_path = sampleDetailPath(path_xyzr, stepLength);
            }

            TreeLeaf* treeLeaf = new TreeLeaf();
            treeLeaf->mergeBranch(branchMap[sign0], branchMap[sign1]);
            // ---------------  Method 1  --------------------
            // 这里可以尝试使用中间截断作为临时方法，这里没任何完备或近似最优保证，希望其能保持近似最优拓扑结构
            // if (branchMap[sign0]->path.size()>1){
            //     treeLeaf->mergePath(branchMap[sign0]->path);
            // }
            // if (branchMap[sign1]->path.size()>1){
            //     treeLeaf->mergePath(branchMap[sign1]->path);
            // }
            // int clip_num = ceil(path.size() * 0.1);
            // if ( path.size() - 2 * clip_num>0 ){
            //     std::vector<size_t> clip_path;
            //     for (size_t kk=clip_num; kk<path.size()-clip_num; kk++){
            //         clip_path.emplace_back(path[kk]);
            //     }
            //     treeLeaf->mergePath(clip_path);
            // }else{
            //     treeLeaf->mergePath(path);
            // }
            // ---------------  Method 2  --------------------
            // 我还是尝试使用 自由段 的方法来优化路经
            treeLeaf->mergePath(branchMap[sign0]->path);
            treeLeaf->mergePath(branchMap[sign1]->path);
            treeLeaf->mergePath(path);
            // -----------------------------------------------
            
            delete branchMap[sign0];
            delete branchMap[sign1];
            for (size_t sign: treeLeaf->relative_set){
                branchMap[sign] = treeLeaf;   
            }

            if ( i==task_seq.size()-1 ){
                success = true;
                delete treeLeaf;
            }
        }
        
        // 残留释放
        if (!success){
            std::set<size_t> release_set;
            for (auto iter: branchMap){
                if (release_set.find(iter.first) != release_set.end()){
                    continue;
                }

                for (size_t sign: iter.second->relative_set){
                    release_set.insert(sign);
                }
                delete iter.second;
            }
            release_set.clear();
            branchMap.clear();
        }

        updateLocTree();

       return success;
    }

    KDTree_XYZRL* locTree;
    bool setup_tree = false;

    void updateLocTree();

    void copy(std::shared_ptr<SpanningTree_GroupSolver> rhs, bool with_path=false);

private:
    void release(){
        for (size_t  i = 0; i < task_seq.size(); i++){
            delete task_seq[i];
        }
        task_seq.clear();

        if (setup_tree){
            delete locTree;
        }
    }
};

}

#endif