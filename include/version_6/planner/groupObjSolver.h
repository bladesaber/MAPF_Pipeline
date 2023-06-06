#ifndef MAPF_PIPELINE_GROUPOBJ_SOLVER_H
#define MAPF_PIPELINE_GROUPOBJ_SOLVER_H

#include "eigen3/Eigen/Core"

#include "common.h"
#include "AstarSolver.h"
#include "kdtree_xyzrl.h"

namespace PlannerNameSpace{

Path_XYZRL sampleDetailPath(Path_XYZR& path_xyzr, double stepLength);

class PathObjectInfo{
public:
    size_t pathIdx;

    size_t start_loc;
    std::vector<size_t> goal_locs;
    double radius;
    bool fixed_end;

    Path_XYZRL res_path;

    int getPathSize(){
        if (fixed_end==false){
            int size = res_path.size() - 1;
            return std::max(size, 0);
        }else{
            return res_path.size();
        }
    }

    PathObjectInfo(size_t pathIdx, bool fixed_end=false):pathIdx(pathIdx), fixed_end(fixed_end){};
    ~PathObjectInfo(){};
};

class MultiObjs_GroupSolver
{
public:
    MultiObjs_GroupSolver(){};
    ~MultiObjs_GroupSolver(){
        release();
    };

    std::vector<PathObjectInfo*> objectiveMap;

    void insert_objs(std::vector<size_t> locs, std::vector<double> radius_list, Instance& instance){
        std::vector<std::pair<size_t, size_t>> res = getSequence_miniumSpanningTree(instance, locs);

        int from_idx, to_idx;
        for (size_t pathIdx=0; pathIdx<res.size(); pathIdx++){
            std::tie(from_idx, to_idx) = res[pathIdx];

            PathObjectInfo* obj = new PathObjectInfo(pathIdx);
            obj->start_loc = locs[from_idx];
            obj->radius = radius_list[from_idx];

            if (pathIdx==0){
                std::vector<size_t> goal_locs;
                goal_locs.emplace_back(locs[to_idx]);
                obj->goal_locs = goal_locs;
                obj->fixed_end = true;
            }

            objectiveMap.emplace_back(obj);
        }
    }

    std::vector<std::pair<size_t, size_t>> getSequence_miniumSpanningTree(Instance& instance, std::vector<size_t> locs);

    bool findPath(AStarSolver* solver, std::vector<ConstrainType> constraints, Instance& instance, double stepLength){
        int from_loc, to_loc;
        double radius;

        Path_XYZR path_xyzr;
        double x, y, z;
        Path path;

        std::vector<size_t> locs_set;

        for (size_t i=0; i<objectiveMap.size(); i++){
            PathObjectInfo* obj = objectiveMap[i];
            
            path.clear();
            if (i>0){
                obj->goal_locs = std::vector<size_t>(locs_set);
            }
            path = solver->findPath(
                obj->radius, constraints, instance,
                obj->start_loc, obj->goal_locs
            );

            if (path.size() == 0){
                return false;
            }

            path_xyzr.clear();
            for (size_t loc : path)
            {
                locs_set.emplace_back(loc);
                std::tie(x, y, z) = instance.getCoordinate(loc);
                path_xyzr.emplace_back(std::make_tuple(x, y, z, obj->radius));
            }
            obj->res_path = sampleDetailPath(path_xyzr, stepLength);
        }

        updateLocTree();

        return true;
    }

    KDTree_XYZRL* locTree;
    bool setup_tree = false;

    void updateLocTree(){
        if (setup_tree){
            delete locTree;
        }

        locTree = new KDTree_XYZRL();
        double x, y, z, radius, length;
        for (PathObjectInfo* obj : objectiveMap)
        {           
            for (size_t i = 0; i < obj->getPathSize(); i++){
                std::tie(x, y, z, radius, length) = obj->res_path[i];
                locTree->insertNode(0, x, y, z, radius, length);
            }
        }
        setup_tree = true;
    }

    void copy(std::shared_ptr<MultiObjs_GroupSolver> rhs){
        for (PathObjectInfo* path : rhs->objectiveMap){
            PathObjectInfo* obj = new PathObjectInfo(path->pathIdx);
            obj->start_loc = path->start_loc;

            if (path->fixed_end){
                obj->goal_locs = path->goal_locs;
            }

            obj->radius = path->radius;
            // obj->res_path = Path_XYZRL(path->res_path);
        }
    }

private:
    void release(){
        for (size_t  i = 0; i < objectiveMap.size(); i++){
            delete objectiveMap[i];
        }
        objectiveMap.clear();

        if (setup_tree){
            delete locTree;
        }
    }

};

/*
class MasterSlave_GroupSet
{
public:
    std::vector<PathObjectInfo*> objectiveMap;

    MasterSlave_GroupSet(){};
    ~MasterSlave_GroupSet(){
        release();
    };

    void insert_MasterSlave(size_t pathIdx, size_t master_loc, size_t slave_loc, double radius){
        PathObjectInfo* obj = new PathObjectInfo(pathIdx);
        obj->start_loc = master_loc;
        
        std::vector<size_t> goal_locs;
        goal_locs.emplace_back(slave_loc);

        obj->goal_locs = goal_locs;
        obj->radius = radius;
        objectiveMap.emplace_back(obj);
    }

    bool findPath(AStarSolver* solver, std::vector<ConstrainType> constraints, Instance& instance, double stepLength){
        Path_XYZR path_xyzr;
        double x, y, z;

        for (size_t  i = 0; i < objectiveMap.size(); i++){
            PathObjectInfo* obj = objectiveMap[i];

            Path path = solver->findPath(
                obj->radius, constraints, instance,
                obj->start_loc, obj->goal_locs
            );

            if (path.size() == 0){
                return false;
            }

            path_xyzr.clear();
            for (size_t loc : path)
            {
                std::tie(x, y, z) = instance.getCoordinate(loc);
                path_xyzr.emplace_back(std::make_tuple(x, y, z, obj->radius));
            }
            obj->res_path = sampleDetailPath(instance, path_xyzr, stepLength);
        }
        return true;
    }

private:
    void release(){
        for (size_t  i = 0; i < objectiveMap.size(); i++){
            delete objectiveMap[i];
        }
        objectiveMap.clear();
    }

};
*/

}

#endif