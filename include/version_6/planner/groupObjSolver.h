#ifndef MAPF_PIPELINE_GROUPOBJ_SOLVER_H
#define MAPF_PIPELINE_GROUPOBJ_SOLVER_H

#include "eigen3/Eigen/Core"

#include "common.h"
#include "AstarSolver.h"
#include "kdtree_xyzra.h"

using namespace PathNameSpace;

namespace PlannerNameSpace{

Path_XYZR sampleDetailPath(Instance& instance, Path_XYZR& path_xyzr, double radius, double stepLength);

class PathObjectInfo{
public:
    size_t pathIdx;

    size_t start_loc;
    std::vector<size_t> goal_locs;
    double radius;

    Path_XYZR res_path;

    PathObjectInfo(size_t pathIdx):pathIdx(pathIdx){};
    ~PathObjectInfo(){};
};

class MultiObjs_GroupSet
{
public:
    MultiObjs_GroupSet(){};
    ~MultiObjs_GroupSet(){
        release();
    };

    std::vector<PathObjectInfo*> ObjectiveMap;

    void insert_objs(std::vector<size_t> locs, std::vector<double> radius_list, Instance& instance){
        std::vector<std::pair<size_t, size_t>> res = getSequence_miniumSpanningTree(instance, locs);

        int from_loc, to_loc;
        for (size_t pathIdx=0; pathIdx<res.size(); pathIdx++){
            std::tie(from_loc, to_loc) = res[pathIdx];

            PathObjectInfo* obj = new PathObjectInfo(pathIdx);
            obj->start_loc = from_loc;
            obj->radius = std::max(radius_list[from_loc], radius_list[to_loc]);

            if (pathIdx==0){
                std::vector<size_t> goal_locs;
                goal_locs.emplace_back(to_loc);
                obj->goal_locs = goal_locs;
            }

            ObjectiveMap.emplace_back(obj);
        }
    }

    std::vector<std::pair<size_t, size_t>> getSequence_miniumSpanningTree(
        Instance& instance, std::vector<size_t> locs
    );

    bool findPath(AStarSolver* solver, std::vector<ConstrainType> constraints, Instance& instance, double stepLength){
        int from_loc, to_loc;
        double radius;

        Path_XYZR path_xyzr;
        double x, y, z;
        Path path;

        std::vector<size_t> locs_set;

        for (size_t i=0; i<ObjectiveMap.size(); i++){
            PathObjectInfo* obj = ObjectiveMap[i];
            
            path.clear();
            if (i==0)
            {
                path = solver->findPath(
                    obj->radius, constraints, instance,
                    obj->start_loc, obj->goal_locs
                );

            }else{
                obj->goal_locs = std::vector<size_t>(locs_set);
                path = solver->findPath(
                    obj->radius, constraints, instance,
                    obj->start_loc, obj->goal_locs
                );
            }

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
            obj->res_path = sampleDetailPath(instance, path_xyzr, obj->radius, stepLength);
        }

        return true;
    }

private:
    // KDTree_XYZRA* locTree;

    void release(){
        for (size_t  i = 0; i < ObjectiveMap.size(); i++){
            delete ObjectiveMap[i];
        }
        ObjectiveMap.clear();
    }

};

class MasterSlave_GroupSet
{
public:
    std::vector<PathObjectInfo*> ObjectiveMap;

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
        ObjectiveMap.emplace_back(obj);
    }

    bool findPath(AStarSolver* solver, std::vector<ConstrainType> constraints, Instance& instance, double stepLength){
        Path_XYZR path_xyzr;
        double x, y, z;

        for (size_t  i = 0; i < ObjectiveMap.size(); i++){
            PathObjectInfo* obj = ObjectiveMap[i];

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
            obj->res_path = sampleDetailPath(instance, path_xyzr, obj->radius, stepLength);
        }
        return true;
    }

private:
    void release(){
        for (size_t  i = 0; i < ObjectiveMap.size(); i++){
            delete ObjectiveMap[i];
        }
        ObjectiveMap.clear();
    }

};

}

#endif