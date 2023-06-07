#ifndef MAPF_PIPELINE_CBS_NODE_H
#define MAPF_PIPELINE_CBS_NODE_H

#include "groupObjSolver.h"
#include "kdtree_xyzrl.h"
#include "conflict.h"

using namespace PlannerNameSpace;

namespace CBSNameSpace{

class CBSNode{
public:
    CBSNode(double stepLength):stepLength(stepLength){};
    ~CBSNode(){
        release();
    }

    std::map<size_t, std::shared_ptr<MultiObjs_GroupSolver>> groupAgentMap;
    std::map<size_t, std::shared_ptr<std::vector<ConstrainType>>> constrainsMap;
    std::map<size_t, double> pipe_conflictLength;
    
    double stepLength;
    double first_conflictDist = DBL_MAX;
    Conflict firstConflict;
    bool isConflict = false;

    size_t node_id = 0;
    double g_val = 0.0;
	double h_val = 0.0;
	int depth = 0;

    void update_Constrains(size_t groupIdx, const std::vector<ConstrainType>& new_constrains){
        constrainsMap[groupIdx] = nullptr;
        constrainsMap[groupIdx] = std::make_shared<std::vector<ConstrainType>>(new_constrains);
    }

    bool update_GroupAgentPath(size_t groupIdx, AStarSolver* solver, Instance& instance){
        std::shared_ptr<MultiObjs_GroupSolver> groupAgent = std::make_shared<MultiObjs_GroupSolver>();
        groupAgent->copy(groupAgentMap[groupIdx]);
        groupAgentMap[groupIdx] = nullptr;

        bool success = groupAgent->findPath(
            solver, 
            *(constrainsMap[groupIdx]),
            instance,
            stepLength
        );
        if ( success==false ){
            return false;
        }

        groupAgentMap[groupIdx] = groupAgent;
        return true;
    }

    void add_GroupAgent(size_t groupIdx, std::vector<size_t> locs, std::vector<double> radius_list, Instance& instance){
        std::shared_ptr<MultiObjs_GroupSolver> groupAgent = std::make_shared<MultiObjs_GroupSolver>();
        groupAgent->insert_objs(locs, radius_list, instance);

        groupAgentMap[groupIdx] = nullptr;
        groupAgentMap[groupIdx] = groupAgent;
    }

    void copy(CBSNode* rhs){
        for (auto iter : rhs->groupAgentMap){
            groupAgentMap[iter.first] = std::shared_ptr<MultiObjs_GroupSolver>(iter.second);
        }
        for (auto iter : rhs->constrainsMap){
            constrainsMap[iter.first] = std::shared_ptr<std::vector<ConstrainType>>(iter.second);
        }
    }

    inline double getFVal() const{
        return g_val + h_val;
    }

    void findFirstPipeConflict();

    void compute_Heuristics();
    void compute_Gval();
    
    std::vector<ConstrainType> getConstrains(size_t groupIdx){
        int useCount = constrainsMap[groupIdx].use_count();
        assert(("Constrain Must Be Exist", useCount > 0));

        return std::vector<ConstrainType>(*(constrainsMap[groupIdx]));
        // return *(constrainsMap[groupIdx]);
    }

    std::vector<PathObjectInfo> getGroupAgent(size_t groupIdx){
        std::vector<PathObjectInfo> resList;
        for (PathObjectInfo* path : groupAgentMap[groupIdx]->objectiveMap){
            PathObjectInfo obj = PathObjectInfo(path->pathIdx);
            obj.start_loc = path->start_loc;
            obj.fixed_end = path->fixed_end;
            obj.goal_locs = path->goal_locs;
            obj.radius = path->radius;
            obj.res_path = Path_XYZRL(path->res_path);
            resList.emplace_back(obj);
        }
        return resList;
    }

    void info(bool with_constrainInfo=false, bool with_pathInfo=false){
        for (auto groupAgent_iter: groupAgentMap){
            std::cout << "GroupIdx:" << groupAgent_iter.first << std::endl;
            
            if (with_constrainInfo){
                std::cout << "  constrain size:" << constrainsMap[groupAgent_iter.first]->size() << std::endl;
            }
            
            if (with_pathInfo){
                for (auto obj: groupAgent_iter.second->objectiveMap){
                    std::cout << "  PathIdx:" << obj->pathIdx << std::endl;
                    std::cout << "    start_loc:" << obj->start_loc << std::endl;
                    std::cout << "    radius:" << obj->radius << std::endl;
                    std::cout << "    fixed_end:" << obj->fixed_end << std::endl;
                    std::cout << "    goal_locs size:" << obj->goal_locs.size() << std::endl;
                    std::cout << "    res_path size:" << obj->res_path.size() << std::endl;
                }
            }
        }
    }

    struct compare_node 
    {
		bool operator()(const CBSNode* n1, const CBSNode* n2) const 
		{
			return n1->g_val + n1->h_val >= n2->g_val + n2->h_val;
		}
	};

private:
    void release(){
        for (auto iter : groupAgentMap){
            iter.second = nullptr;
        }
        groupAgentMap.clear();

        for (auto iter : constrainsMap){
            iter.second = nullptr;
        }
        constrainsMap.clear();
        pipe_conflictLength.clear();
    }
};

}

#endif