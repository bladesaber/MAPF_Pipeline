#ifndef MAPF_PIPELINE_CBS_NODE_H
#define MAPF_PIPELINE_CBS_NODE_H

#include "kdtree_xyzrl.h"
#include "conflict.h"
#include "spanningTree_groupSolver.h"

using namespace PlannerNameSpace;

namespace CBSNameSpace{

class CBSNode{
public:
    CBSNode(double stepLength):stepLength(stepLength){};
    ~CBSNode(){
        release();
    }

    std::map<size_t, std::shared_ptr<SpanningTree_GroupSolver>> groupAgentMap;
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
        std::shared_ptr<SpanningTree_GroupSolver> groupAgent = std::make_shared<SpanningTree_GroupSolver>();
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

    void add_GroupAgent(size_t groupIdx, std::map<size_t, double>& pipeMap, Instance& instance){
        std::shared_ptr<SpanningTree_GroupSolver> groupAgent = std::make_shared<SpanningTree_GroupSolver>();
        groupAgent->insertPipe(pipeMap, instance);

        groupAgentMap[groupIdx] = nullptr;
        groupAgentMap[groupIdx] = groupAgent;
    }

    void copy(CBSNode* rhs){
        for (auto iter : rhs->groupAgentMap){
            groupAgentMap[iter.first] = std::shared_ptr<SpanningTree_GroupSolver>(iter.second);
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

    std::vector<TaskInfo> getGroupAgent(size_t groupIdx){
        std::vector<TaskInfo> resList;
        for (TaskInfo* task : groupAgentMap[groupIdx]->task_seq){
            TaskInfo new_obj = TaskInfo();
            // new_obj.link_sign0 = task->link_sign0;
            // new_obj.link_sign1 = task->link_sign1;
            new_obj.radius0 = task->radius0;
            new_obj.radius1 = task->radius1;
            new_obj.res_path = Path_XYZRL(task->res_path);
            resList.emplace_back(new_obj);
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
                for (auto task: groupAgent_iter.second->task_seq){
                    std::cout << "    radius0:" << task->radius0 << std::endl;
                    std::cout << "    radius1:" << task->radius1 << std::endl;
                    std::cout << "    res_path size:" << task->res_path.size() << std::endl;
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