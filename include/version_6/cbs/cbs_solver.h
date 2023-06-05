#ifndef MAPF_PIPELINE_CBS_SOLVER_H
#define MAPF_PIPELINE_CBS_SOLVER_H

#include "cbs_node.h"
#include "AstarSolver.h"

namespace CBSNameSpace{

class CBSSolver{
public:
    CBSSolver(){}
    ~CBSSolver(){
        releaseNodes();
        releaseEngines();
    }

    bool isGoal(CBSNode* node);

    void pushNode(CBSNode* node){
        open_list.push(node);
    }

    CBSNode* popNode(){
        CBSNode* node = open_list.top();
        open_list.pop();
        return node;
    }

    bool is_openList_empty(){
        return this->open_list.empty();
    }

    void addSearchEngine(size_t groupIdx, bool with_AnyAngle, bool with_OrientCost){
        searchEngines[groupIdx] = new AStarSolver(with_AnyAngle, with_OrientCost);
    }

    bool update_GroupAgentPath(size_t groupIdx, CBSNode* node, Instance& instance){
        return node->update_GroupAgentPath(groupIdx, searchEngines[groupIdx], instance);
    }

private:
    std::map<size_t, AStarSolver*> searchEngines;

    boost::heap::pairing_heap< CBSNode*, boost::heap::compare<CBSNode::compare_node>> open_list;

    inline void releaseNodes(){
        open_list.clear();
    }

    inline void releaseEngines(){
        for (auto iter : searchEngines){
            iter.second->releaseNodes();
            delete iter.second;
        }
        searchEngines.clear();
    }

};

}

#endif