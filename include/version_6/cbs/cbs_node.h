#ifndef MAPF_PIPELINE_CBS_NODE_H
#define MAPF_PIPELINE_CBS_NODE_H

#include "groupObjSolver.h"
#include "kdtree_xyzra.h"

using namespace PlannerNameSpace;
using namespace PathNameSpace;

namespace CBSNameSpace{

class GroupAgentInfo{
public:
    std::shared_ptr<std::vector<ConstrainType>> constrains;
    std::shared_ptr<KDTree_XYZRA> pathTree;

}

class CBSNode{

}

}

#endif