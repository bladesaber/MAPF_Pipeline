#include "cbs_solver.h"

namespace CBSNameSpace{

bool CBSSolver::isGoal(CBSNode* node){
    return node->isConflict;
}

}