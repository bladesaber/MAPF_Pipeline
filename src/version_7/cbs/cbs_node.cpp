#include "cbs_node.h"


namespace CBSNameSpace{

void CBSNode::findFirstPipeConflict(){
    int num_of_agentGroup = groupAgentMap.size();

    // Init Param
    isConflict = false;
    first_conflictDist = DBL_MAX;
    for (auto iter : pipe_conflictLength){
        iter.second = 0.0;
    }

    // Starting ......
    std::shared_ptr<SpanningTree_GroupSolver> groupAgent_i;
    std::shared_ptr<SpanningTree_GroupSolver> groupAgent_j;

    double x, y, z, radius, length, dist;
    KDTree_XYZRL_Res res;
    double conflict_length;

    for (size_t i=0; i<num_of_agentGroup; i++){
        groupAgent_i= groupAgentMap[i];

        for (size_t j=i+1; j<num_of_agentGroup; j++){
            groupAgent_j = groupAgentMap[j];

            conflict_length = 0.0;
            for (TaskInfo* task : groupAgent_i->task_seq)
            {
                for (size_t k=0; k < task->res_path.size(); k++){
                    std::tie(x, y, z, radius, length) = task->res_path[k];
                    groupAgent_j->locTree->nearest(x, y, z, res);

                    dist = norm2_distance(x, y, z, res.x, res.y, res.z);
                    if (dist >= radius + res.data->radius){
                        continue;
                    }

                    isConflict = true;
                    double min_length = std::min(length, res.data->length);
                    if (min_length < first_conflictDist){
                        first_conflictDist = min_length;
                        firstConflict = Conflict(
                            i, res.x, res.y, res.z, res.data->radius,
                            j, x, y, z, radius
                        );

                        // std::cout << "Conflict:" << std::endl;
                        // std::cout << "  GroupIdx:" << i << " x:" << x << " y:" << y << " z:" << z << " radius:" << radius << std::endl;
                        // std::cout << "  GroupIdx:" << j << " x:" << res.x << " y:" << res.y << " z:" << res.z << " radius:" << res.data->radius << std::endl;
                    }

                    conflict_length += stepLength;
                }
            }
        }
        pipe_conflictLength[i] = conflict_length;
    }
}

void CBSNode::compute_Heuristics(){
    h_val = 0.0;
    for (auto iter : pipe_conflictLength)
    {
        h_val += iter.second;
    }
}

void CBSNode::compute_Gval(){
    g_val = 0.0;

    double x, y ,z, radius, length;
    for (auto agentGroup_iter : groupAgentMap){
        for (auto task : agentGroup_iter.second->task_seq){
            std::tie(x, y, z, radius, length) = task->res_path.back();
            g_val += length;
        }
    }
}

}
