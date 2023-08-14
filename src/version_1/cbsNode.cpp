#include "cbsNode.h"

void CBSNode::findConflicts(){
    node_conflicts.clear();

    std::map<int, std::vector<std::pair<int, size_t>>> records;

    for (auto iter = paths.begin(); iter != paths.end(); iter++){
        Path path = iter->second;
        int a1 = iter->first;

        for (int i = 0; i < path.size(); i++){
            int loc = path[i];
            size_t a1_timeStep = i;

            if (records[loc].size() > 0)
            {
                for (auto i = records[loc].begin(); i != records[loc].end(); i++)
                {
                    int a2 = i->first;
                    size_t a2_timeStep = i->second;
                    node_conflicts.emplace_back(Conflict(a1, a2, a1_timeStep, a2_timeStep, loc));
                } 
            }
            
            records[loc].emplace_back(std::make_pair(a1, a1_timeStep));
        }
    }
}
