import numpy as np
import pandas as pd
from typing import Dict

from build import mapf_pipeline

class SmoothSolver(object):
    def __init__(self, agentInfos:Dict):
        self.agentInfos = agentInfos

    def compute_normal_path(self, stepLength):
        solver = mapf_pipeline.SmootherXYZG2O()

        for agentIdx in self.agentInfos.keys():
            agentInfo = self.agentInfos[agentIdx]
            
            path_xyzr = []
            for (x, y, z, length) in agentInfo['detail_path']:
                path_xyzr.append((x, y, z, agentInfo['radius']))
            
            self.agentInfos[agentIdx]['normal_path'] = solver.detailSamplePath(path_xyzr, stepLength)

    def solve(self, obs_df:pd.DataFrame, outer_loop=10, inner_loop=1):
        solver = mapf_pipeline.SmootherXYZG2O()
        
        count_dict = {}
        for agentIdx in self.agentInfos.keys():
            agentInfo = self.agentInfos[agentIdx]
            groupIdx = agentInfo['groupIdx']

            if groupIdx not in count_dict.keys():
                count_dict[groupIdx] = 0
            else:
                count_dict[groupIdx] += 1
            pathIdx = count_dict[groupIdx]
            self.agentInfos[agentIdx]['pathIdx'] = pathIdx

            solver.addPath(
                groupIdx = agentInfo['groupIdx'], 
                pathIdx = agentInfo['pathIdx'], 
                path_xyzr = agentInfo['normal_path'], 
                startDire = agentInfo['startDire'], 
                endDire = agentInfo['endDire']
            )
        
        for _, row in obs_df.iterrows():
            solver.insertStaticObs(row.x, row.y, row.z, row.radius, row.alpha, row.theta)

        solver.initOptimizer()
        for outer_i in range(outer_loop):

            ### Step 2.1 Build Graph 
            solver.build_graph(
                # elasticBand_weight=elasticBand_weight,
                # kinematic_weight=kinematic_weight,
                # obstacle_weight=obstacle_weight,
                # pipeConflict_weight=pipeConflict_weight
            )

            if outer_i == 0:
                solver.info()
                # solver.loss_info(
                #     elasticBand_weight=elasticBand_weight,
                #     kinematic_weight=kinematic_weight,
                #     obstacle_weight=obstacle_weight,
                #     pipeConflict_weight=pipeConflict_weight
                # )
        
            ### Step 2.2 Optimize
            solver.optimizeGraph(inner_loop, False)

            ### Step 3.3 Update Vertex to Node
            solver.update2groupVertex()

            # print('------------------------------')
            # solver.loss_info(
            #     elasticBand_weight=elasticBand_weight,
            #     kinematic_weight=kinematic_weight,
            #     obstacle_weight=obstacle_weight,
            #     pipeConflict_weight=pipeConflict_weight
            # )

            # ### Step 2.4 Clear Graph
            solver.clear_graph()

        for group_key in solver.groupMap.keys():
            groupPath = solver.groupMap[group_key]
            for pathIdx in groupPath.pathIdxs_set:
                nodeIdxs_path = groupPath.extractPath(pathIdx)

                path_xyzr = []
                for _, nodeIdx in enumerate(nodeIdxs_path):
                    node = groupPath.nodeMap[nodeIdx]
                    path_xyzr.append([node.x, node.y, node.z, node.radius])
                
                for agentIdx in self.agentInfos.keys():
                    agentInfo = self.agentInfos[agentIdx]
                    groupIdx = agentInfo['groupIdx']

                    if (groupIdx == group_key) and (agentInfo['pathIdx'] == pathIdx):
                        self.agentInfos[agentIdx]['smooth_path'] = np.array(path_xyzr)

        return self.agentInfos
