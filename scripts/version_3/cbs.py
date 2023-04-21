import numpy as np
from typing import Dict

from build import mapf_pipeline

class CBSSolver(object):
    def __init__(self, config, agentInfos: Dict):
        self.instance = mapf_pipeline.Instance(
            config['x'], config['y'], config['z']
        )

        self.cbs_planner = mapf_pipeline.CBS()
        self.agentInfos = agentInfos

        # 由于CBSNode是由Python构造，必须由Python自己管理
        self.allNodes = {}
        self.node_id = 0

        self.agent_idxs = self.agentInfos.keys()

    def solve(self):
        ### 1. init root cbsNode
        root = mapf_pipeline.CBSNode()

        ### 1.1 init agents of root
        for agentIdx in self.agent_idxs:
            agentInfo = self.agentInfos[agentIdx]
            agent = mapf_pipeline.AgentInfo(
                agentIdx=agentInfo['agentIdx'], 
                radius=agentInfo['radius']
            )
            root.setAgentInfo(agent.agentIdx, agent)

        ### 1.2 init agent constrains
        for a1 in self.agent_idxs:
            constrains = []

            for a2 in self.agent_idxs:
                if a1 != a2:
                    
                    constrains.append((
                        self.agentInfos[a2]['startPos'][0],
                        self.agentInfos[a2]['startPos'][1],
                        self.agentInfos[a2]['startPos'][2],
                        self.agentInfos[a2]['radius'],
                    ))

                    constrains.append((
                        self.agentInfos[a2]['endPos'][0],
                        self.agentInfos[a2]['endPos'][1],
                        self.agentInfos[a2]['endPos'][2],
                        self.agentInfos[a2]['radius'],
                    ))
            
            root.update_Constrains(a1, constrains)

        ### 1.3 compute all agent path
        for agentIdx in self.agent_idxs:
            self.cbs_planner.update_AgentPath(self.instance, root, agentIdx)

        ### 1.4 find all the conflict
        root.findAllAgentConflict()

        ### 1.5 compute cost and heuristics
        self.cbs_planner.compute_Heuristics(root)
        self.cbs_planner.compute_Gval(root)

        ### 1.6 push node into list
        self.cbs_planner.pushNode(root)

        while self.cbs_planner.is_openList_empty():
            pass
