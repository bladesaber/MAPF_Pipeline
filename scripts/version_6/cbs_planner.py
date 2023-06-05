import numpy as np
import pandas as pd
from typing import Dict

from build import mapf_pipeline

class CBS_Planner(object):
    def __init__(self, config, instance):
        self.config = config
        self.instance = instance
        self.cbs_planner = mapf_pipeline.CBSSolver()

        # 由于CBSNode是由Python构造，必须由Python自己管理
        self.allNodes = {}
        self.node_id = 0
    
    def solve(self, staticObs_df:pd.DataFrame):
        print("Starting Solving ...")

        ### 1. init root cbsNode
        root = mapf_pipeline.CBSNode(self.config['stepLength'])

        ### 1.1 init agents of root
        groupInfos: Dict = self.config['group']
        for groupIdx in groupInfos.keys():
            groupInfo = groupInfos[groupIdx]

            locs, radius_list = [], []
            for objIdx in groupInfo.keys():
                objInfo = groupInfo[objIdx]
                locs.append(self.instance.linearizeCoordinate(objInfo['x'], objInfo['y'], objInfo['z']))
                radius_list.append(objInfo['radius'])
            
            root.add_GroupAgent(groupIdx, locs, radius_list, self.instance)
            self.cbs_planner.addSearchEngine(groupIdx, with_AnyAngle=False, with_OrientCost=True)
        
        ### 1.2 init agent constrains
        for groupIdx_i in groupInfos.keys():
            constrains = []

            for groupIdx_j in groupInfos.keys():
                if groupIdx_i == groupIdx_j:
                    continue
                
                for objIdx in groupInfo.keys():
                    objInfo = groupInfo[objIdx]
                    constrains.append(
                        objInfo['x'], objInfo['y'], objInfo['z'], objInfo['radius']
                    )
            
            for _, row in staticObs_df.iterrows():
                constrains.append((row.x, row.y, row.z, row.radius))

            root.update_Constrains(groupIdx_i, constrains)
        
        ## 1.3 compute all agent path
        for groupIdx_i in groupInfos.keys():
            # print('Solving Agent: %d' % agentIdx)
            success = self.cbs_planner.update_GroupAgentPath(groupIdx, root, self.instance)
            print("AgentIdx:%d Solving Cost: %f success:%d" % (groupIdx, self.cbs_planner.runtime_search, success))

            if not success:
                print("[Debug]: Conflict Exist in Start Or End Pos")
                return {'status': False}

        ### 1.4 find all the conflict and compute cost and heuristics
        root.depth = 0
        root.findFirstPipeConflict()
        root.compute_Heuristics()
        root.compute_Gval()

        ### 1.5 push node into list
        self.pushNode(root)

        run_times = 1
        success_node = None
        while not self.cbs_planner.is_openList_empty():
            node = self.popNode()

            if self.cbs_planner.isGoal(node):
                success_node = node
                break
                
            childNodes = self.extendNode(node)
            for child_node in childNodes:
                self.pushNode(child_node)

            run_times += 1
            if run_times > 300:
                print("[DEBUG]: Out of Resource !!!")
                break

            print("Running ... %d" % run_times)

    def pushNode(self, node):
        node.node_id = self.node_id
        self.node_id += 1
        self.cbs_planner.pushNode(node)
        self.allNodes[node.node_id] = node

    def popNode(self):
        node = self.cbs_planner.popNode()
        del self.allNodes[node.node_id]
        return node

    def extendNode(self, node):        
        select_conflict = node.firstConflict
        select_conflict.conflictExtend()

        # print('Constrain1: agentIdx:%d x:%.1f y:%.1f z:%.1f radius:%.1f' % (
        #     select_conflict.agent1, 
        #     select_conflict.conflict1_x, select_conflict.conflict1_y, select_conflict.conflict1_z, select_conflict.conflict1_radius
        # ))
        # print('Constrain2: agentIdx:%d x:%.1f y:%.1f z:%.1f radius:%.1f' % (
        #     select_conflict.agent2, 
        #     select_conflict.conflict2_x, select_conflict.conflict2_y, select_conflict.conflict2_z, select_conflict.conflict2_radius
        # ))

        new_constrains = [
            (select_conflict.groupIdx1, select_conflict.constrain1), 
            (select_conflict.groupIdx2, select_conflict.constrain2)
        ]
        childNodes = []
        for groupIdx, constrain in new_constrains:
            success, new_node = self.createCBSNode(node, groupIdx, constrain)

            if not success:
                continue

            # ### ------ Just For Debug
            # self.print_NodeInfo(new_node, print_constrains=True, print_path=True)
            # self.print_NodeGraph(new_node, select_agentIdx=agentIdx)
            # print('------------------------------------------------')
            # ### ---------------------------

            childNodes.append(new_node)

        return childNodes

    def createCBSNode(self, node, groupIdx, new_constrain):
        constrains = node.getConstrains(groupIdx)
        constrains.append(new_constrain)

        childNode = mapf_pipeline.CBSNode(self.config['stepLength'])
        childNode.copy(node)

        childNode.update_Constrains(groupIdx, constrains)

        success = self.cbs_planner.update_GroupAgentPath(groupIdx, node, self.instance)
        if not success:
            return False, None

        childNode.depth = node.depth + 1
        childNode.findFirstPipeConflict()
        childNode.compute_Heuristics()
        childNode.compute_Gval()

        return True, childNode