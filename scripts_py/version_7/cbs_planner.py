import numpy as np
import pandas as pd
from typing import Dict
from scripts_py.visulizer import VisulizerVista

from build import mapf_pipeline

class CBS_Planner(object):
    def __init__(self, config, instance, staticObs_df:pd.DataFrame):
        self.config = config
        self.instance = instance
        self.staticObs_df = staticObs_df
        self.stepLength = self.config['stepLength']

        self.cbs_planner = mapf_pipeline.CBSSolver()

        # 由于CBSNode是由Python构造，必须由Python自己管理
        self.allNodes = {}
        self.node_id = 0

        self.groupConfig = {}
        self.group_keys = []
        self.group_radius = {}
        for groupPipe in self.config['pipeConfig']:
            self.group_keys.append(groupPipe['groupIdx'])
            self.groupConfig[groupPipe['groupIdx']] = groupPipe['pipe']
            self.group_radius[groupPipe['groupIdx']] = groupPipe['grid_radius']

        self.record_cfg = {}
    
    def solve(self):
        print("Init Setting")
        for groupIdx in self.group_keys:
            print('groupIdx: %d' % groupIdx)
            for objInfo in self.groupConfig[groupIdx]:
                print('  x:%d y:%d z:%d radius:%f' % (
                    objInfo['grid_position'][0], objInfo['grid_position'][1], objInfo['grid_position'][2], 
                    self.group_radius[groupIdx]
                ))

        print("Starting Solving ...")

        ### 1. init root cbsNode
        root = mapf_pipeline.CBSNode(self.config['stepLength'])
        
        ### 1.1 init agents of root
        for groupIdx in self.group_keys:
            pipeMap = {}
            for objInfo in self.groupConfig[groupIdx]:
                pipeMap[
                    self.instance.linearizeCoordinate(
                        objInfo['grid_position'][0], 
                        objInfo['grid_position'][1], 
                        objInfo['grid_position'][2]
                    )] = self.group_radius[groupIdx] # replace objInfo['grid_radius']
            root.add_GroupAgent(groupIdx, pipeMap, self.instance)

            self.cbs_planner.addSearchEngine(groupIdx, with_AnyAngle=False, with_OrientCost=True)
        # root.info()

        ### 1.2 init agent constrains
        for groupIdx_i in self.group_keys:
            constrains = []

            for groupIdx_j in self.group_keys:
                if groupIdx_i == groupIdx_j:
                    continue
                
                for objInfo in self.groupConfig[groupIdx_j]:
                    constrains.append((
                        objInfo['grid_position'][0], 
                        objInfo['grid_position'][1], 
                        objInfo['grid_position'][2], 
                        self.group_radius[groupIdx_j] # replace objInfo['grid_radius']
                    ))
            
            for _, row in self.staticObs_df.iterrows():
                constrains.append((row.x, row.y, row.z, row.radius))

            root.update_Constrains(groupIdx_i, constrains)
        # root.info(with_constrainInfo=True)

        ### 1.3 compute all agent path
        for groupIdx in self.group_keys:
            # print('Solving GroupIdx: %d' % groupIdx)
            success = self.cbs_planner.update_GroupAgentPath(groupIdx, root, self.instance)
            print("groupIdx:%d success:%d" % (groupIdx, success))

            if not success:
                print("[Debug]: Conflict Exist in Start Or End Pos")
                return {'status': False}

        ### 1.4 find all the conflict and compute cost and heuristics
        root.depth = 0
        root.findFirstPipeConflict()
        root.compute_Heuristics()
        root.compute_Gval()
        # self.print_pathGraph(root, groupIdx=2)
        # self.print_pathGraph(root)

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
            if run_times > 1000:
                print("[DEBUG]: Out of Resource !!!")
                break

            print("Running ... %d" % run_times)

        if success_node is not None:
            self.print_pathGraph(success_node)

            return {
                'status': True,
                'res': self.extractPath(success_node)
            }

        else:
            print('[DEBUG]: Fail Find Any Solution')
            return {
                'status': False
            }
        
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

        # print('Constrain1: groupIdx:%d x:%.1f y:%.1f z:%.1f radius:%.1f' % (
        #     select_conflict.groupIdx1, 
        #     select_conflict.conflict1_x, select_conflict.conflict1_y, select_conflict.conflict1_z, select_conflict.conflict1_radius
        # ))
        # print('Constrain2: groupIdx:%d x:%.1f y:%.1f z:%.1f radius:%.1f' % (
        #     select_conflict.groupIdx2, 
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

            childNodes.append(new_node)

        return childNodes

    def createCBSNode(self, node, groupIdx, new_constrain):
        constrains = node.getConstrains(groupIdx)
        constrains.append(new_constrain)

        childNode = mapf_pipeline.CBSNode(self.config['stepLength'])
        childNode.copy(node)

        childNode.update_Constrains(groupIdx, constrains)

        success = self.cbs_planner.update_GroupAgentPath(groupIdx, childNode, self.instance)
        if not success:
            return False, None

        childNode.depth = node.depth + 1
        childNode.findFirstPipeConflict()
        childNode.compute_Heuristics()
        childNode.compute_Gval()

        # self.print_nodeGraph(node, groupIdx)
        
        return True, childNode
    
    def print_pathGraph(self, node, groupIdx=None):
        vis = VisulizerVista()

        random_colors = np.random.uniform(0.0, 1.0, size=(len(self.group_keys), 3))
        if groupIdx is not None:
            random_colors[groupIdx] = [1.0, 0.0, 0.0]

        for groupIdx in self.group_keys:
            obj_list = node.getGroupAgent(groupIdx)
            for obj in obj_list:
                path_xyzrl = np.array(obj.res_path)
                if path_xyzrl.shape[0] > 0:
                    tube_mesh = vis.create_tube(path_xyzrl[:, :3], radius=self.group_radius[groupIdx])
                    vis.plot(tube_mesh, color=tuple(random_colors[groupIdx]))

        obstacle_mesh = vis.create_pointCloud(self.staticObs_df[['x', 'y', 'z']].values)
        vis.plot(obstacle_mesh, (0.0, 1.0, 0.0))

        if node.isConflict:
            conflict = node.firstConflict

            # print('[DEBUG]: Insert Conflict1 groupIdx:%d x:%.1f y:%.1f z:%.1f radius:%.1f' % (
            #     conflict.groupIdx1, conflict.conflict1_x, conflict.conflict1_y, conflict.conflict1_z, conflict.conflict1_radius
            # ))
            conflict1_mesh = vis.create_sphere(
                np.array([conflict.conflict1_x, conflict.conflict1_y, conflict.conflict1_z]), conflict.conflict1_radius
            )
            vis.plot(conflict1_mesh, (0.0, 1.0, 0.0))

            # print('[DEBUG]: Insert Conflict2 groupIdx:%d x:%.1f y:%.1f z:%.1f radius:%.1f' % (
            #     conflict.groupIdx2, conflict.conflict2_x, conflict.conflict2_y, conflict.conflict2_z, conflict.conflict2_radius
            # ))
            # conflict2_mesh = vis.create_sphere(
            #     np.array([conflict.conflict2_x, conflict.conflict2_y, conflict.conflict2_z]), conflict.conflict2_radius
            # )
            # vis.plot(conflict2_mesh, (0.0, 1.0, 0.0))

        vis.show()

    def print_nodeGraph(self, node, groupIdx):
        vis = VisulizerVista()

        obj_list = node.getGroupAgent(groupIdx)
        for obj in obj_list:
            path_xyzrl = np.array(obj.res_path)
            if path_xyzrl.shape[0] > 0:
                tube_mesh = vis.create_tube(path_xyzrl[:, :3], radius=self.group_radius[groupIdx])
                vis.plot(tube_mesh, color=(1.0, 0.0, 0.0))
        
        constrains = node.getConstrains(groupIdx)
        constrains_np = np.array(constrains)

        static_constrain = constrains_np[constrains_np[:, 3] == 0]
        obstacle_mesh = vis.create_pointCloud(static_constrain[:, :3])
        vis.plot(obstacle_mesh, (0.0, 1.0, 0.0))

        dynamic_constrain = constrains_np[constrains_np[:, 3] != 0]
        dynamic_constrain = dynamic_constrain.reshape((-1, 4))
        for (x, y, z, r) in dynamic_constrain:
            mesh = vis.create_sphere(np.array([x, y, z]), r)
            vis.plot(mesh, (0.0, 1.0, 0.0))

        vis.show()
    
    def extractPath(self, node):
        groupAgent = {}
        for groupIdx in self.group_keys:
            obj_list = node.getGroupAgent(groupIdx)

            groupAgent[groupIdx] = []
            for obj in obj_list:
                path_xyzrl = np.array(obj.res_path)
                if path_xyzrl.shape[0] > 0:
                    groupAgent[groupIdx].append(path_xyzrl)
        return groupAgent
    