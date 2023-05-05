import numpy as np
from typing import Dict

from build import mapf_pipeline

from scripts.visulizer import VisulizerVista

class CBSSolver(object):
    def __init__(self, config, agentInfos: Dict):
        self.instance = mapf_pipeline.Instance(
            config['x'], config['y'], config['z']
        )
        self.instance.info()

        self.cbs_planner = mapf_pipeline.CBS()
        self.agentInfos = agentInfos

        # 由于CBSNode是由Python构造，必须由Python自己管理
        self.allNodes = {}
        self.node_id = 0

        self.agent_idxs = self.agentInfos.keys()
        self.num_of_agents = len(self.agent_idxs)

        # for agentIdx in self.agentInfos.keys():
        #     print('AgentIdx: %d' % agentIdx)
        #     for key in self.agentInfos[agentIdx]:
        #         print('   %s: %s' % (key, self.agentInfos[agentIdx][key]))

    def solve(self):
        print("Starting Solving ...")

        ### 1. init root cbsNode
        root = mapf_pipeline.CBSNode(self.num_of_agents)

        ### 1.1 init agents of root
        for agentIdx in self.agent_idxs:
            agentInfo = self.agentInfos[agentIdx]
            root.addAgent(
                agentIdx=agentInfo['agentIdx'],
                radius=agentInfo['radius'],
                startPos=agentInfo['startPos'],
                endPos=agentInfo['endPos'],
            )
            self.cbs_planner.addSearchEngine(agentInfo['agentIdx'],agentInfo['radius'])
        # self.print_NodeInfo(root)

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
        # self.print_NodeInfo(root, print_constrains=True)

        ## 1.3 compute all agent path
        for agentIdx in self.agent_idxs:
            success = self.cbs_planner.update_AgentPath(self.instance, root, agentIdx)
            print("AgentIdx:%d Solving Cost: %f success:%d" % (agentIdx, self.cbs_planner.runtime_search, success))

            if not success:
                print("[Debug]: Conflict Exist in Start Or End Pos")
                return False

        # self.print_NodeInfo(root, print_path=False, print_constrains=True)
        # self.print_NodeGraph(root)

        ### 1.4 find all the conflict and compute cost and heuristics
        root.depth = 0
        root.findAllAgentConflict()
        # self.print_NodeInfo(root, print_conflict=True)
        # self.print_NodeGraph(root)

        self.cbs_planner.compute_Heuristics(root)
        self.cbs_planner.compute_Gval(root)
        # print("[Debug]: Heuristics:%.1f Cost:%1f" % (root.h_val, root.g_val))

        # ### 1.6 push node into list
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

        ### ------ Just For Debug
        # if success_node is not None: 
        #     self.print_NodeInfo(success_node)
        #     self.print_NodeGraph(success_node)
        #     return True
        # 
        # else:
        #     print('Search Fail')
        #     return False
        
        return success_node

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
        select_conflict = None
        minConflict_length = np.inf

        for agentIdx in node.agentMap:
            agent = node.agentMap[agentIdx]
            min_length = agent.firstConflict.getMinLength()
            if min_length < minConflict_length:
                minConflict_length = min_length
                select_conflict = agent.firstConflict
        
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
            (select_conflict.agent1, select_conflict.constrain1), 
            (select_conflict.agent2, select_conflict.constrain2)
        ]
        childNodes = []
        for agentIdx, constrain in new_constrains:
            success, new_node = self.createCBSNode(
                node, agentIdx, constrain
            )

            if not success:
                continue

            # ### ------ Just For Debug
            # self.print_NodeInfo(new_node, print_constrains=True, print_path=True)
            # self.print_NodeGraph(new_node, select_agentIdx=agentIdx)
            # print('------------------------------------------------')
            # ### ---------------------------

            childNodes.append(new_node)

        return childNodes

    def createCBSNode(self, node, agentIdx, new_constrain):
        constrains = node.agentMap[agentIdx].getConstrains()
        constrains.append(new_constrain)

        childNode = mapf_pipeline.CBSNode(self.num_of_agents)
        childNode.copy(node)

        childNode.update_Constrains(agentIdx, constrains)

        success = self.cbs_planner.update_AgentPath(self.instance, childNode, agentIdx)
        if not success:
            return False, None

        childNode.depth = node.depth + 1
        childNode.findAllAgentConflict()

        self.cbs_planner.compute_Heuristics(childNode)
        self.cbs_planner.compute_Gval(childNode)

        return True, childNode

    def print_NodeInfo(
            self, node, 
            print_constrains=False,
            print_conflict=False,
            print_path=False,
        ):
        print("CBSNode:")
        print(" Depth:", node.depth)
        for agentIdx in node.agentMap:
            agent = node.agentMap[agentIdx]
            agent.info()

            if print_conflict:
                agent.firstConflict.info()

            if print_constrains:
                # print('Constrains:', agent.getConstrains())
                agent.debug()

            if print_path:
                print("Path: ")
                if agent.findPath_Success:
                    path = agent.getDetailPath()
                    path = np.array(path)
                    for xyzl in path:
                        print("   x: %.1f y:%.1f z:%.1f length:%.1f" % (xyzl[0], xyzl[1], xyzl[2], xyzl[3]))
                else:
                    print("   Fail")

    def print_NodeGraph(self, node, select_agentIdx=None):
        vis = VisulizerVista()

        random_colors = np.random.uniform(0.0, 1.0, size=(self.num_of_agents, 3))
        if select_agentIdx is not None:
            random_colors[select_agentIdx] = [1.0, 0.0, 0.0]

        for agentIdx in node.agentMap.keys():
            agent = node.agentMap[agentIdx]

            if agent.findPath_Success:
                path = agent.getDetailPath()
                path_xyz = np.array(path)[:, :3]

                startDire = np.array(self.agentInfos[agentIdx]['startDire'])
                padding_start = np.array([
                    path_xyz[0] + startDire * 3.,
                    path_xyz[0] + startDire * 2.,
                    path_xyz[0] + startDire * 1.,
                ])
                endDire = np.array(self.agentInfos[agentIdx]['endDire'])
                padding_end = np.array([
                    path_xyz[-1] + endDire * 1.,
                    path_xyz[0-1] + endDire * 2.,
                    path_xyz[0-1] + endDire * 3.,
                ])

                path_xyz = np.concatenate([
                    padding_start, path_xyz, padding_end
                ], axis=0)

                tube_mesh = vis.create_tube(path_xyz, radius=agent.radius)
                vis.plot(tube_mesh, color=tuple(random_colors[agentIdx]))

            vis.plot(
                vis.create_box(
                    np.array(self.agentInfos[agent.agentIdx]['startPos']),
                    length=agent.radius*2.0
                ), 
                color=tuple(random_colors[agentIdx])
            )
            vis.plot(
                vis.create_box(
                    np.array(self.agentInfos[agent.agentIdx]['endPos']),
                    length=agent.radius*2.0
                ), 
                color=tuple(random_colors[agentIdx])
            )

            # for conflict in agent.conflictSet:
            #     print(conflict)
            #     box = vis.create_box(np.array([conflict[0], conflict[1], conflict[2]]))
            #     vis.plot(box, color=(1.0, 0.0, 0.0))

        if select_agentIdx is not None:
            agent = node.agentMap[select_agentIdx]
            constrains = agent.getConstrains()

            for constrain in constrains:
                obs = vis.create_sphere(np.array([constrain[0], constrain[1], constrain[2]]), radius=constrain[3] + 0.1)
                vis.plot(obs, color=(0.0, 1.0, 0.0))

        vis.show()
