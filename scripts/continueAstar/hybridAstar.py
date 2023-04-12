import numpy as np
import math

from scripts.continueAstar.instance import Instance
from scripts.continueAstar.constrainTable import ConstrainTable

from build import mapf_pipeline

class HybridAstarWrap(object):
    def __init__(
        self, 
        radius,
        instance:Instance, 
        constrainTable:ConstrainTable
    ):
        self.instance = instance
        self.constrainTable = constrainTable

        self.allNodes = {}
        self.hybridAstar = mapf_pipeline.HybridAstar()

        self.radius = radius

    def findPath(self, start_pos, goal_pose):
        self.startNode = mapf_pipeline.HybridAstarNode(
            x=start_pos[0], y=start_pos[1], z=start_pos[2],
            alpha=start_pos[3], beta=start_pos[4], 
            parent=None, in_openlist=False
        )
        startNode_round = self.instance.getRoundCoordinate(self.startNode.getCoodr())
        self.startNode.setRoundCoodr(startNode_round)

        self.goalNode = mapf_pipeline.HybridAstarNode(
            x=goal_pose[0], y=goal_pose[1], z=goal_pose[2],
            alpha=goal_pose[3], beta=goal_pose[4], 
            parent=None, in_openlist=False
        )
        goalNode_round = self.instance.getRoundCoordinate(self.goalNode.getCoodr())
        self.goalNode.setRoundCoodr(goalNode_round)

        self.startNode.h_val = 0.0
        self.startNode.g_val = 0.0

        self.hybridAstar.pushNode(self.startNode)
        self.allNodes[self.startNode.hashTag] = self.startNode      

        self.lower_bound = np.inf
        self.best_node = None

        while(not self.hybridAstar.is_openList_empty()):
            node = self.hybridAstar.popNode()

            if node.getFval() > self.lower_bound:
                continue

            if self.isGoal(node):
                self.best_node = node
                break

            coodr = node.getCoodr()
            for near_coodr in self.instance.getNeighbors(coodr):
                if not self.instance.isValidGrid(near_coodr):
                    continue

                if self.constrainTable.coodrIsConstrained(x=near_coodr[0], y=near_coodr[1], z=near_coodr[2]):
                    continue
                
                nearNode = mapf_pipeline.HybridAstarNode(
                    x=near_coodr[0], y=near_coodr[1], z=near_coodr[2],
                    alpha=near_coodr[3], beta=near_coodr[4], 
                    parent=node, in_openlist=False
                )
                nearNode.setRoundCoodr(self.instance.getRoundCoordinate(nearNode.getCoodr()))

                nearNode.h_val, (nearNode.dubins_solutions, nearNode.invert_yz) = self.getHeuristic(nearNode)
                if np.random.uniform(0.0, 1.0) > 0.65:
                    find_dubinsShot = self.test_dubinsShot(nearNode)
                    if find_dubinsShot:
                        self.lower_bound = nearNode.dubinsLength3D + nearNode.g_val
                        self.best_node = nearNode

                nearNode.g_val = nearNode.parent.g_val + self.instance.step_length

                hashTag = nearNode.hashTag
                if nearNode.hashTag not in self.allNodes.keys():
                    self.allNodes[hashTag] = nearNode
                    self.hybridAstar.pushNode(nearNode)
                
                else:
                    exit_node = self.allNodes[hashTag]
                    if exit_node.getFVal() <= nearNode.getFVal():
                        continue

                    exit_node.copy(nearNode)
                    self.hybridAstar.pushNode(exit_node)

        self.release()

        if self.best_node is None:
            return None

        path = self.updatePath(self.best_node)
        return path

    def test_dubinsShot(self, node:mapf_pipeline.HybridAstarNode):
        (res0, res1) = node.dubins_solutions
        invert_yz = node.invert_yz

        mapf_pipeline.compute_dubins_info(res0)
        path_xys = mapf_pipeline.sample_dubins_path(res0, 30)
        path_xys = np.array(path_xys)

        mapf_pipeline.compute_dubins_info(res1)
        path_xzs = mapf_pipeline.sample_dubins_path(res1, 30)
        path_xzs = np.array(path_xzs)

        path_xyzs = np.concatenate([
            path_xys, 
            path_xzs[:, 1:2]
        ], axis=1)
        if invert_yz:
            path_xyzs = np.concatenate([
                path_xyzs[:, 0:1], path_xyzs[:, 2:3], path_xyzs[:, 1:2]
            ], axis=1)

        if self.constrainTable.lineIsConstrained(path_xyzs, self.radius):
            return False
        
        total_dist = self.instance.compute_pathDist(path_xyzs)
        node.dubinsLength3D = total_dist

        dubins_list = []
        for xyz in path_xyzs:
            dubins_list.append((xyz[0], xyz[1], xyz[2]))
        node.dubinsPath3D = dubins_list
        
        return True

    def isValid(self, node:mapf_pipeline.HybridAstarNode):
        isValid = self.constrainTable.isConstrained(node)
        return isValid

    def getHeuristic(self, node:mapf_pipeline.HybridAstarNode):
        dist0 = self.instance.getEulerDistance(
            pos0 = (node.x, node.y, node.z),
            pos1 = (self.goal_node.x, self.goal_node.y, self.goal_node.z)
        )
        dist1, (solutions, invert_yz) = self.instance.getDubinsDistance(
            pos0 = (node.x, node.y, node.z, node.alpha, node.beta),
            pos1 = (self.goal_node.x, self.goal_node.y, self.goal_node.z, self.goal_node.alpha, self.goal_node.beta)
        )
        return max(dist0, dist1), (solutions, invert_yz)

    def updatePath(self, node:mapf_pipeline.HybridAstarNode):
        if not node.equal(self.goalNode):
            use_dubins = True
            dubinsPath3D = node.dubinsPath3D

        path = []
        while True:
            path.append([node.x, node.y, node.z])

            node = node.parent
            if node is None:
                break
        
        if use_dubins:
            path.extend(dubinsPath3D)
        
        return path

    def isGoal(self, node:mapf_pipeline.HybridAstarNode):
        return node.equal(self.goalNode)

    def release(self):
        self.hybridAstar.release()
        self.allNodes.clear()

    ### --------------- Just For Debug
    def findPath_init(self, start_pos, goal_pose):
        '''
        pos: (x, y, z, alpha, beta)
        '''
                
        self.release()

        self.startNode = mapf_pipeline.HybridAstarNode(
            x=start_pos[0], y=start_pos[1], z=start_pos[2], alpha=start_pos[3], beta=start_pos[4], 
            parent=None, in_openlist=False
        )
        startNode_round = self.instance.getRoundCoordinate(self.startNode.getCoodr())
        self.startNode.setRoundCoodr(startNode_round)

        self.goalNode = mapf_pipeline.HybridAstarNode(
            x=goal_pose[0], y=goal_pose[1], z=goal_pose[2], alpha=goal_pose[3], beta=goal_pose[4], 
            parent=None, in_openlist=False
        )
        goalNode_round = self.instance.getRoundCoordinate(self.goalNode.getCoodr())
        self.goalNode.setRoundCoodr(goalNode_round)

        self.startNode.h_val = 0.0
        self.startNode.g_val = 0.0
        
        self.hybridAstar.pushNode(self.startNode)
        self.allNodes[self.startNode.hashTag] = self.startNode      

        self.lower_bound = np.inf
        self.best_node = None
    
    def findPath_step(self):
        res = {
            'state': 'Error',
            'nodes': [],
            'cur_node': None
        }

        node = self.hybridAstar.popNode()
        res['cur_node'] = node

        if node.getFval() > self.lower_bound:
            res.update( {'state': 'skip'})
            return res

        if self.isGoal(node):
            self.best_node = node
            res.update( {'state': 'find_direct_goal'})
            return res
        
        coodr = node.getCoodr()
        for near_coodr in self.instance.getNeighbors(coodr):
            if not self.instance.isValidGrid(near_coodr):
                continue

            if self.constrainTable.coodrIsConstrained(x=near_coodr[0], y=near_coodr[1], z=near_coodr[2]):
                continue
                
            nearNode = mapf_pipeline.HybridAstarNode(
                x=near_coodr[0], y=near_coodr[1], z=near_coodr[2],
                alpha=near_coodr[3], beta=near_coodr[4], 
                parent=node, in_openlist=False
            )
            nearNode.setRoundCoodr(self.instance.getRoundCoordinate(nearNode.getCoodr()))

            nearNode.h_val, (nearNode.dubins_solutions, nearNode.invert_yz) = self.getHeuristic(nearNode)
            if np.random.uniform(0.0, 1.0) > 0.65:
                find_dubinsShot = self.test_dubinsShot(nearNode)
                if find_dubinsShot:
                    self.lower_bound = nearNode.dubinsLength3D + nearNode.g_val
                    self.best_node = nearNode

            nearNode.g_val = nearNode.parent.g_val + self.instance.step_length

            hashTag = nearNode.hashTag
            if nearNode.hashTag not in self.allNodes.keys():
                self.allNodes[hashTag] = nearNode
                self.hybridAstar.pushNode(nearNode)
                res['nodes'].append(nearNode)
                
            else:
                exit_node = self.allNodes[hashTag]
                if exit_node.getFVal() <= nearNode.getFVal():
                    continue

                exit_node.copy(nearNode)
                self.hybridAstar.pushNode(exit_node)
                res['nodes'].append(nearNode)

        res.update({'state': 'searching'})

        return res

if __name__ == '__main__':
    instance = Instance(40, 40, 40, radius=1.5, cell_size=1.0, horizon_discrete_num=24, vertical_discrete_num=12)

    ### --------- HybridAstarNode Debug
    # node0 = mapf_pipeline.HybridAstarNode(1.3, 2.3, 3.1, np.deg2rad(0.), np.deg2rad(0.), None, False)
    # node0.setRoundCoodr(instance.getRoundCoordinate(node0.getCoodr()))
    # print("Node0 x:%f y:%f z:%f alpha:%f beta:%f" % (node0.x, node0.y, node0.z, node0.alpha, node0.beta))
    # print("Node0 xRound:%d yRound:%d zRound:%d alphaRound:%d betaRound:%d" % (
    #     node0.x_round, node0.y_round, node0.z_round, node0.alpha_round, node0.beta_round
    # ))
    
    # node1 = mapf_pipeline.HybridAstarNode(1.6, 2., 3., np.deg2rad(0.), np.deg2rad(0.), None, False)
    # node1.setRoundCoodr(instance.getRoundCoordinate(node1.getCoodr()))
    # print("Node1 x:%f y:%f z:%f alpha:%f beta:%f" % (node1.x, node1.y, node1.z, node1.alpha, node1.beta))
    # print("Node1 xRound:%d yRound:%d zRound:%d alphaRound:%d betaRound:%d" % (
    #     node1.x_round, node1.y_round, node1.z_round, node1.alpha_round, node1.beta_round
    # ))

    # print(node0.equal(node1))
    ### ---------------------------------------------

    constrainTable = ConstrainTable()
    constrainTable.insert2CT(x=20, y=20, z=20, radius=1.0)
    constrainTable.update_numpy()

    model = HybridAstarWrap(
        radius=0.5, 
        instance=instance, 
        constrainTable=constrainTable,
        start_pos=(0.0, 10.0, 5.0, np.deg2rad(0.), np.deg2rad(0.)),
        goal_pose=(40.0, 20.0, 30.0, np.deg2rad(0.), np.deg2rad(0.)),
    )
    model.findPath()
