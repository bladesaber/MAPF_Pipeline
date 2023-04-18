import numpy as np
import math
import time

from scripts.continueAstar.instance import Instance
from scripts.continueAstar.constrainTable import ConstrainTable
from scripts.continueAstar.dubinsCruve import compute_dubinsPath3D

# from scripts.visulizer import VisulizerVista
from scripts.visulizer import VisulizerO3D

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
        self.searching_time = None

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

        start_time = time.time()

        best_node = None
        while(not self.hybridAstar.is_openList_empty()):
            node = self.hybridAstar.popNode()

            self.hybridAstar.num_expanded += 1

            ### ------ debug print
            # self.printNode(node)
            ### ------------

            if self.isGoal(node):
                best_node = node
                break

            coodr = node.getCoodr()
            for near_coodr in self.instance.getNeighbors(coodr):
                if not self.instance.isValidGrid(near_coodr):
                    continue

                # if self.constrainTable.coodrIsConstrained(x=near_coodr[0], y=near_coodr[1], z=near_coodr[2], radius=self.radius):
                #     continue
                
                nearNode = mapf_pipeline.HybridAstarNode(
                    x=near_coodr[0], y=near_coodr[1], z=near_coodr[2],
                    alpha=near_coodr[3], beta=near_coodr[4], 
                    parent=node, in_openlist=False
                )
                nearNode.setRoundCoodr(self.instance.getRoundCoordinate(nearNode.getCoodr()))
                nearNode.parentTag = node.hashTag

                nearNode.h_val = self.getHeuristic(nearNode)
                nearNode.g_val = nearNode.parent.g_val + self.instance.step_length

                ### ------ debug print
                # self.printNode(nearNode)
                ### ------------

                hashTag = nearNode.hashTag
                if nearNode.hashTag not in self.allNodes.keys():
                    self.hybridAstar.pushNode(nearNode)
                    self.allNodes[hashTag] = nearNode
                
                else:
                    exit_node = self.allNodes[hashTag]
                    if exit_node.getFVal() <= nearNode.getFVal():
                        continue

                    exit_node.copy(nearNode)
                    self.hybridAstar.pushNode(exit_node)

        self.searching_time = time.time() - start_time

        path = None
        if best_node is not None:
            path = self.updatePath(best_node)
        
        self.release()

        return path

    def isValid(self, node:mapf_pipeline.HybridAstarNode):
        isValid = self.constrainTable.isConstrained(node)
        return isValid

    def getHeuristic(self, node:mapf_pipeline.HybridAstarNode):
        dist0 = self.instance.getEulerDistance(
            pos0 = np.array(node.getCoodr()),
            pos1 = np.array(self.goalNode.getCoodr())
        )

        ### dubins启发式不适合在局部空间使用，因为跨出局部空间的最短路径不是有效评估
        # dist1, (solutions, invert_yz) = self.instance.getDubinsDistance(
        #     pos0 = np.array(node.getCoodr()),
        #     pos1 = np.array(np.array(self.goalNode.getCoodr()))
        # )
        # return max(dist0, dist1), (solutions, invert_yz)

        dist1 = self.instance.getThetaDistance(
            pos0 = np.array(node.getCoodr()),
            pos1 = np.array(self.goalNode.getCoodr())
        )
        return dist0 + dist1

    def updatePath(self, node:mapf_pipeline.HybridAstarNode):
        path = []
        while True:
            path.append([node.x, node.y, node.z, node.alpha, node.beta])
            # print('x:%.2f y:%.2f z:%.2f parent_tag:%s' % (node.x, node.y, node.z, node.parentTag))

            node = node.parent
            if node is None:
                break
        
        return path[::-1]

    def isGoal(self, node:mapf_pipeline.HybridAstarNode):
        return node.equal(self.goalNode)

    def release(self):
        self.hybridAstar.release()
        self.allNodes.clear()

    def printNode(self, node):
        print("nearNode x:%f y:%f z:%f alpha:%f beta:%f" % (
            node.x, node.y, node.z, np.rad2deg(node.alpha), np.rad2deg(node.beta)
        ))
        print("nearNode xRound:%d yRound:%d zRound:%d alphaRound:%d betaRound:%d" % (
            node.x_round, node.y_round, node.z_round, 
            node.alpha_round, node.beta_round
        ))
        print("nearNode Tag: %s" % (node.hashTag))

    ### --------------- Just For Debug
    '''
    def findPath_init(self, start_pos, goal_pose):
        ### pos: (x, y, z, alpha, beta)
                
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

        self.best_node = None

    def findPath_step(self):
        node = self.hybridAstar.popNode()
        res = {
            'state': 'Error',
            'nodes': [],
            'cur_node': node
        }

        if self.isGoal(node):
            self.best_node = node
            res.update({'state': 'find_direct_goal'})
            return res
        
        coodr = node.getCoodr()
        for near_coodr in self.instance.getNeighbors(coodr):
            
            if not self.instance.isValidGrid(near_coodr):
                continue
            
            if self.constrainTable.coodrIsConstrained(x=near_coodr[0], y=near_coodr[1], z=near_coodr[2], radius=self.radius):
                continue
            
            nearNode = mapf_pipeline.HybridAstarNode(
                x=near_coodr[0], y=near_coodr[1], z=near_coodr[2],
                alpha=near_coodr[3], beta=near_coodr[4], 
                parent=node, in_openlist=False
            )
            nearNode.setRoundCoodr(self.instance.getRoundCoordinate(nearNode.getCoodr()))

            nearNode.h_val = self.getHeuristic(nearNode)
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

    def compute_dubinsPath(self, node):
        (res0, res1), cost, invert_yz = compute_dubinsPath3D(
            np.array(node.getCoodr()), 
            np.array(self.goalNode.getCoodr()), 
            self.radius
        )

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

        dubins_list = []
        for xyz in path_xyzs:
            dubins_list.append((xyz[0], xyz[1], xyz[2]))
        node.dubinsPath3D = dubins_list

        if self.constrainTable.lineIsConstrained(path_xyzs, self.radius):
            node.findValidDubinsPath = False
            return False
        
        if not self.instance.lineIsValidGrid(path_xyzs):
            node.findValidDubinsPath = False
            return False

        node.findValidDubinsPath = True
        total_dist = self.instance.compute_pathDist(path_xyzs)
        node.dubinsLength3D = total_dist

        return True

    '''
    
if __name__ == '__main__':
    # instance = Instance(10, 10, 10, radius=1.5, cell_size=1.0, horizon_discrete_num=24, vertical_discrete_num=12)

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

    instance = Instance(15, 15, 15, radius=1.5, cell_size=1.0, horizon_discrete_num=24, vertical_discrete_num=12)

    constrainTable = ConstrainTable()
    constrainTable.insert2CT(x=1., y=1., z=1., radius=2.0)
    constrainTable.update_numpy()

    start_pos=(0.0, 2.0, 2.0, np.deg2rad(0.), np.deg2rad(0.))
    goal_pose=(15.0, 12.0, 12.0, np.deg2rad(0.), np.deg2rad(0.))

    radius = 0.5
    model = HybridAstarWrap(
        radius=radius, 
        instance=instance, 
        constrainTable=constrainTable
    )

    print("Start Searching ......")
    path = model.findPath(
        start_pos, goal_pose
    )
    path = np.array(path)

    print("num_expanded:%d num_generated:%d searchingTime:%.3f" % (
        model.hybridAstar.num_expanded, model.hybridAstar.num_generated,
        model.searching_time
    ))

    # path = np.array(path)
    # print(np.round(path, decimals=2))

    # ### ------ Vista
    # vis = VisulizerVista()
    # tube_mesh = vis.create_tube(path, radius=radius)
    # vis.plot(tube_mesh, color=(0.1, 0.5, 0.8))
    # vis.show()

    ### ------ open3D
    vis = VisulizerO3D()
    vis.addArrow(start_pos, color=np.array([1.0, 0.0, 0.0]))
    vis.addArrow(goal_pose, color=np.array([0.0, 0.0, 1.0]))
    vis.addPathPoint(path[:, :3], color=np.array([0.0, 0.0, 0.0]))
    vis.show()
