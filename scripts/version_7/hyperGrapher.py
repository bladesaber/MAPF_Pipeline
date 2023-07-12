import os
import json
import numpy as np
import networkx as nx
from sklearn.neighbors import KDTree
from typing import Dict
import matplotlib.pyplot as plt
import math

class HyperGraph(object):
    def __init__(self, env_config):
        self.nodeInfo = {}
        self.groupPathInfo = {}
        self.groupNodeInfo = {}

        self.pose2NodeIdx = {}

        self.network = nx.Graph()
        self.env_config = env_config

    def parse_pathRes(self, pathRes:Dict):
        self.pose2NodeIdx.clear()
        nodeIdx_num = 0

        for groupIdx in pathRes.keys():
            self.groupNodeInfo[groupIdx] = []

            for path_xyzrls in pathRes[groupIdx]:
                
                last_nodeIdx = None
                for i, (x, y, z, radius, length) in enumerate(path_xyzrls):

                    if (x, y, z) not in self.pose2NodeIdx.keys():
                        current_nodeIdx = nodeIdx_num
                        self.network.add_node(current_nodeIdx)
                        self.pose2NodeIdx[(x, y, z)] = current_nodeIdx

                        nodeIdx_num += 1

                    else:
                        current_nodeIdx = self.pose2NodeIdx[(x, y, z)]
                    
                    self.nodeInfo[current_nodeIdx] = {
                        'radius': radius,
                        'groupIdx': groupIdx,
                        'pose': np.array([x, y, z])
                    }
                    self.groupNodeInfo[groupIdx].append(current_nodeIdx)

                    if i > 0:
                        self.network.add_edge(last_nodeIdx, current_nodeIdx)

                    last_nodeIdx = current_nodeIdx

        for groupIdx in self.groupNodeInfo:
            self.groupNodeInfo[groupIdx] = np.array(set(self.groupNodeInfo[groupIdx]))

    def definePath(
            self, startPos:np.array, startDire, endPos:np.array, endDire, groupIdx
        ):
        startNodeIdx = self.pose2NodeIdx[(startPos[0], startPos[1], startPos[2])]
        self.nodeInfo[startNodeIdx].update({
            'direction': startDire,
            'direVec': self.polar2vec(startDire)
        })

        endNodeIdx = self.pose2NodeIdx[(endPos[0], endPos[1], endPos[2])]
        self.nodeInfo[endNodeIdx].update({
            'direction': endDire,
            'direVec': self.polar2vec(endDire)
        })

        path_nodeIdxs = nx.shortest_path(self.network, startNodeIdx, endNodeIdx)
        assert len(path_nodeIdxs) > 0
        
        if groupIdx not in self.groupPathInfo:
            self.groupPathInfo[groupIdx] = []

        self.groupPathInfo[groupIdx].append(path_nodeIdxs)

    def polar2vec(self, polarVec, length=1.0):
        dz = length * math.sin(polarVec[1])
        dl = length * math.cos(polarVec[1])
        dx = dl * math.cos(polarVec[0])
        dy = dl * math.sin(polarVec[0])
        return np.array([dx, dy, dz])

    def plotGraph(self):
        nx.draw_shell(self.network, with_labels=True, font_weight='bold')
        plt.show()

    def plotPath(self, groupIdxs=None):
        if groupIdxs is not None:
            group_keys = groupIdxs
        else:
            group_keys = self.groupPathInfo.keys()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim3d(-1, self.env_config['grid_x']+1)
        ax.set_ylim3d(-1, self.env_config['grid_y']+1)
        ax.set_zlim3d(-1, self.env_config['grid_z']+1)

        colors = np.random.uniform(0.0, 1.0, (5, 3))
        for groupIdx in group_keys:
            for path_nodeIdxs in self.groupPathInfo[groupIdx]:

                path_xyz = []
                for nodeIdx in path_nodeIdxs:
                    node_info = self.nodeInfo[nodeIdx]

                    pose = node_info['pose']
                    path_xyz.append(pose)
                    
                    if 'direVec' in node_info.keys():
                        vec = node_info['direVec']
                        ax.quiver(
                            pose[0], pose[1], pose[2], 
                            vec[0], vec[1], vec[2], 
                            length=5.0, normalize=True, color='r'
                        )

                path_xyz = np.array(path_xyz)
                ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], '*-', c=colors[groupIdx])
        
        plt.show()

    def clear(self):
        self.network.clear()
        self.nodeInfo.clear()
        self.pose2NodeIdx.clear()

def main():
    envJsonFile = '/home/quan/Desktop/tempary/application_pipe/cond.json'
    with open(envJsonFile, 'r') as f:
        env_config = json.load(f)
    
    groupCells = {}
    for group_cfg in env_config['pipeConfig']:
        for cell in group_cfg['pipe']:
            groupCells[cell['name']] = cell

    grouplinks = {
        0: [
            {"start": 'p', 'end': 'p1', 'startFlexRatio': 0.0, 'endFlexRatio': 0.2},
            {"start": 'p', 'end': 'M1', 'startFlexRatio': 0.0, 'endFlexRatio': 0.4},
            {"start": 'p', 'end': 'p_valve', 'startFlexRatio': 0.0, 'endFlexRatio': 0.2}
        ],
        1: [
            {"start": 'B_valve', 'end': 'M3', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
            {"start": 'B_valve', 'end': 'B', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}
        ],
        2: [
            {"start": 'T_valve', 'end': 'T', 'startFlexRatio': 0.2, 'endFlexRatio': 0.0},
            {"start": 'A2T', 'end': 'T', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}
        ],
        3: [
            {"start": 'A_valve', 'end': 'A2valve_01', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
            {"start": 'A_valve', 'end': 'A2valve_02', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}
        ],
        4: [
            {"start": 'valve_01', 'end': 'A', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
            {"start": 'valve_02', 'end': 'A', 'startFlexRatio': 0.25, 'endFlexRatio': 0.0},
            {"start": 'valve_03','end': 'A', 'startFlexRatio': 0.45, 'endFlexRatio': 0.0},
            {"start": 'valve_03','end': 'M2', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}

            # {"start": 'valve_01', 'end': 'valve_02', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
            # {"start": 'valve_02', 'end': 'A', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
            # {"start": 'valve_03','end': 'A', 'startFlexRatio': 0.45, 'endFlexRatio': 0.0},
            # {"start": 'valve_03','end': 'M2', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}
        ]
    }

    pathRes = np.load(os.path.join(env_config['projectDir'], 'res.npy'), allow_pickle=True).item()

    graph = HyperGraph(env_config)
    graph.parse_pathRes(pathRes)

    for groupIdx in grouplinks.keys():
        for linkCfg in grouplinks[groupIdx]:
            startCell = groupCells[linkCfg['start']]
            endCell = groupCells[linkCfg['end']]

            graph.definePath(
                startPos=startCell['grid_position'], startDire=[startCell['alpha'], startCell['theta']],
                endPos=endCell['grid_position'], endDire=[endCell['alpha'], endCell['theta']],
                groupIdx=groupIdx
            )

    graph.plotPath()

if __name__ == '__main__':
    main()

    
