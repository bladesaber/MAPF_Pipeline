import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from scripts.version_7.hyperGrapher import HyperGraph

class TorchPathSmoother(object):
    def __init__(self, env_config, graph:HyperGraph):
        self.graph = graph
    
        self.groupTrees = {}
        for groupIdx in graph.groupNodeInfo.keys():

            group_xyzs = []
            groupNodeIdxs = graph.groupNodeInfo[groupIdx]
            for nodeIdx in groupNodeIdxs:
                group_xyzs.append(self.graph.nodeInfo[nodeIdx]['pose'])
            group_xyzs = np.array(group_xyzs)

            self.groupTrees[groupIdx] = KDTree(group_xyzs)
        
        obs_df = pd.read_csv(env_config['obstacleSavePath'], index_col=0)
        self.obs_tree = KDTree(obs_df[['x', 'y', 'z']].values)
    
    def init_graphVars(self):
        torch.from_numpy()
