import numpy as np
import pandas as pd
import networkx as nx
from typing import List
import matplotlib.pyplot as plt
from copy import copy, deepcopy

class MiniumSpanTreeTaskRunner(object):
    def __init__(self):
        self.graph = nx.Graph()

    def add_nodes(self, nodes:List[int]):
        self.graph.add_nodes_from(nodes)

    def add_edge(self, node0, node1, weight):
        # self.graph.add_weighted_edges_from([(node0, node1, weight)])
        self.graph.add_edges_from([
            (node0, node1, {'weight': weight}),
        ])

    def getTaskTrees(self):
        res = nx.minimum_spanning_tree(self.graph, weight='weight')
        res = sorted(res.edges(data=False))
        return res

    def clear(self):
        self.graph.clear()

    def plotGraph(self):
        nx.draw(self.graph, with_labels=True, font_weight='bold')
        plt.show()

class MiniumDistributeTreeTaskRunner(object):
    def __init__(self):
        self.names = []
        self.poses = []

    def add_node(self, name, pose):
        self.names.append(name)
        self.poses.append(pose)

    def compute_dist(self, pose0, pose1):
        return np.sum(np.abs(pose0 - pose1))

    def getTaskTrees(self, ):
        poses = np.array(self.poses)
        size = poses.shape[0]

        dist_mat = np.ones((size, size)) * np.inf
        for i in range(size):
            for j in range(i + 1, size, 1):
                dist_mat[i, j] = self.compute_dist(poses[i], poses[j])
                dist_mat[j, i] = self.compute_dist(poses[i], poses[j])

        tree_links = {}
        for i in range(size):
            tree_links[i] = [i]

        res_list = []
        for _ in range(size - 1):
            x_ind, y_ind = np.unravel_index(np.argmin(dist_mat, axis=None), dist_mat.shape)

            for j in tree_links[y_ind]:
                for i in tree_links[x_ind]:
                    dist_mat[i, j] = np.inf
                    dist_mat[j, i] = np.inf

            new_set = tree_links[x_ind]
            new_set.extend(tree_links[y_ind])
            tree_links[x_ind] = new_set
            tree_links[y_ind] = new_set

            res_list.append((self.names[x_ind], self.names[y_ind]))

        return res_list

    def clear(self):
        self.poses.clear()
        self.names.clear()

class SizeTreeTaskRunner(object):
    def __init__(self):
        self.names = []
        self.poses = []
        self.radius = []

    def add_node(self, name, pose, radius):
        self.names.append(name)
        self.poses.append(pose)
        self.radius.append(radius)

    def compute_dist(self, pose0, pose1):
        return np.sum(np.abs(pose0 - pose1))

    def compute_leafDist(self, leaf0, leaf1):
        poses_0 = np.array(leaf0['poses'])
        poses_1 = np.array(leaf1['poses'])
        names_0 = leaf0['names']
        names_1 = leaf1['names']

        pair_i, pair_j, best_dist = None, None, np.inf
        for i in range(poses_0.shape[0]):
            for j in range(poses_1.shape[0]):
                if names_0[i] == names_1[j]:
                    continue

                pose_i = poses_0[i, :]
                pose_j = poses_1[j, :]

                dist = self.compute_dist(pose_i, pose_j)
                if dist < best_dist:
                    best_dist = dist
                    pair_i = i
                    pair_j = j

        return pair_i, pair_j, best_dist

    def getTaskTrees(self, method: str = 'method2'):
        radius = np.array(self.radius)
        poses = np.array(self.poses)
        size = len(self.names)

        search_leaves = {}
        for i in range(size):
            search_leaves[self.names[i]] = {
                'radius': radius[i],
                'names': [self.names[i]],
                'poses': [poses[i, :]]
            }

        res_list = []
        for _ in range(size - 1):
            search_radius = np.array([search_leaves[k]['radius'] for k in search_leaves.keys()])
            idxs_0 = np.where(search_radius == np.max(search_radius))[0]

            if idxs_0.shape[0] > 1:
                idxs_1 = idxs_0

            else:
                search_radius[idxs_0[0]] = -1.0
                idxs_1 = np.where(search_radius == np.max(search_radius))[0]

            pairLeaf_tag0, pairLeaf_tag1, tag0_SelectIdx, tag1_SelectIdx, best_dist = None, None, None, None, np.inf

            tags = np.array(list(search_leaves.keys()))
            tags_0 = tags[idxs_0]
            tags_1 = tags[idxs_1]

            for tag0 in tags_0:
                for tag1 in tags_1:
                    if tag0 == tag1:
                        continue

                    leaf0 = search_leaves[tag0]
                    leaf1 = search_leaves[tag1]

                    selectIdx_i, selectIdx_j, dist = self.compute_leafDist(leaf0, leaf1)
                    if dist < best_dist:
                        best_dist = dist
                        pairLeaf_tag0 = tag0
                        pairLeaf_tag1 = tag1
                        tag0_SelectIdx = selectIdx_i
                        tag1_SelectIdx = selectIdx_j

            ### ------ Method 1
            if method == 'method1':
                pair_radius = (search_leaves[pairLeaf_tag0]['radius'] + search_leaves[pairLeaf_tag1]['radius']) / 2.0
                res_list.append((
                    search_leaves[pairLeaf_tag0]['names'][tag0_SelectIdx],
                    search_leaves[pairLeaf_tag1]['names'][tag1_SelectIdx],
                    pair_radius
                ))

            ### ------ Method 2
            elif method == 'method2':
                if search_leaves[pairLeaf_tag0]['radius'] < search_leaves[pairLeaf_tag1]['radius']:
                    pair_radius = search_leaves[pairLeaf_tag0]['radius']
                    res_list.append((
                        search_leaves[pairLeaf_tag0]['names'][tag0_SelectIdx],
                        search_leaves[pairLeaf_tag1]['names'][tag1_SelectIdx],
                        pair_radius
                    ))
                else:
                    pair_radius = search_leaves[pairLeaf_tag1]['radius']
                    res_list.append((
                        search_leaves[pairLeaf_tag1]['names'][tag1_SelectIdx],
                        search_leaves[pairLeaf_tag0]['names'][tag0_SelectIdx],
                        pair_radius
                    ))
            else:
                 raise ValueError

            new_tag = f'{pairLeaf_tag0}+{pairLeaf_tag1}'

            names = search_leaves[pairLeaf_tag0]['names'].copy()
            names.extend(search_leaves[pairLeaf_tag1]['names'])

            poses = deepcopy(search_leaves[pairLeaf_tag0]['poses'])
            poses.extend(search_leaves[pairLeaf_tag1]['poses'])

            search_leaves[new_tag] = {
                'radius': pair_radius,
                'names': names,
                'poses': poses
            }

            del search_leaves[pairLeaf_tag0], search_leaves[pairLeaf_tag1]

        return res_list

if __name__ == '__main__':
    allocater = SizeTreeTaskRunner()

    pipeInfos = [
        {
            'name': 'P1',
            'pose': [10, 26, 23],
            'radius': 2.1
        },
        {
            'name': 'M1',
            'pose': [7, 24, 36],
            'radius': 1.05
        },
        {
            'name': 'P_to_slave0',
            'pose': [22, 23, 43],
            'radius': 1.05
        },
        {
            'name': 'P_port',
            'pose': [31, 26, 23],
            'radius': 2.1
        },
    ]

    for pipeInfo in pipeInfos:
        allocater.add_node(pipeInfo['name'], pipeInfo['pose'], pipeInfo['radius'])
    res = allocater.getTaskTrees()
    print(res)