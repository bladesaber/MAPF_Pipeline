import numpy as np
import networkx as nx
from typing import List, Callable
import matplotlib.pyplot as plt

"""
Note: Main channel must have same and largest radius
"""


class MinimumSpanTree(object):
    def __init__(self):
        self.graph = nx.Graph()

    def _add_edge(self, node0, node1, weight: float):
        self.graph.add_edge(node0, node1, weight=weight)

    def _add_edges(self, edges_info):
        for node0, node1, weight in edges_info:
            self.graph.add_edge(node0, node1, weight=weight)

    def _compute(self, algorithm='kruskal', revert=False):
        """
        algorithm:
            kruskal: tree is construct through numbers of distribute set
            prim: tree grow from center
        """
        res_graph = nx.minimum_spanning_tree(self.graph, weight='weight', algorithm=algorithm)
        res_edges = list(res_graph.edges(data=True))
        res_edges = sorted(res_edges, key=lambda x: x[2]['weight'], reverse=revert)
        return res_edges

    def draw_connect_network(self, minimum_tree: nx.Graph = None, with_plot=True):
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(self.graph, pos, edge_color='black', width=1)
        nx.draw_networkx_labels(self.graph, pos, font_size=12)
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in self.graph.edges(data=True)}
        )
        nx.draw_networkx_edges(minimum_tree, pos, edge_color='red', width=2)
        if with_plot:
            plt.show()

    def draw_sequence_network(self, res_edges, with_plot=True):
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(self.graph, pos, edge_color='black', width=1)
        for seq_idx, (node_0, node_1, data) in enumerate(res_edges):
            self.graph.add_edge(node_0, node_1, seq_idx=seq_idx)
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels={(u, v): d['seq_idx'] for u, v, d in self.graph.edges(data=True)}
        )
        if with_plot:
            plt.show()

    @staticmethod
    def create_graph_from_node(node_dict: dict):
        graph = nx.Graph()
        names = list(node_dict.keys())
        for i in range(len(names)):
            node_i = node_dict[names[i]]
            for j in range(i + 1, len(names), 1):
                node_j = node_dict[names[j]]
                weight = np.linalg.norm(np.array(node_i['coord']) - np.array(node_j['coord']), ord=2)
                graph.add_edge(names[i], names[j], weight=weight)
        return graph


class MinimumDistanceTree(MinimumSpanTree):
    def add_item(self, name0: str, name1: str, dist: float):
        self._add_edge(name0, name1, dist)

    def compute(self, algorithm='kruskal'):
        super()._compute(algorithm=algorithm)


class MinimumVolumeTree(MinimumSpanTree):
    def add_item(self, name0: str, name1: str, dist: float, area0: float, area1: float):
        self._add_edge(name0, name1, dist * max(area0, area1))

    def compute(self, algorithm='kruskal'):
        super()._compute(algorithm=algorithm, revert=True)


class MinimumSizeTree(MinimumSpanTree):
    def add_item(self, name0: str, name1: str, area0: float, area1: float):
        weight = (area0 + area1) * 0.5
        self._add_edge(name0, name1, weight)

    def compute(self, algorithm='kruskal'):
        super()._compute(algorithm=algorithm, revert=True)


class HyperRadiusGrowthTree(object):

    def __init__(self):
        super().__init__()
        self.node_dict = {}

    def add_item(self, name: str, radius: float, coord: np.ndarray):
        self.node_dict[name] = {'coord': coord, 'radius': radius}

    @staticmethod
    def compute_leaf_dist(leaf_0: dict, leaf_1: dict):
        """
        leaf: {coords:List[xyz], names:List[str], radius:float}
        """
        poses_0, poses_1 = np.array(leaf_0['coords']), np.array(leaf_1['coords'])
        names_0, names_1 = leaf_0['names'], leaf_1['names']
        pair_0_name, pair_1_name, best_dist = None, None, np.inf
        for i in range(len(names_0)):
            for j in range(len(names_1)):
                if names_0[i] == names_1[j]:
                    continue
                dist = np.linalg.norm(poses_0[i] - poses_1[j], ord=2)
                if dist < best_dist:
                    best_dist = dist
                    pair_0_name = names_0[i]
                    pair_1_name = names_1[j]
        return pair_0_name, pair_1_name, best_dist

    @staticmethod
    def merge_leafs(leafs_dict: dict, merge_tags: List[str], merge_radius_fun: Callable = np.min):
        """
        leafs_dict: {tag: leaf}
        """
        new_tag = '+'.join(merge_tags)
        new_radius = merge_radius_fun([leafs_dict[tag]['radius'] for tag in merge_tags])

        names = leafs_dict[merge_tags[0]]['names'].copy()
        coords = leafs_dict[merge_tags[0]]['coords'].copy()
        for tag in merge_tags[1:]:
            names.extend(leafs_dict[tag]['names'])
            coords.extend(leafs_dict[tag]['coords'])

        leafs_dict[new_tag] = {'radius': new_radius, 'names': names, 'coords': coords}
        for tag in merge_tags:
            del leafs_dict[tag]

        return leafs_dict, new_tag

    @staticmethod
    def union_leafs(main_leaf: dict, leaf: dict, radius: float = None):
        main_leaf['names'].extend(leaf['names'])
        main_leaf['coords'].extend(leaf['coords'])
        if radius is not None:
            main_leaf['radius'] = radius
        return main_leaf

    def compute_block(self, block_names, main_leaf: dict = None):
        leafs_dict = {}
        if main_leaf is not None:
            leafs_dict['main'] = main_leaf

        for name in block_names:
            leafs_dict[name] = {
                'names': [name], 'coords': [self.node_dict[name]['coord']], 'radius': self.node_dict[name]['radius']
            }

        res_list = []
        for _ in range(len(leafs_dict) - 1):
            # max radius hs the highest priority
            radius_list = np.array([leafs_dict[i]['radius'] for i in leafs_dict.keys()])
            first_priority_idxs = np.where(radius_list == np.max(radius_list))[0]

            # if first priority has only one element, use smaller radius as second priority
            if first_priority_idxs.shape[0] > 1:
                second_priority_idxs = first_priority_idxs
            else:
                radius_list[first_priority_idxs[0]] = -1.0
                second_priority_idxs = np.where(radius_list == np.max(radius_list))[0]

            tags = np.array(list(leafs_dict.keys()))
            first_priority_tags = tags[first_priority_idxs]
            second_priority_tags = tags[second_priority_idxs]

            best_tag_0, best_tag_1, best_name_0, best_name_1, best_dist = None, None, None, None, np.inf
            for tag_0 in first_priority_tags:
                for tag_1 in second_priority_tags:
                    if tag_0 == tag_1:
                        continue

                    name_0, name_1, dist = self.compute_leaf_dist(leafs_dict[tag_0], leafs_dict[tag_1])
                    if dist < best_dist:
                        best_dist = dist
                        best_tag_0, best_tag_1 = tag_0, tag_1
                        best_name_0, best_name_1 = name_0, name_1

            leafs_dict, new_tag = self.merge_leafs(leafs_dict, merge_tags=[best_tag_0, best_tag_1])
            res_list.append((
                best_name_0, best_name_1, {'radius': leafs_dict[new_tag]['radius']}
            ))

        main_leaf = list(leafs_dict.values())[0]
        return res_list, main_leaf

    def draw_connect_network(self, with_plot=True):
        graph = MinimumSpanTree.create_graph_from_node(self.node_dict)
        plt.figure()
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(graph, pos, edge_color='black', width=1)
        nx.draw_networkx_labels(graph, pos, font_size=12)
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in graph.edges(data=True)}
        )
        if with_plot:
            plt.show()

    def draw_sequence_network(self, res_edges, with_plot=True):
        graph = MinimumSpanTree.create_graph_from_node(self.node_dict)
        plt.figure()
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(graph, pos, edge_color='black', width=1)
        nx.draw_networkx_labels(graph, pos, font_size=12)
        edge_labels = {}
        for seq_idx, (node_0, node_1, _) in enumerate(res_edges):
            edge_labels[(node_0, node_1)] = seq_idx
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        if with_plot:
            plt.show()

# TODO: 基于原理图的network连接方法,大概分为2个步骤:
#   1.将多inlet多outlet原理图以上面的某个方法(eg.最小距离生成树)形成一个网络
#   2.约束类型a:如果存在某个outlet需要经过超出n个其他outlet才能与inlet相连接，
#             则断开该outlet所有连接，将其连接到最接近的inlet或最接近的小于n的其他outlet集中
#   3.约束类型b:如果存在某个outlet其与最接近的inlet的连通长度超出距离m,则断开该outlet所有连接，将其连接到最接近的inlet
#   4.约束类型c:划分n个outlet，且这些outlet相互间的连通长度小于m，则称为1个团，1个团只能包含小于等于n个outlet。如果存在某个outlet，
#              其不属于团A，则其与团A的小于连通长度应该大于k
#   5.拆分多应用场景，根据每个应用场景查看当前network是否满足要求，不满足则删除不满足的edge，或添加新的edge
