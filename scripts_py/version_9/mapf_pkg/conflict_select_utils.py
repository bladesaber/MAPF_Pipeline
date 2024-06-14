import numpy as np
import networkx as nx
from typing import Dict, List

from build import mapf_pipeline


class HeadConflictSelector(object):
    class ConflictSortCell(object):
        def __init__(self, conflict: mapf_pipeline.ConflictCell):
            self.conflict = conflict
            self.group_weight = {}
            self.group_sort = {}

        def add_weight(self, group_idx, base_name, weight):
            weight_dict = self.group_weight.setdefault(group_idx, {})
            weight_dict[base_name] = weight

    @staticmethod
    def process(
            pipe_cfgs: dict, group_graph: Dict[int, nx.Graph],
            conflict_list: List[mapf_pipeline.ConflictCell], grid_env: mapf_pipeline.DiscreteGridEnv
    ):
        group_sort_cell_dict = {}
        all_cells: List[HeadConflictSelector.ConflictSortCell] = []

        for conflict in conflict_list:
            sort_cell = HeadConflictSelector.ConflictSortCell(conflict)

            conflict_infos = [
                (conflict.idx0, conflict.x1, conflict.y1, conflict.z1),
                (conflict.idx1, conflict.x0, conflict.y0, conflict.z0)
            ]
            for group_idx, x, y, z in conflict_infos:
                graph = group_graph[group_idx]
                for pipe_name in pipe_cfgs:
                    pipe_info = pipe_cfgs[pipe_name]
                    if pipe_info['group_idx'] != group_idx:
                        continue
                    if pipe_info['loc_flag'] not in list(graph.nodes):
                        continue
                    dist = nx.shortest_path_length(
                        graph, source=pipe_info['loc_flag'], target=grid_env.xyz2flag(x, y, z), weight='weight'
                    )
                    sort_cell.add_weight(group_idx, pipe_info['loc_flag'], dist)

                group_sort_cell_dict.setdefault(group_idx, []).append(sort_cell)
            all_cells.append(sort_cell)

        for group_idx in group_sort_cell_dict.keys():
            cell_list: List[HeadConflictSelector.ConflictSortCell] = group_sort_cell_dict[group_idx]
            pipe_flags = list(cell_list[0].group_weight[group_idx].keys())
            data = []
            for pipe_flag in pipe_flags:
                data.append([cell.group_weight[group_idx][pipe_flag] for cell in cell_list])
            data = np.array(data)
            data = np.argsort(data, axis=1)
            data = np.min(data, axis=0)
            for i, cell in enumerate(cell_list):
                cell.group_sort[group_idx] = data[i]

        head_cell, min_value = None, np.inf
        for cell in all_cells:
            sort_values = np.array(list(cell.group_sort.values()))
            if not np.any(sort_values == 0):
                continue

            if sort_values.sum() < min_value:
                min_value = sort_values.sum()
                head_cell = cell

        return head_cell.conflict
