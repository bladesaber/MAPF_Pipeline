import numpy as np
from typing import List, Union, Dict
import torch
import torch.nn.functional as F
from sklearn.neighbors import KDTree
import networkx as nx
import pandas as pd

from scripts_py.version_9.mapf_pkg import smooth_segment_utils
from scripts_py.version_9.mapf_pkg.torch_utils import TorchUtils, MeanTestLearner, CostUtils
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils


# todo
#  1.自干涉不是产生大曲率的原因，连接点灵活度不够才是产生大曲率的原因
#  2.Bspline不够均匀
#  3.函数约束比离散约束更好

class SelfConflictCell(object):
    def __init__(
            self,
            group_idx: int,
            segment_idx0: int, conflict_pcd0_idxs: Union[np.ndarray, List[int]],
            segment_idx1: int, conflict_pcd1_idxs: Union[np.ndarray, List[int]],
            require_distance: Union[np.ndarray, List[float]]
    ):
        self.group_idx = group_idx
        self.segment_idx0 = segment_idx0
        self.conflict_pcd0_idxs = conflict_pcd0_idxs
        self.segment_idx1 = segment_idx1
        self.conflict_pcd1_idxs = conflict_pcd1_idxs
        self.require_distance = require_distance


class SelfConflictGraph(object):
    @staticmethod
    def create_float_graph(net_cell: smooth_segment_utils.NetworkPath, inner_set_length: float, decimals_r=1e-8):
        df = []
        for seg_idx, seg_cell in net_cell.segments.items():
            pcd = TorchUtils.tensor2np(seg_cell.tensor_info['path_pcd_t'])
            sub_df = pd.DataFrame({
                'x': pcd[:, 0], 'y': pcd[:, 1], 'z': pcd[:, 2],
                'radius': seg_cell.spline_info['radius'].reshape(-1),
                'seg_idx': np.full(shape=(pcd.shape[0]), fill_value=seg_idx),
                'pcd_idx': np.arange(0, pcd.shape[0], 1)
            })
            df.append(sub_df)
        df = pd.concat(df, axis=0, ignore_index=True)

        pcd = df[['x', 'y', 'z']].values
        tree = KDTree(pcd)
        idxs_list = tree.query_radius(pcd, r=decimals_r, return_distance=False)

        df['flag'], flag = -1, 0
        for i, idxs in zip(range(df.shape[0]), idxs_list):
            if df.iloc[i]['flag'] != -1:
                continue
            df.iloc[idxs, df.columns.get_loc('flag')] = flag
            flag += 1
        df['flag'] = df['flag'].astype(int)

        if not np.all(df['flag'] > -1):
            raise ValueError

        graph = nx.Graph()
        for seg_idx_tuple, seg_df in df.groupby(by=['seg_idx']):
            for i in range(seg_df.shape[0] - 1):
                graph.add_edge(seg_df.iloc[i]['flag'], seg_df.iloc[i + 1]['flag'])

        # ------ debug connect graph
        degree_count = {}
        for node_name in graph.nodes:
            node_degree = graph.degree(node_name)
            if node_degree not in degree_count:
                degree_count[node_degree] = 0
            degree_count[node_degree] += 1
        print(f'[INFO]: Update Self Conflict, Group idx:{net_cell.group_idx} '
              f'degree_count:{degree_count} Shape:{len(graph)}.')
        # ------

        # ------ compute inner set flags
        # todo 不足够好
        inner_set_dict = {}
        pcd = df[['x', 'y', 'z']].values
        radius = df['radius'].values
        search_radius = radius + inner_set_length

        tree = KDTree(pcd)
        idxs_list = tree.query_radius(pcd, search_radius, return_distance=False)
        for i, idxs in zip(range(pcd.shape[0]), idxs_list):
            cur_series: pd.Series = df.iloc[i]
            cur_flag = cur_series['flag']

            sub_df: pd.DataFrame = df.iloc[idxs]
            sub_flags = sub_df['flag'].values

            sub_graph = graph.subgraph(sub_flags)
            components = list(nx.connected_components(sub_graph))
            for component in components:
                component_flag = list(component)
                if cur_flag in component_flag:
                    inner_set_dict[cur_flag] = component_flag

        df.set_index(keys=['seg_idx', 'pcd_idx'], inplace=True)

        return df, graph, inner_set_dict

    @staticmethod
    def find_self_conflict_pair(
            unit_lr: float, df: pd.DataFrame, inner_set_dict: Dict[int, List[int]],
            group_idx: int, net_cell: smooth_segment_utils.NetworkPath,
    ):
        pcds = df[['x', 'y', 'z']].values
        radius = df['radius'].values
        flags = df['flag'].values

        tree_i = KDTree(pcds)
        search_radius = radius + np.max(radius) + unit_lr * 2.0
        idxs_list, dist_list = tree_i.query_radius(pcds, search_radius, return_distance=True)

        conflict_pairs = {}
        for i, (idxs, dists) in enumerate(zip(idxs_list, dist_list)):
            if len(idxs) == 0:
                continue

            cur_flag = flags[i]
            inner_flags = inner_set_dict[cur_flag]
            sub_flags_set = flags[idxs]
            conflict_flags = np.setdiff1d(sub_flags_set, inner_flags)

            for conflict_flag in conflict_flags:
                tag = (min(cur_flag, conflict_flag), max(cur_flag, conflict_flag))
                if tag not in conflict_pairs.keys():
                    conflict_pairs[tag] = True

        arrange_df = df.reset_index()
        arrange_df.drop_duplicates(subset=['flag'], inplace=True)
        arrange_df.set_index(keys=['flag'], inplace=True)

        conflict_cells = {}
        for left_flag, right_flag in conflict_pairs.keys():
            left_series, right_series = arrange_df.loc[left_flag], arrange_df.loc[right_flag]
            left_seg_idx, right_seg_idx = int(left_series['seg_idx']), int(right_series['seg_idx'])

            tag = (min(left_seg_idx, right_seg_idx), max(left_seg_idx, right_seg_idx))
            if tag not in conflict_cells.keys():
                conflict_cells[tag] = {
                    'conflict_pcd0_idxs': [],
                    'conflict_pcd1_idxs': [],
                    'left_radius': [],
                    'right_radius': []
                }

            conflict_cells[tag]['conflict_pcd0_idxs'].append(int(left_series['pcd_idx']))
            conflict_cells[tag]['conflict_pcd1_idxs'].append(int(right_series['pcd_idx']))
            conflict_cells[tag]['left_radius'].append(left_series['radius'])
            conflict_cells[tag]['right_radius'].append(right_series['radius'])

        cells = []
        for seg_pair, info in conflict_cells.items():
            seg_idx_i, seg_idx_j = seg_pair
            segment_i, segment_j = net_cell.segments[seg_idx_i], net_cell.segments[seg_idx_j]

            radius_i, radius_j = np.array(info['left_radius']), np.array(info['right_radius'])
            aux_radius = np.maximum(
                np.maximum(radius_i * segment_i.spring_radius_conflict_ratio, segment_i.min_conflict_thickness),
                np.maximum(radius_j * segment_j.spring_radius_conflict_ratio, segment_j.min_conflict_thickness)
            )

            cell = SelfConflictCell(
                group_idx=group_idx,
                segment_idx0=seg_idx_i, segment_idx1=seg_idx_j,
                conflict_pcd0_idxs=np.array(info['conflict_pcd0_idxs']),
                conflict_pcd1_idxs=np.array(info['conflict_pcd1_idxs']),
                require_distance=radius_i + radius_j + aux_radius
            )
            cells.append(cell)

        return cells


class PathConflictCell(object):
    def __init__(
            self,
            group_idx0: int, segment_idx0: int, conflict_pcd0_idxs: Union[np.ndarray, List[int]],
            group_idx1: int, segment_idx1: int, conflict_pcd1_idxs: Union[np.ndarray, List[int]],
            require_distance: Union[np.ndarray, List[float]]
    ):
        self.group_idx0 = group_idx0
        self.segment_idx0 = segment_idx0
        self.conflict_pcd0_idxs = conflict_pcd0_idxs
        self.group_idx1 = group_idx1
        self.segment_idx1 = segment_idx1
        self.conflict_pcd1_idxs = conflict_pcd1_idxs
        self.require_distance = require_distance


class ObstacleConflictCell(object):
    def __init__(
            self,
            group_idx: int, segment_idx: int, conflict_pcd_idxs: Union[np.ndarray, List[int]],
            obs_pcd_idxs: Union[np.ndarray, List[int]], require_distance: Union[np.ndarray, List[float]]
    ):
        self.group_idx = group_idx
        self.segment_idx = segment_idx
        self.conflict_pcd_idxs = conflict_pcd_idxs
        self.obs_pcd_idxs = obs_pcd_idxs
        self.require_distance = require_distance


class PathOptimizer(object):
    def __init__(
            self,
            paths_result: Dict[int, Dict[str, np.ndarray]],
            obstacle_xyzr: np.ndarray, conflict_cfg: Dict, grid_cfg: Dict
    ):
        self.network_cells: Dict[int, smooth_segment_utils.NetworkPath] = {}
        for group_idx, path_dict in paths_result.items():
            self.network_cells[group_idx] = smooth_segment_utils.NetworkPath(path_dict, group_idx, grid_cfg)

        self.obstacle_xyzr = obstacle_xyzr
        self.obstacle_num = self.obstacle_xyzr.shape[0]
        if self.obstacle_num > 0:
            self.obstacle_tensor = TorchUtils.np2tensor(obstacle_xyzr[:, :3], require_grad=False)
            self.obstacle_tree = KDTree(obstacle_xyzr[:, :3])

        self.obstacle_conflict_cells: List[ObstacleConflictCell] = []
        self.path_conflict_cells: List[PathConflictCell] = []

        self.self_conflict_info: Dict[int, Dict[str, Union[nx.Graph, pd.DataFrame]]] = {}
        self.self_conflict_cells: List[SelfConflictCell] = []

        self.conflict_cfg = conflict_cfg

    def init_optimize_cfg(self, group_idx, segment_infos: Dict, path_infos: Dict):
        self.network_cells[group_idx].update_optimize_info(segment_infos, path_infos)

    def prepare_tensor_and_self_graph(self):
        for group_idx, net_cell in self.network_cells.items():
            update_info = False
            for seg_idx, seg_cell in net_cell.segments.items():
                is_update = seg_cell.update_spline_info(
                    net_cell.control_xyzr_np, seg_cell.spline_info['degree'],
                    seg_cell.spline_info['density'], seg_cell.spline_info['update_length_threshold']
                )
                update_info = update_info or is_update

            net_cell.prepare_tensor()

            # if update_info or (group_idx not in self.self_conflict_info.keys()):
            #     df, graph = SelfConflictGraph.create_float_graph(net_cell)
            #     self.self_conflict_info.update({group_idx: {'graph': graph, 'df': df}})

    def update_tensor_and_self_graph(self, iteration: int, update_iteration=10):
        for group_idx, net_cell in self.network_cells.items():
            update_info = False
            if iteration % update_iteration == 0:
                net_cell.update_state(with_tensor=False, with_np=True)
                for seg_idx, seg_cell in net_cell.segments.items():
                    is_update = seg_cell.update_spline_info(
                        net_cell.control_xyzr_np, seg_cell.spline_info['degree'],
                        seg_cell.spline_info['density'], seg_cell.spline_info['update_length_threshold'],
                        tag=f"Group id:{group_idx}"
                    )
                    update_info = update_info or is_update

            if update_info:
                net_cell.prepare_tensor()
                # df, graph = SelfConflictGraph.create_float_graph(net_cell)
                # self.self_conflict_info.update({group_idx: {'graph': graph, 'df': df}})

            else:
                net_cell.update_state(with_tensor=True, with_np=False)
                # self.update_self_conflict_pcd()

    def update_self_conflict_pcd(self):
        for group_idx, net_cell in self.network_cells.items():
            df = self.self_conflict_info[group_idx]['df']
            for seg_idx, seg_cell in net_cell.segments.items():
                pcd = TorchUtils.tensor2np(seg_cell.tensor_info['path_pcd_t'])

                index_list = []
                for pcd_idx in np.arange(0, pcd.shape[0], 1):
                    index_list.append((seg_idx, pcd_idx))
                df.loc[index_list, ['x', 'y', 'z']] = pcd

    def find_obstacle_conflict_cells(self, unit_lr: float):
        self.obstacle_conflict_cells.clear()
        if self.obstacle_num == 0:
            return self.obstacle_conflict_cells

        for group_idx, net_cell in self.network_cells.items():
            fix_pcd = net_cell.control_xyzr_np[net_cell.fix_pcd_idxs, :3]

            for seg_idx, segment_cell in net_cell.segments.items():
                pcd_np = TorchUtils.tensor2np(segment_cell.tensor_info['path_pcd_t'])
                radius = segment_cell.spline_info['radius']

                detection_radius = radius + np.maximum(
                    radius * segment_cell.spring_radius_conflict_ratio, segment_cell.min_conflict_thickness
                ) + unit_lr
                search_radius = detection_radius + np.max(self.obstacle_xyzr[:, -1])

                idxs_list, dist_list = self.obstacle_tree.query_radius(pcd_np, r=search_radius, return_distance=True)

                conflict_pcd_idxs, obs_pcd_idxs, require_radius = [], [], []
                for pcd_idx, (obs_idxs, dists) in enumerate(zip(idxs_list, dist_list)):
                    if len(obs_idxs) == 0:
                        continue

                    if np.any(np.linalg.norm(pcd_np[pcd_idx] - fix_pcd, ord=2, axis=1) == 0.0):
                        continue

                    for obs_idx, dist in zip(obs_idxs, dists):
                        if dist <= detection_radius[pcd_idx] + self.obstacle_xyzr[obs_idx, 3]:
                            conflict_pcd_idxs.append(pcd_idx)
                            obs_pcd_idxs.append(obs_idx)
                            seg_radius = radius[pcd_idx]
                            obs_radius = self.obstacle_xyzr[obs_idx, 3]
                            aux_radius = max(
                                seg_radius * segment_cell.spring_radius_conflict_ratio,
                                segment_cell.min_conflict_thickness
                            )
                            require_radius.append(seg_radius + obs_radius + aux_radius)

                if len(conflict_pcd_idxs) > 0:
                    conflict_pcd_idxs = np.array(conflict_pcd_idxs)
                    obs_pcd_idxs = np.array(obs_pcd_idxs)
                    require_radius = np.array(require_radius)
                    self.obstacle_conflict_cells.append(
                        ObstacleConflictCell(
                            group_idx, segment_cell.idx, conflict_pcd_idxs, obs_pcd_idxs, require_radius
                        )
                    )

        return self.obstacle_conflict_cells

    def find_path_conflict_cells(self, unit_lr: float):
        self.path_conflict_cells.clear()
        group_idxs = list(self.network_cells.keys())
        num = len(group_idxs)

        for i in range(num):
            cell_i = self.network_cells[group_idxs[i]]
            for j in range(i + 1, num, 1):
                cell_j = self.network_cells[group_idxs[j]]

                for _, segment_i in cell_i.segments.items():
                    pcd_i = TorchUtils.tensor2np(segment_i.tensor_info['path_pcd_t'])
                    radius_i = segment_i.spline_info['radius']
                    aux_radius_i = np.maximum(
                        radius_i * segment_i.spring_radius_conflict_ratio, segment_i.min_conflict_thickness
                    )
                    tree_i = KDTree(pcd_i)

                    for _, segment_j in cell_j.segments.items():
                        pcd_j = TorchUtils.tensor2np(segment_j.tensor_info['path_pcd_t'])
                        radius_j = segment_j.spline_info['radius']
                        aux_radius_j = np.maximum(
                            radius_j * segment_j.spring_radius_conflict_ratio, segment_j.min_conflict_thickness
                        )

                        search_radius = (
                                np.max(radius_i) + np.max(radius_j)
                                + max(np.max(aux_radius_i), np.max(aux_radius_j))
                                + unit_lr * 2.0
                        )
                        idxs_list, dist_list = tree_i.query_radius(pcd_j, search_radius, return_distance=True)

                        pcd0_idxs, pcd1_idxs, require_radius = [], [], []
                        for pcd_j_idx, (pcd_i_idx_list, dists) in enumerate(zip(idxs_list, dist_list)):
                            if len(pcd_i_idx_list) == 0:
                                continue

                            for pcd_i_idx, dist in zip(pcd_i_idx_list, dists):
                                detection_radius = (
                                        radius_i[pcd_i_idx] + radius_j[pcd_j_idx]
                                        + max(aux_radius_i[pcd_i_idx], aux_radius_j[pcd_j_idx])
                                        + unit_lr * 2.0
                                )
                                if dist <= detection_radius:
                                    pcd0_idxs.append(pcd_i_idx)
                                    pcd1_idxs.append(pcd_j_idx)
                                    seg_i_radius = radius_i[pcd_i_idx]
                                    seg_j_radius = radius_j[pcd_j_idx]
                                    aux_radius = max(aux_radius_i[pcd_i_idx], aux_radius_j[pcd_j_idx])
                                    require_radius.append(seg_i_radius + seg_j_radius + aux_radius)

                        if len(pcd0_idxs) > 0:
                            pcd0_idxs = np.array(pcd0_idxs)
                            pcd1_idxs = np.array(pcd1_idxs)
                            require_radius = np.array(require_radius)
                            self.path_conflict_cells.append(PathConflictCell(
                                group_idx0=group_idxs[i], segment_idx0=segment_i.idx, conflict_pcd0_idxs=pcd0_idxs,
                                group_idx1=group_idxs[j], segment_idx1=segment_j.idx, conflict_pcd1_idxs=pcd1_idxs,
                                require_distance=require_radius
                            ))

        return self.path_conflict_cells

    def find_self_conflict(self, unit_lr: float):
        self.self_conflict_cells.clear()
        # for group_idx, net_cell in self.network_cells.items():
        #     cell = SelfConflictGraph.find_self_conflict_pair(
        #         unit_lr,
        #         df=self.self_conflict_info[group_idx]['df'],
        #         graph=self.self_conflict_info[group_idx]['graph'],
        #         group_idx=group_idx,
        #         net_cell=net_cell
        #     )
        #     self.self_conflict_cells.extend(cell)
        return self.self_conflict_cells

    def compute_conflict_cost(self) -> Dict[str, Union[torch.Tensor, float]]:
        cost_dict = {
            'obstacle_cost': 0.0,
            'path_conflict_cost': 0.0,
            'self_conflict_cost': 0.0
        }

        for cell in self.obstacle_conflict_cells:
            seg_cell = self.network_cells[cell.group_idx].segments[cell.segment_idx]
            costs = CostUtils.compute_shape_conflict_cost(
                pcd0=seg_cell.tensor_info['path_pcd_t'][cell.conflict_pcd_idxs, :],
                pcd1=self.obstacle_tensor[cell.obs_pcd_idxs, :],
                require_dist=cell.require_distance,
                weight=1.0
            )
            cost_dict['obstacle_cost'] += costs.mean() * self.conflict_cfg['obstacle_conflict_weight']

        for cell in self.path_conflict_cells:
            seg0_cell = self.network_cells[cell.group_idx0].segments[cell.segment_idx0]
            seg1_cell = self.network_cells[cell.group_idx1].segments[cell.segment_idx1]
            costs = CostUtils.compute_shape_conflict_cost(
                pcd0=seg0_cell.tensor_info['path_pcd_t'][cell.conflict_pcd0_idxs, :],
                pcd1=seg1_cell.tensor_info['path_pcd_t'][cell.conflict_pcd1_idxs, :],
                require_dist=cell.require_distance,
                weight=1.0
            )
            cost_dict['path_conflict_cost'] += costs.mean() * self.conflict_cfg['path_conflict_weight']

        for cell in self.self_conflict_cells:
            seg0_cell = self.network_cells[cell.group_idx].segments[cell.segment_idx0]
            seg1_cell = self.network_cells[cell.group_idx].segments[cell.segment_idx1]
            costs = CostUtils.compute_shape_conflict_cost(
                pcd0=seg0_cell.tensor_info['path_pcd_t'][cell.conflict_pcd0_idxs, :],
                pcd1=seg1_cell.tensor_info['path_pcd_t'][cell.conflict_pcd1_idxs, :],
                require_dist=cell.require_distance,
                weight=1.0
            )
            cost_dict['self_conflict_cost'] += costs.mean() * self.conflict_cfg['self_conflict_weight']

        return cost_dict

    def run(self, max_iter: int, lr: float):
        lr_unit = np.linalg.norm(np.array([lr, lr, lr]))
        loss_record, loss_info = [], {}

        for it in range(1, max_iter + 1):
            self.find_obstacle_conflict_cells(lr_unit)
            self.find_path_conflict_cells(lr_unit)
            # self.find_self_conflict(lr_unit)

            # self.draw_constraints(
            #     with_spline=False, with_control=True, with_obstacle_conflict=True,
            #     with_self_conflict=False, with_path_conflict=False
            # )

            loss_info.clear()
            for group_idx, net_cell in self.network_cells.items():
                for name, cost in net_cell.compute_segment_cost().items():
                    loss_info[f"{group_idx}_{name}"] = cost
                for name, cost in net_cell.compute_path_cost().items():
                    loss_info[f"{group_idx}_{name}"] = cost
            loss_info.update(self.compute_conflict_cost())

            loss: torch.Tensor = 0.0
            for cost_name, cost in loss_info.items():
                loss += cost

            loss_np = loss.detach().numpy()
            loss_record.append(loss_np)
            log_txt = f"[iter:{it}] loss:{loss_np:.6f}"
            print(log_txt)

            loss.backward()
            with torch.no_grad():
                for group_idx, net_cell in self.network_cells.items():
                    grad = net_cell.control_pcd_tensor.grad
                    grad[net_cell.fix_pcd_idxs, :] = 0.0
                    grad = grad / (torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-8)
                    net_cell.control_pcd_tensor -= grad * lr
                    net_cell.control_pcd_tensor.grad.zero_()

            # for group_idx, net_cell in self.network_cells.items():
            #     net_cell.update_state(with_tensor=True, with_np=False)
            self.update_tensor_and_self_graph(iteration=it, update_iteration=10)

        for group_idx, net_cell in self.network_cells.items():
            net_cell.update_state(with_tensor=False, with_np=True)

        loss_info_np = {}
        for key in list(loss_info.keys()):
            val = loss_info.pop(key)
            if isinstance(val, torch.Tensor):
                loss_info_np[key] = val.detach().numpy()
            else:
                loss_info_np[key] = val

        return loss_record, loss_info_np

    def auto_run(self, learner: Union[MeanTestLearner]):
        lr_unit = np.linalg.norm(np.array([learner.lr, learner.lr, learner.lr]))
        loss_info, it = {}, 0

        while True:
            it += 1
            self.find_obstacle_conflict_cells(lr_unit)
            self.find_path_conflict_cells(lr_unit)
            # self.find_self_conflict(lr_unit)

            loss_info.clear()
            for group_idx, net_cell in self.network_cells.items():
                for name, cost in net_cell.compute_segment_cost().items():
                    loss_info[f"{name}_{group_idx}"] = cost
                for name, cost in net_cell.compute_path_cost().items():
                    loss_info[f"{name}_{group_idx}"] = cost
            loss_info.update(self.compute_conflict_cost())

            loss: torch.Tensor = 0.0
            for cost_name, cost in loss_info.items():
                loss += cost

            loss_np = loss.detach().numpy()
            stat_info = learner.get_lr(loss_np)
            if stat_info['state']:
                break
            lr = stat_info['lr']

            log_txt = f"[iter:{it}] loss:{loss_np:.6f} lr:{lr}"
            print(log_txt)

            loss.backward()
            with torch.no_grad():
                for group_idx, net_cell in self.network_cells.items():
                    grad = net_cell.control_pcd_tensor.grad
                    grad[net_cell.fix_pcd_idxs, :] = 0.0
                    grad = grad / (torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-8)
                    net_cell.control_pcd_tensor -= grad * lr
                    net_cell.control_pcd_tensor.grad.zero_()

            # for group_idx, net_cell in self.network_cells.items():
            #     net_cell.update_state(with_tensor=True, with_np=False)
            self.update_tensor_and_self_graph(iteration=it, update_iteration=10)

        for group_idx, net_cell in self.network_cells.items():
            net_cell.update_state(with_tensor=False, with_np=True)

        loss_info_np = {}
        for key in list(loss_info.keys()):
            val = loss_info.pop(key)
            if isinstance(val, torch.Tensor):
                loss_info_np[key] = val.detach().numpy()
            else:
                loss_info_np[key] = val

        return loss_info_np, learner

    def loss_check(self):
        loss_info = {}
        for group_idx, net_cell in self.network_cells.items():
            for name, cost in net_cell.compute_segment_cost().items():
                loss_info[f"{group_idx}_{name}"] = cost
            for name, cost in net_cell.compute_path_cost().items():
                loss_info[f"{group_idx}_{name}"] = cost

        loss_info_np = {}
        for key in list(loss_info.keys()):
            val = loss_info.pop(key)
            if isinstance(val, torch.Tensor):
                loss_info_np[key] = val.detach().numpy()
            else:
                loss_info_np[key] = val
        return loss_info_np

    def draw_constraints(
            self, lr_unit=None, vis: VisUtils = None, with_control=True, with_spline=True,
            with_obstacle_conflict=True, with_path_conflict=True, with_self_conflict=True,
            with_obstacle=True
    ):
        if lr_unit is not None:
            self.find_obstacle_conflict_cells(lr_unit)
            self.find_path_conflict_cells(lr_unit)
            # self.find_self_conflict(lr_unit)

        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        for group_idx, net_cell in self.network_cells.items():
            color = np.random.uniform(0.0, 1.0, size=(3,))
            for seg_idx, seg_cell in net_cell.segments.items():
                if with_control:
                    control_pcd = TorchUtils.tensor2np(net_cell.control_pcd_tensor[seg_cell.pcd_idxs, :])
                    line_set = VisUtils.create_line_set(np.arange(0, seg_cell.pcd_idxs.shape[0], 1))
                    mesh = VisUtils.create_line(control_pcd, line_set)
                    vis.plot(mesh, color=color, point_size=5)

                if with_spline:
                    spline_pcd = TorchUtils.tensor2np(seg_cell.tensor_info['path_pcd_t'])
                    line_set = VisUtils.create_line_set(np.arange(0, spline_pcd.shape[0], 1))
                    mesh = VisUtils.create_line(spline_pcd[:, :3], line_set)
                    # vis.plot(mesh, color=color, style='wireframe', point_size=0.1, line_width=2.0)
                    vis.plot(mesh, color=color, point_size=5)

        if with_obstacle:
            mesh = VisUtils.create_point_cloud(self.obstacle_xyzr[:, :3])
            vis.plot(mesh, color=(0.3, 0.3, 0.3), opacity=1.0)

        if with_obstacle_conflict:
            for cell in self.obstacle_conflict_cells:
                seg_cell = self.network_cells[cell.group_idx].segments[cell.segment_idx]
                pcd0 = TorchUtils.tensor2np(seg_cell.tensor_info['path_pcd_t'][cell.conflict_pcd_idxs, :])
                pcd1 = self.obstacle_xyzr[cell.obs_pcd_idxs, :3]

                pcd0_idxs = np.arange(0, pcd0.shape[0], 1)
                pcd1_idxs = pcd0_idxs + pcd0.shape[0]
                pcd = np.concatenate([pcd0, pcd1], axis=0)
                line_set = np.concatenate([
                    np.full(shape=(pcd0.shape[0], 1), fill_value=2),
                    pcd0_idxs.reshape((-1, 1)), pcd1_idxs.reshape((-1, 1))
                ], axis=1)
                mesh = VisUtils.create_line(pcd, line_set)
                vis.plot(mesh, color=(1.0, 0.0, 0.0), point_size=4)

        if with_path_conflict:
            for cell in self.path_conflict_cells:
                seg0_cell = self.network_cells[cell.group_idx0].segments[cell.segment_idx0]
                seg1_cell = self.network_cells[cell.group_idx1].segments[cell.segment_idx1]
                pcd0 = TorchUtils.tensor2np(seg0_cell.tensor_info['path_pcd_t'][cell.conflict_pcd0_idxs, :])
                pcd1 = TorchUtils.tensor2np(seg1_cell.tensor_info['path_pcd_t'][cell.conflict_pcd1_idxs, :])
                pcd0_idxs = np.arange(0, pcd0.shape[0], 1)
                pcd1_idxs = pcd0_idxs + pcd0.shape[0]
                pcd = np.concatenate([pcd0, pcd1], axis=0)
                line_set = np.concatenate([
                    np.full(shape=(pcd0.shape[0], 1), fill_value=2),
                    pcd0_idxs.reshape((-1, 1)), pcd1_idxs.reshape((-1, 1))
                ], axis=1)
                mesh = VisUtils.create_line(pcd, line_set)
                vis.plot(mesh, color=(0.0, 0.0, 1.0), point_size=4)

        if with_self_conflict:
            for cell in self.self_conflict_cells:
                seg0_cell = self.network_cells[cell.group_idx].segments[cell.segment_idx0]
                seg1_cell = self.network_cells[cell.group_idx].segments[cell.segment_idx1]
                pcd0 = TorchUtils.tensor2np(seg0_cell.tensor_info['path_pcd_t'][cell.conflict_pcd0_idxs, :])
                pcd1 = TorchUtils.tensor2np(seg1_cell.tensor_info['path_pcd_t'][cell.conflict_pcd1_idxs, :])
                pcd0_idxs = np.arange(0, pcd0.shape[0], 1)
                pcd1_idxs = pcd0_idxs + pcd0.shape[0]
                pcd = np.concatenate([pcd0, pcd1], axis=0)
                line_set = np.concatenate([
                    np.full(shape=(pcd0.shape[0], 1), fill_value=2),
                    pcd0_idxs.reshape((-1, 1)), pcd1_idxs.reshape((-1, 1))
                ], axis=1)
                mesh = VisUtils.create_line(pcd, line_set)
                vis.plot(mesh, color=(0.0, 1.0, 0.0), point_size=4)

        if show_plot:
            vis.show()
