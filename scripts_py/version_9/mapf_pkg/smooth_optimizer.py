import numpy as np
from typing import List, Union, Dict
import torch
import torch.nn.functional as F
from sklearn.neighbors import KDTree

from scripts_py.version_9.mapf_pkg import smooth_utils
from scripts_py.version_9.mapf_pkg.torch_utils import TorchUtils
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils
from build import mapf_pipeline


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
            grid: mapf_pipeline.DiscreteGridEnv,
            paths_result: Dict[int, Dict[str, np.ndarray]],
            obstacle_xyzr: np.ndarray,
            info: Dict
    ):
        self.network_cells: Dict[int, smooth_utils.NetworkPath] = {}
        for group_idx, path_dict in paths_result.items():
            self.network_cells[group_idx] = smooth_utils.NetworkPath(grid, path_dict, group_idx)

        self.obstacle_xyzr = obstacle_xyzr
        self.obstacle_tensor = TorchUtils.np2tensor(obstacle_xyzr[:, :3], require_grad=False)
        self.obstacle_tree = KDTree(obstacle_xyzr[:, :3])

        self.obstacle_conflict_cells: List[ObstacleConflictCell] = []
        self.path_conflict_cells: List[PathConflictCell] = []

        self.info = info

    def find_obstacle_conflict_cells(self, unit_lr: float):
        self.obstacle_conflict_cells.clear()

        for group_idx, network_cell in self.network_cells.items():
            for seg_idx, segment_cell in network_cell.segments.items():
                pcd_np = TorchUtils.tensor2np(segment_cell.pcd_tensor)
                search_radius = np.max(segment_cell.radius_np) + np.max(self.obstacle_xyzr[:, -1]) + unit_lr
                idxs_list, dist_list = self.obstacle_tree.query_radius(pcd_np, r=search_radius, return_distance=True)

                conflict_pcd_idxs, obs_pcd_idxs, require_radius = [], [], []
                for pcd_idx, (obs_idxs, dists) in enumerate(zip(idxs_list, dist_list)):
                    if len(obs_idxs) == 0:
                        continue

                    for obs_idx, dist in zip(obs_idxs, dists):
                        if dist < segment_cell.radius_np[pcd_idx] + self.obstacle_xyzr[obs_idx, 3] + unit_lr:
                            conflict_pcd_idxs.append(pcd_idx)
                            obs_pcd_idxs.append(obs_idx)
                            require_radius.append(segment_cell.radius_np[pcd_idx] + self.obstacle_xyzr[obs_idx, 3])

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
                    pcd_i = TorchUtils.tensor2np(segment_i.pcd_tensor)
                    radius_i = segment_i.radius_np
                    tree_i = KDTree(pcd_i)

                    for _, segment_j in cell_j.segments.items():
                        pcd_j = TorchUtils.tensor2np(segment_j.pcd_tensor)
                        radius_j = segment_j.radius_np

                        search_radius = np.max(radius_i) + np.max(radius_j) + unit_lr
                        idxs_list, dist_list = tree_i.query_radius(pcd_j, search_radius, return_distance=True)

                        pcd0_idxs, pcd1_idxs, require_radius = [], [], []
                        for pcd_j_idx, (pcd_i_idx_list, dists) in enumerate(zip(idxs_list, dist_list)):
                            if len(pcd_i_idx_list) == 0:
                                continue

                            for pcd_i_idx, dist in zip(pcd_i_idx_list, dists):
                                if dist < radius_i[pcd_i_idx] + radius_j[pcd_j_idx] + unit_lr:
                                    pcd0_idxs.append(pcd_i_idx)
                                    pcd1_idxs.append(pcd_j_idx)
                                    require_radius.append(radius_i[pcd_i_idx] + radius_j[pcd_j_idx])

                        self.path_conflict_cells.append(PathConflictCell(
                            group_idx0=group_idxs[i], segment_idx0=segment_i.idx, conflict_pcd0_idxs=pcd0_idxs,
                            group_idx1=group_idxs[j], segment_idx1=segment_j.idx, conflict_pcd1_idxs=pcd1_idxs,
                            require_distance=require_radius
                        ))

        return self.path_conflict_cells

    @staticmethod
    def compute_shape_conflict_cost(pcd0: torch.Tensor, pcd1: torch.Tensor, require_dist: np.ndarray):
        require_dist = TorchUtils.np2tensor(require_dist.reshape((-1, 1)), require_grad=False)
        return F.relu(require_dist - torch.norm(pcd0 - pcd1, p=2, dim=1))

    def compute_conflict_cost(self) -> Dict[str, torch.Tensor]:
        cost_dict = {
            'obstacle_cost': 0.0,
            'path_conflict_cost': 0.0
        }

        for cell in self.obstacle_conflict_cells:
            seg_cell = self.network_cells[cell.group_idx].segments[cell.segment_idx]
            costs = self.compute_shape_conflict_cost(
                pcd0=seg_cell.pcd_tensor[cell.conflict_pcd_idxs, :],
                pcd1=self.obstacle_tensor[cell.obs_pcd_idxs, :],
                require_dist=cell.require_distance,
            )
            cost_dict['obstacle_cost'] += costs.mean() * self.info['obstacle_weight']

        for cell in self.path_conflict_cells:
            seg0_cell = self.network_cells[cell.group_idx0].segments[cell.segment_idx0]
            seg1_cell = self.network_cells[cell.group_idx1].segments[cell.segment_idx1]
            costs = self.compute_shape_conflict_cost(
                pcd0=seg0_cell.pcd_tensor[cell.conflict_pcd0_idxs, :],
                pcd1=seg1_cell.pcd_tensor[cell.conflict_pcd1_idxs, :],
                require_dist=cell.require_distance,
            )
            cost_dict['path_conflict_cost'] += costs.mean() * self.info['path_conflict_weight']

        return cost_dict

    def draw_conflict_graph(self, vis: VisUtils = None, with_obstacle=True, with_path_conflict=True):
        if vis is None:
            vis = VisUtils()

        for group_idx, network_cell in self.network_cells.items():
            network_cell.draw_segment(with_spline=True, with_control=True, vis=vis)

        if with_obstacle:
            for cell in self.obstacle_conflict_cells:
                seg_cell = self.network_cells[cell.group_idx].segments[cell.segment_idx]
                pcd0 = TorchUtils.tensor2np(seg_cell.pcd_tensor[cell.conflict_pcd_idxs, :])
                pcd1 = self.obstacle_xyzr[cell.obs_pcd_idxs, :3]
                pcd0_idxs = np.arange(0, pcd0.shape[0], 1)
                pcd1_idxs = pcd0_idxs + pcd0.shape[0]
                pcd = np.concatenate([pcd0, pcd1], axis=0)
                line_set = np.concatenate([
                    np.full(shape=(pcd0.shape[0], 1), fill_value=2), pcd0_idxs.reshape((-1, 1)), pcd1_idxs.reshape((-1, 1))
                ])
                mesh = VisUtils.create_line(pcd, line_set)
                vis.plot(mesh, color=(1.0, 0.0, 0.0), point_size=4)

        if with_path_conflict:
            for cell in self.path_conflict_cells:
                seg0_cell = self.network_cells[cell.group_idx0].segments[cell.segment_idx0]
                seg1_cell = self.network_cells[cell.group_idx1].segments[cell.segment_idx1]
                pcd0 = TorchUtils.tensor2np(seg0_cell.pcd_tensor[cell.conflict_pcd0_idxs, :])
                pcd1 = TorchUtils.tensor2np(seg1_cell.pcd_tensor[cell.conflict_pcd1_idxs, :])
                pcd0_idxs = np.arange(0, pcd0.shape[0], 1)
                pcd1_idxs = pcd0_idxs + pcd0.shape[0]
                pcd = np.concatenate([pcd0, pcd1], axis=0)
                line_set = np.concatenate([
                    np.full(shape=(pcd0.shape[0], 1), fill_value=2), pcd0_idxs.reshape((-1, 1)), pcd1_idxs.reshape((-1, 1))
                ])
                mesh = VisUtils.create_line(pcd, line_set)
                vis.plot(mesh, color=(0.0, 0.0, 1.0), point_size=4)

        vis.show()
