import os
import shutil
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Union, Literal
import pandas as pd
import pyvista
import torch
import torch.nn.functional as F
import h5py
from sklearn.neighbors import KDTree

from scripts_py.mapf_pkg.bspline_utils import BsplineUtils
from scripts_py.mapf_pkg.torch_utils import TorchUtils, CostUtils, TorchMath
from scripts_py.mapf_pkg.visual_utils import VisUtils


class SegmentCell(object):
    def __init__(self, idx: int, group_idx: int, c_idxs: np.ndarray, parent_cell):
        self.idx = idx
        self.group_idx = group_idx
        self.c_idxs = c_idxs
        self.parent_cell: GroupCell = parent_cell
        self.cell_type: Literal["segment", "connector"] = 'segment'

        # ------ connector cfg
        self.left_clamp = False
        self.right_clamp = False
        self.left_vector: np.ndarray = np.array([0., 0., 0.])
        self.right_vector: np.ndarray = np.array([0., 0., 0.])

        self.left_overlap_num: int = None
        self.right_overlap_num: int = None
        self.left_intercept_num: int = None
        self.right_intercept_num: int = None

        # ------ B-spline cfg
        self.density: float = 0.5
        self.degree: int = None
        self.spline_length: float = 0.0
        self.knot: np.ndarray = None
        self.fill_last_one: bool = None
        self.ts: np.ndarray = None

        self.spline_mat: np.ndarray = None
        self.mat_order1: np.ndarray = None
        self.mat_order2: np.ndarray = None
        self.seg_scale = 1.0
        self.radius: np.ndarray = None
        self.c_radius: np.ndarray = None

        # ------ Cost cfg
        self.cost_cfgs: List[Dict] = []
        self.path_thickness = 0.01
        self.obstacle_thickness = 0.01

        # ------ Tensor
        self.spline_mat_t: torch.Tensor = None
        self.mat_order1_t: torch.Tensor = None
        self.mat_order2_t: torch.Tensor = None
        self.spline_xyz_t: torch.Tensor = None
        self.spline_curvature_t: torch.Tensor = None
        self.spline_vector_t: torch.Tensor = None

        self.with_analytical_tangent = False
        self.with_analytical_curvature = False

    def __str__(self):
        return f"segment_{self.idx}"

    def __len__(self):
        return self.c_idxs.shape[0]

    def init_cfg(self, cfg: Dict, parent_cell=None):
        self.degree = cfg['bspline_degree']
        self.density = cfg['bspline_density']
        self.with_analytical_curvature = cfg['with_analytical_curvature_computed']
        self.with_analytical_tangent = cfg['with_analytical_tangent_computed']

        self.seg_scale = cfg['segment_scale']
        c_tabel_idxs = self.parent_cell.c_table.index[self.c_idxs]
        self.parent_cell.c_table.loc[c_tabel_idxs, 'seg_scale'] = self.seg_scale

        assert cfg['left_overlap_ratio'] + cfg['right_overlap_ratio'] <= 0.9
        assert (cfg['left_overlap_ratio'] > 0.0) and (cfg['right_overlap_ratio'] > 0.0)
        assert len(self) >= self.degree * 2 + 1

        self.left_overlap_num = max(
            int(math.floor(cfg['left_overlap_ratio'] * len(self))), self.degree
        )
        self.right_overlap_num = max(
            int(math.floor(cfg['right_overlap_ratio'] * len(self))), self.degree
        )

        assert (1.0 > cfg['left_intercept_ratio'] > 0.0) and (1.0 > cfg['right_intercept_ratio'] > 0.0)

        left_intercept_num = int(math.floor((1.0 - cfg['left_intercept_ratio']) * self.left_overlap_num))
        self.left_intercept_num = min(self.left_overlap_num, max(left_intercept_num, self.degree))

        right_intercept_num = int(math.floor((1.0 - cfg['right_intercept_ratio']) * self.right_overlap_num))
        self.right_intercept_num = min(self.right_overlap_num, max(right_intercept_num, self.degree))

        # print(f"seg_idx:{self.idx} length:{len(self)} ")
        # debug_str = f"left_overlap_num:{self.left_overlap_num} " \
        #             f"right_overlap_num:{self.right_overlap_num} " \
        #             f"left_intercept_num:{self.left_intercept_num} " \
        #             f"right_intercept_num:{self.right_intercept_num}\n"
        # print(debug_str)

        self.update_cost_cfg(cfg['cost_setting'])

    def update_cost_cfg(self, cfg: Dict):
        self.cost_cfgs = cfg['cost_cfgs']
        self.path_thickness = cfg['path_thickness']
        self.obstacle_thickness = cfg['obstacle_thickness']

    def compute_spline_length(self, c_xyz: np.ndarray, sample_num_init, sample_change_ratio, atol=1e-2):
        length = BsplineUtils.compute_spline_length(
            c_xyz, self.degree, self.left_clamp, self.right_clamp,
            sample_num_init, sample_change_ratio, atol=atol
        )
        return length

    def update_spline_NpCfg(self, table_xyzr: np.ndarray, update_threshold=0.1, c_radius: np.ndarray = None):
        c_xyz = table_xyzr[self.c_idxs, :3]
        if c_radius is None:
            self.c_radius = table_xyzr[self.c_idxs, 3:4] * self.seg_scale
        else:
            self.c_radius = c_radius
        n_controls = c_xyz.shape[0]

        length_approx = np.linalg.norm(c_xyz[1:, :] - c_xyz[:-1, :], ord=2, axis=1).sum()
        sample_num_approx = max(math.ceil(length_approx), 5)

        length = self.compute_spline_length(c_xyz, sample_num_approx, sample_change_ratio=1.25)

        length_change_ratio = np.abs(length / (self.spline_length + 1e-6) - 1.0)
        if length_change_ratio > update_threshold:
            sample_num = max(math.ceil(length / self.density), 5)
            knot_cfg = BsplineUtils.create_knot_cfg(n_controls, self.degree, self.left_clamp, self.right_clamp)
            knot, fill_last_one = knot_cfg['knot'], knot_cfg['fill_last_one']

            ts = BsplineUtils.create_ts(
                sample_num, self.degree, knot, self.left_clamp, self.right_clamp,
                left_split_num=self.left_intercept_num, right_split_num=self.right_intercept_num
            )
            spline_mat = BsplineUtils.compute_mat(ts, n_controls, self.degree, knot, fill_last_one)

            self.spline_length = length
            self.knot, self.fill_last_one, self.ts = knot, fill_last_one, ts
            self.spline_mat = spline_mat
            self.radius = spline_mat.dot(self.c_radius).reshape(-1)

            if self.with_analytical_tangent or self.with_analytical_curvature:
                self.mat_order1 = BsplineUtils.compute_first_order_mat(ts, n_controls, self.degree, knot)

            if self.with_analytical_curvature:
                self.mat_order2 = BsplineUtils.compute_second_order_mat(ts, n_controls, self.degree, knot)

            print(f"[DEBUG]: Group[{self.group_idx}]/{self.cell_type}_{self.idx} "
                  f"length:{length:.2f} c_num:{n_controls} t_num:{ts.shape[0]} update")
            return True

        return False

    def prepare_tensor(self):
        self.spline_mat_t = TorchUtils.np2tensor(self.spline_mat, require_grad=False)
        if self.with_analytical_tangent or self.with_analytical_curvature:
            self.mat_order1_t = TorchUtils.np2tensor(self.mat_order1, require_grad=False)
        if self.with_analytical_curvature:
            self.mat_order2_t = TorchUtils.np2tensor(self.mat_order2, require_grad=False)

    def update_tensor(self, table_xyz: torch.Tensor):
        c_xyz = table_xyz[self.c_idxs, :3]

        # ------ update spline position
        self.spline_xyz_t = self.spline_mat_t.matmul(c_xyz)

        # ------ update spline tangent vector
        if self.with_analytical_tangent or self.with_analytical_curvature:
            self.spline_vector_t = self.mat_order1_t.matmul(c_xyz)
            if self.cell_type == 'segment':
                if self.left_clamp:
                    self.spline_vector_t[0, 0] = self.left_vector[0]
                    self.spline_vector_t[0, 1] = self.left_vector[1]
                    self.spline_vector_t[0, 2] = self.left_vector[2]
                if self.right_clamp:
                    self.spline_vector_t[-1, 0] = self.right_vector[0]
                    self.spline_vector_t[-1, 1] = self.right_vector[1]
                    self.spline_vector_t[-1, 2] = self.right_vector[2]
            self.spline_vector_t = self.spline_vector_t / torch.norm(self.spline_vector_t, p=2, dim=1, keepdim=True)

        # ------ update spline curvature
        if self.with_analytical_curvature:
            # ------ update spline curvature
            pcd_order1 = self.mat_order1_t.matmul(c_xyz)
            pcd_order2 = self.mat_order2_t.matmul(c_xyz)
            norm_vec = torch.cross(pcd_order1, pcd_order2, dim=1)
            a = torch.norm(norm_vec, p=2, dim=1)
            b = torch.pow(torch.norm(pcd_order1, p=2, dim=1), 3.0)
            curvature = (a / (b + 1e-8))

            if self.left_clamp and self.right_clamp:
                self.spline_curvature_t = curvature[1:-1]
            elif self.left_clamp and (not self.right_clamp):
                self.spline_curvature_t = curvature[1:]
            elif (not self.left_clamp) and self.right_clamp:
                self.spline_curvature_t = curvature[:-1]
            else:
                self.spline_curvature_t = curvature

    def compute_spline_np(self, table_xyzr: np.ndarray):
        c_xyz = table_xyzr[self.c_idxs, :4]
        return self.spline_mat.dot(c_xyz)

    def compute_curvature_np(self, table_xyzr: np.ndarray):
        return BsplineUtils.compute_curvature(self.ts, table_xyzr[self.c_idxs, :3], self.degree, self.knot)

    def compute_tangent_np(self, table_xyzr: np.ndarray):
        c_xyz = table_xyzr[self.c_idxs, :3]
        vec = self.mat_order1.dot(c_xyz)
        if self.cell_type == 'segment':
            if self.left_clamp:
                vec[0] = self.left_vector
            if self.right_clamp:
                vec[-1] = self.right_vector
        vec = vec / np.linalg.norm(vec, ord=2, axis=1, keepdims=True)
        return vec

    def compute_cell_cost(self, table_xyz: torch.Tensor, with_default_weight: bool):
        cost_dict = {}
        for cfg in self.cost_cfgs:
            cost_tag = f"{self.group_idx}/{self.cell_type}_{self.idx}/{cfg['cost_type']}"
            cost_weight = 1.0 if with_default_weight else cfg['weight']

            if cfg['cost_type'] == 'length':
                if cfg['data_type'] == 'spline':
                    cost_dict[cost_tag] = CostUtils.compute_length_cost(self.spline_xyz_t, cfg, cost_weight)
                elif cfg['data_type'] == 'control_points':
                    cost_dict[cost_tag] = CostUtils.compute_length_cost(table_xyz[self.c_idxs, :3], cfg, cost_weight)
                else:
                    raise ValueError(f"[ERROR]: Non-valid data type {cfg['cost_type']}")

            elif cfg['cost_type'] == 'curvature':
                if cfg['curvature_method'] == 'approximate_curvature':
                    cfg['value'] = 1.0 / TorchUtils.np2tensor(self.radius[1:-1] * cfg['curvature_scale'], False)
                    cost_dict[cost_tag] = CostUtils.compute_curvature_cost(self.spline_xyz_t, cfg, cost_weight)

                elif cfg['curvature_method'] == 'analytical_curvature':
                    if self.left_clamp and self.right_clamp:
                        r_radius = self.radius[1:-1]
                    elif self.left_clamp and (not self.right_clamp):
                        r_radius = self.radius[1:]
                    elif (not self.left_clamp) and self.right_clamp:
                        r_radius = self.radius[:-1]
                    else:
                        r_radius = self.radius
                    cfg['value'] = 1.0 / TorchUtils.np2tensor(r_radius * cfg['curvature_scale'], False) * 0.9
                    cost_dict[cost_tag] = CostUtils.compute_curvature_cost(self.spline_curvature_t, cfg, cost_weight)

            elif cfg['cost_type'] == 'length_distribution':
                if cfg['data_type'] == 'spline':
                    cost_dict[cost_tag] = CostUtils.compute_distribution_cost(self.spline_xyz_t, cfg, cost_weight)
                elif cfg['data_type'] == 'control_points':
                    cost_dict[cost_tag] = CostUtils.compute_distribution_cost(
                        table_xyz[self.c_idxs, :3], cfg, cost_weight
                    )
                else:
                    raise ValueError(f"[ERROR]: Non-valid data type {cfg['cost_type']}")

            elif cfg['cost_type'] == 'curvature_distribution':
                if cfg['curvature_method'] == 'approximate_curvature':
                    curvature = TorchMath.compute_approximate_curvature(self.spline_xyz_t)
                    cost_dict[cost_tag] = CostUtils.compute_distribution_cost(curvature, cfg, cost_weight)

                elif cfg['curvature_method'] == 'analytical_curvature':
                    cost_dict[cost_tag] = CostUtils.compute_distribution_cost(
                        self.spline_curvature_t, cfg, cost_weight
                    )

            elif cfg['cost_type'] == 'clamp_vector':
                if self.left_clamp:
                    idxs_range = max(math.floor(cfg['vector_length'] / self.density), 2)
                    cost_dict[cost_tag] = CostUtils.compute_vector_cost(
                        self.spline_xyz_t[:idxs_range, :], cfg, 'left2any', cost_weight
                    )

                if self.right_clamp:
                    idxs_range = max(math.floor(cfg['vector_length'] / self.density), 2)
                    cost_dict[cost_tag] = CostUtils.compute_vector_cost(
                        self.spline_xyz_t[-idxs_range:, :], cfg, 'any2right', cost_weight
                    )

            else:
                raise ValueError(f"[ERROR]: Non-valid seg cost type {cfg['cost_type']}")

        return cost_dict

    def export_cfg(self):
        cfg = {
            'seg_idx': self.idx,
            'c_idxs': self.c_idxs.tolist(),
            'group_idx': self.group_idx,
            'cell_type': self.cell_type,
            'left_clamp': self.left_clamp,
            'right_clamp': self.right_clamp,
            'left_vector': self.left_vector.tolist(),
            'right_vector': self.right_vector.tolist(),
            # just for front end
            'left_intercept_num': self.left_intercept_num,
            'right_intercept_num': self.right_intercept_num
        }        
        return cfg

    @staticmethod
    def restore_cfg(cfg: Dict, parent_cell):
        seg_cell = SegmentCell(cfg['seg_idx'], cfg['group_idx'], np.array(cfg['c_idxs']), parent_cell)
        seg_cell.left_clamp = cfg['left_clamp']
        seg_cell.right_clamp = cfg['right_clamp']
        seg_cell.left_vector = np.array(cfg['left_vector'])
        seg_cell.right_vector = np.array(cfg['right_vector'])
        return seg_cell


class ConnectorCell(SegmentCell):
    def __init__(self, idx: int, group_idx: int, left_seg_idx: int, right_seg_idx: int, parent_cell):
        self.idx = idx
        self.group_idx = group_idx
        self.left_seg_idx = left_seg_idx
        self.right_seg_idx = right_seg_idx
        self.left_seg: SegmentCell = None
        self.right_seg: SegmentCell = None
        self.c_idxs: np.ndarray = None
        self.parent_cell: GroupCell = parent_cell
        self.cell_type: Literal["segment", "connector"] = 'connector'

        # ------ connector cfg
        self.left_clamp = False
        self.right_clamp = False
        self.left_intercept_num: int = None
        self.right_intercept_num: int = None

        # ------ B-spline cfg
        self.degree: int = None
        self.spline_length: float = 0.0
        self.knot: np.ndarray = None
        self.fill_last_one: bool = None
        self.ts: np.ndarray = None

        self.spline_mat: np.ndarray = None
        self.mat_order1: np.ndarray = None
        self.mat_order2: np.ndarray = None
        self.radius: np.ndarray = None
        self.c_radius: np.ndarray = None

        # ------ Cost cfg
        self.cost_cfgs: List[Dict] = []
        self.path_thickness = 0.01
        self.obstacle_thickness = 0.01

        # ------ Tensor
        self.spline_mat_t: torch.Tensor = None
        self.mat_order1_t: torch.Tensor = None
        self.mat_order2_t: torch.Tensor = None
        self.spline_xyz_t: torch.Tensor = None
        self.spline_curvature_t: torch.Tensor = None

        self.with_analytical_curvature = False
        self.with_analytical_tangent = False

    def init_cfg(self, cfg: Dict, seg_dict: Dict[int, SegmentCell]):
        # todo 共用点导致自由度依然不足够，需要独立开来

        self.density = cfg['bspline_density']
        self.with_analytical_curvature = cfg['with_analytical_curvature_computed']
        self.with_analytical_tangent = cfg['with_analytical_tangent_computed']
        self.left_seg = seg_dict[self.left_seg_idx]
        self.right_seg = seg_dict[self.right_seg_idx]

        assert self.left_seg.degree == self.right_seg.degree
        self.degree = self.left_seg.degree

        left_c_idxs = self.left_seg.c_idxs.copy()
        left_overlap_num = self.left_seg.right_overlap_num
        left_other_intercept_num = self.left_seg.right_intercept_num

        right_c_idxs = self.right_seg.c_idxs.copy()
        right_overlap_num = self.right_seg.left_overlap_num
        right_other_overlap_num = self.right_seg.left_intercept_num

        if left_c_idxs[-1] == right_c_idxs[0]:
            pass

        elif left_c_idxs[0] == right_c_idxs[0]:
            left_c_idxs = left_c_idxs[::-1]
            left_overlap_num = self.left_seg.left_overlap_num
            left_other_intercept_num = self.left_seg.left_intercept_num

        elif left_c_idxs[-1] == right_c_idxs[-1]:
            right_c_idxs = right_c_idxs[::-1]
            right_overlap_num = self.right_seg.right_overlap_num
            right_other_overlap_num = self.right_seg.right_intercept_num

        elif left_c_idxs[0] == right_c_idxs[1]:
            left_c_idxs = left_c_idxs[::-1]
            left_overlap_num = self.left_seg.left_overlap_num
            left_other_intercept_num = self.left_seg.left_intercept_num

            right_c_idxs = right_c_idxs[::-1]
            right_overlap_num = self.right_seg.right_overlap_num
            right_other_overlap_num = self.right_seg.right_intercept_num

        else:
            raise ValueError("[ERROR]: Non-Valid connector.")

        self.c_idxs = np.concatenate([left_c_idxs[-left_overlap_num:], right_c_idxs[1:right_overlap_num]], axis=0)
        # degree + overlap_num = left_intercept_num + right_intercept_num
        self.left_intercept_num = left_overlap_num + self.degree - left_other_intercept_num
        self.right_intercept_num = right_overlap_num + self.degree - right_other_overlap_num

        # print(f"con_idx:{self.idx} length:{len(self)} "
        #       f"left_overlap_num:{left_overlap_num} right_overlap_num:{right_overlap_num}")
        # debug_str = f"left_intercept_num:{self.left_intercept_num} " \
        #             f"right_intercept_num:{self.right_intercept_num}\n"
        # print(debug_str)

        self.update_cost_cfg(cfg['cost_setting'])

    def update_spline_NpCfg(self, table_xyzr: np.ndarray, update_threshold=0.1):
        seg_scale = self.parent_cell.c_table['seg_scale'].values
        orig_radius = table_xyzr[self.c_idxs, -1] * seg_scale[self.c_idxs]
        c_radius = np.linspace(orig_radius[0], orig_radius[-1], num=self.c_idxs.shape[0])
        return super().update_spline_NpCfg(table_xyzr, update_threshold, c_radius)

    def export_cfg(self):
        cfg = {
            'con_idx': self.idx,
            'group_idx': self.group_idx,
            'cell_type': self.cell_type,
            'left_seg_idx': self.left_seg_idx,
            'right_seg_idx': self.right_seg_idx,
            # just for front end
            'c_idxs': self.c_idxs.tolist(),
            'left_intercept_num': self.left_intercept_num,
            'right_intercept_num': self.right_intercept_num
        }        
        return cfg

    @staticmethod
    def restore_cfg(cfg: Dict, parent_cell):
        con_cell = ConnectorCell(cfg['con_idx'], cfg['group_idx'], cfg['left_seg_idx'], cfg['right_seg_idx'], parent_cell)
        return con_cell


class AuxCostUtil(object):
    @staticmethod
    def compute_point2line_dist(pcd: np.ndarray, pcd_of_line: np.ndarray, vec_of_line: np.ndarray):
        """ assume vec_of_line is normal vector"""
        line2point = pcd - pcd_of_line
        line2point_length = np.linalg.norm(line2point, ord=2, axis=1)
        cos_length = np.sum(line2point * vec_of_line, axis=1)
        return np.sqrt(np.power(line2point_length, 2.0) - np.power(cos_length, 2.0))

    @staticmethod
    def project_point_to_line(pcd: np.ndarray, pcd_of_line: np.ndarray, vec_of_line: np.ndarray):
        a = np.sum((pcd - pcd_of_line) * vec_of_line, axis=1, keepdims=True)
        b = np.sum(vec_of_line * vec_of_line, axis=1, keepdims=True)
        t_closest = a / b
        project_pcd = pcd_of_line + t_closest * vec_of_line
        return project_pcd

    @staticmethod
    def is_pcd_in_same_plane(left_pcd: np.ndarray, right_pcd: np.ndarray, vec: np.ndarray):
        """assume left_vec and right_vec is normal vector"""
        left2right_vec = right_pcd - left_pcd
        return np.sum(left2right_vec * vec, axis=1) == 0.0

    @staticmethod
    def detect_path_self_conflict(path_pcd: np.ndarray, radius: np.ndarray, unit_length=0.0):
        tree = KDTree(path_pcd)
        search_radius = radius + np.max(radius)
        idxs_list, dist_list = tree.query_radius(path_pcd, search_radius, return_distance=True)

        conflict_pair_idxs = []
        for pcd_idx, (relate_idxs, dists) in enumerate(zip(idxs_list, dist_list)):
            conflict_bool = dists < (radius[pcd_idx] + radius[relate_idxs] + unit_length)
            relate_idxs = relate_idxs[conflict_bool]
            if len(relate_idxs) == 0:
                continue

            shift_idx = relate_idxs - pcd_idx
            while True:
                is_connect = np.any(shift_idx == 0)
                if is_connect:
                    shift_idx = shift_idx - 1.0
                else:
                    break
            conflict_larger_idx = relate_idxs[shift_idx > 0.0]

            shift_idx = relate_idxs - pcd_idx
            while True:
                is_connect = np.any(shift_idx == 0)
                if is_connect:
                    shift_idx = shift_idx + 1.0
                else:
                    break
            conflict_less_idx = relate_idxs[shift_idx < 0.0]

            conflict_idxs = np.concatenate([conflict_larger_idx, conflict_less_idx], axis=-1)
            if conflict_idxs.shape[0] == 0:
                continue

            conflict_pair_idxs.append(
                np.concatenate([
                    np.full((conflict_idxs.shape[0], 1), fill_value=pcd_idx),
                    conflict_idxs.reshape((-1, 1))
                ], axis=1)
            )

        if len(conflict_pair_idxs) > 0:
            return np.concatenate(conflict_pair_idxs, axis=0)
        else:
            return np.array([])

    @staticmethod
    def compute_intersect_pcd(
            left_pcd: np.ndarray, left_vec: np.ndarray, right_pcd: np.ndarray, right_vec: np.ndarray,
            is_project_mid_pcd=False, project_orig_pcd=False
    ):
        res_pack = {}

        pair_num = left_pcd.shape[0]
        vec_inter = np.cross(left_vec, right_vec, axisa=1)
        vec_inter = vec_inter / np.linalg.norm(vec_inter, axis=1, ord=2, keepdims=True)

        A = np.concatenate([
            np.tile(left_vec[:, np.newaxis, :], (1, 1, 1)),
            np.tile(right_vec[:, np.newaxis, :], (1, 1, 1)),
            np.tile(vec_inter[:, np.newaxis, :], (1, 1, 1))
        ], axis=-2)
        b = np.concatenate([
            np.tile(np.sum(left_vec * left_pcd, axis=1, keepdims=True)[:, np.newaxis, :], (1, 1, 1)),
            np.tile(np.sum(right_vec * right_pcd, axis=1, keepdims=True)[:, np.newaxis, :], (1, 1, 1)),
            np.zeros((pair_num, 1, 1))
        ], axis=-2)
        pcd_inter = np.linalg.solve(A, b)
        pcd_inter = pcd_inter.reshape((-1, 3))

        # ------ compute the mid point project to intersection line
        if is_project_mid_pcd:
            mid_pcd = (left_pcd + right_pcd) * 0.5
            pcd_inter = AuxCostUtil.project_point_to_line(mid_pcd, pcd_inter, vec_inter)

        if project_orig_pcd:
            res_pack['left_project_pcd'] = AuxCostUtil.project_point_to_line(left_pcd, pcd_inter, vec_inter)
            res_pack['right_project_pcd'] = AuxCostUtil.project_point_to_line(right_pcd, pcd_inter, vec_inter)

        # # ------ check
        # left2inter_vec = pcd_inter - left_pcd
        # print(f"[CHECK]: Orthogonal to left:{np.all(np.isclose(np.sum(left2inter_vec * left_vec, axis=1), 0.0))}")
        # right2inter_vec = pcd_inter - right_pcd
        # print(f"[CHECK]: Orthogonal to right:{np.all(np.isclose(np.sum(right2inter_vec * right_vec, axis=1), 0.0))}")
        # left_project = AuxCostUtil.compute_point_project2line(left_pcd, pcd_inter, vec_inter)
        # print(f"[CHECK]: Orthogonal to left pcd:{np.all(np.isclose(np.sum((left_project - left_pcd) * vec_inter, axis=1), 0.0))}")
        # right_project = AuxCostUtil.compute_point_project2line(right_pcd, pcd_inter, vec_inter)
        # print(f"[CHECK]: Orthogonal to right pcd:{np.all(np.isclose(np.sum((right_project - right_pcd) * vec_inter, axis=1), 0.0))}")
        #
        # for xyz0, vec0, xyz1, vec1, xyz2, vec2 in zip(
        #         left_pcd, left_vec, right_pcd, right_vec, pcd_inter, vec_inter
        # ):
        #     plane_left = VisUtils.create_plane(xyz0, vec0, x_size=5, y_size=5)
        #     plane_right = VisUtils.create_plane(xyz1, vec1, x_size=5, y_size=5)
        #     inter_axis = VisUtils.create_cylinder(xyz2, vec2, 0.01, 5.0)
        #
        #     sphere0 = VisUtils.create_sphere(xyz0, radius=0.05)
        #     sphere1 = VisUtils.create_sphere(xyz1, radius=0.05)
        #     mid_sphere = VisUtils.create_sphere(xyz2, radius=0.05)
        #
        #     vis = VisUtils()
        #     vis.plot(plane_left, color=(0., 1., 0.), opacity=0.35)
        #     vis.plot(plane_right, color=(0., 0., 1.), opacity=0.35)
        #     vis.plot(inter_axis, color=(1., 0., 0.))
        #
        #     vis.plot(sphere0, color=(0., 1., 0.))
        #     vis.plot(sphere1, color=(0., 0., 1.))
        #     vis.plot(mid_sphere, color=(1., 0., 0.))
        #
        #     vis.show()
        # # ------

        res_pack['pcd_inter'] = pcd_inter
        res_pack['vec_inter'] = vec_inter

        return res_pack

    @staticmethod
    def detect_adjacent_curvature_conflict(
            path_pcd: np.ndarray, path_vec: np.ndarray, path_radius: np.ndarray,
            is_project_mid_pcd, project_orig_pcd, radius_scale=1.0,
    ):
        """ assume path_vec is normal vector"""
        pcd_idxs = np.arange(0, path_pcd.shape[0], 1)
        left_idxs, right_idxs = pcd_idxs[:-1], pcd_idxs[1:]

        inter_vec = np.cross(path_vec[left_idxs, :], path_vec[right_idxs, :], axisa=1)
        valid_bool = np.bitwise_and(
            np.sum(inter_vec, axis=1) != 0., ~np.isnan(np.sum(inter_vec, axis=1))
        )
        left_idxs = left_idxs[valid_bool]
        right_idxs = right_idxs[valid_bool]

        res_pack = AuxCostUtil.compute_intersect_pcd(
            path_pcd[left_idxs, :], path_vec[left_idxs, :], path_pcd[right_idxs, :], path_vec[right_idxs, :],
            is_project_mid_pcd, project_orig_pcd
        )
        inter_pcd, inter_vec = res_pack['pcd_inter'], res_pack['vec_inter']

        left2line_dist = AuxCostUtil.compute_point2line_dist(path_pcd[left_idxs, :], inter_pcd, inter_vec)
        right2line_dist = AuxCostUtil.compute_point2line_dist(path_pcd[right_idxs, :], inter_pcd, inter_vec)
        target_length = (path_radius[left_idxs] + path_radius[right_idxs]) * radius_scale
        conflict_bool = target_length - (left2line_dist + right2line_dist) > 0.0

        conflict_pair = np.concatenate([
            left_idxs[conflict_bool].reshape((-1, 1)), right_idxs[conflict_bool].reshape((-1, 1))
        ], axis=1)

        res_pack['pcd_inter'] = inter_pcd[conflict_bool, :]
        res_pack['vec_inter'] = inter_vec[conflict_bool, :]

        return conflict_pair, res_pack


class PathCell(object):
    def __init__(
            self, path_name: str, idx: int, group_idx: int, cell_idxs: List[int], cell_types: List[str], parent_cell
    ):
        self.path_name = path_name
        self.idx = idx
        self.group_idx = group_idx

        self.parent_cell: GroupCell = parent_cell
        self.cell_idxs = cell_idxs
        self.cell_types = cell_types
        self.cost_cfgs: List[Dict] = []
        self.element_cost_cfgs: List[Dict] = []

        self.spline_xyz_t: torch.Tensor = None
        self.radius: np.ndarray = None
        self.spline_tangent_t: torch.Tensor = None

    def update_tensor(self, with_analytical_tangent=False):
        # print(f"Group idx:{self.group_idx}, name:{self.path_name}, idx:{self.idx}")
        path_xyz_list, path_radius = [], []
        if with_analytical_tangent:
            path_tangent_list = []

        last_cell: Union[SegmentCell, ConnectorCell] = None
        for i in range(len(self.cell_idxs)):
            if self.cell_types[i] == 'segment':
                cell = self.parent_cell.seg_dict[self.cell_idxs[i]]
            else:
                cell = self.parent_cell.connector_dict[self.cell_idxs[i]]

            if len(self.cell_idxs) > 1:
                if i == 0:
                    if self.cell_types[i + 1] == 'segment':
                        next_cell = self.parent_cell.seg_dict[self.cell_idxs[i + 1]]
                    else:
                        next_cell = self.parent_cell.connector_dict[self.cell_idxs[i + 1]]

                    if cell.c_idxs[-1] in next_cell.c_idxs:
                        path_xyz_list.append(cell.spline_xyz_t)
                        path_radius.append(cell.radius)
                        if with_analytical_tangent:
                            path_tangent_list.append(cell.spline_vector_t)

                    elif cell.c_idxs[0] in next_cell.c_idxs:
                        path_xyz_list.append(cell.spline_xyz_t.flip(0))
                        path_radius.append(cell.radius[::-1])
                        if with_analytical_tangent:
                            path_tangent_list.append(cell.spline_vector_t * -1.0)

                    else:
                        raise ValueError(f"[ERROR]: Wrong path flags.")

                    # # ------ debug visualization
                    # if cell.c_idxs[-1] in next_cell.c_idxs:
                    #     tmp_xyz = TorchUtils.tensor2np(cell.spline_xyz_t)
                    #     print(tmp_xyz[0], tmp_xyz[-1])
                    # elif cell.c_idxs[0] in next_cell.c_idxs:
                    #     tmp_xyz = TorchUtils.tensor2np(cell.spline_xyz_t.flip(0))
                    #     print(tmp_xyz[0], tmp_xyz[-1])
                    # # ------

                else:
                    if cell.c_idxs[0] in last_cell.c_idxs:
                        path_xyz_list.append(cell.spline_xyz_t[1:])
                        path_radius.append(cell.radius[1:])
                        if with_analytical_tangent:
                            path_tangent_list.append(cell.spline_vector_t[1:])

                    elif cell.c_idxs[-1] in last_cell.c_idxs:
                        path_xyz_list.append(cell.spline_xyz_t.flip(0)[1:])
                        path_radius.append(cell.radius[::-1][1:])
                        if with_analytical_tangent:
                            path_tangent_list.append(cell.spline_vector_t[1:] * -1.0)

                    else:
                        raise ValueError(f"[ERROR]: Wrong path flags.")

                    # # ------ debug visualization
                    # if cell.c_idxs[0] in last_cell.c_idxs:
                    #     tmp_xyz = TorchUtils.tensor2np(cell.spline_xyz_t)
                    #     print(tmp_xyz[0], tmp_xyz[-1])
                    # elif cell.c_idxs[-1] in last_cell.c_idxs:
                    #     tmp_xyz = TorchUtils.tensor2np(cell.spline_xyz_t.flip(0))
                    #     print(tmp_xyz[0], tmp_xyz[-1])
                    # # ------
            
            else:
                path_xyz_list.append(cell.spline_xyz_t)
                path_radius.append(cell.radius)
                if with_analytical_tangent:
                    path_tangent_list.append(cell.spline_vector_t)

            last_cell = cell

        self.spline_xyz_t = torch.concatenate(path_xyz_list, dim=0)
        self.radius = np.concatenate(path_radius)
        if with_analytical_tangent:
            self.spline_tangent_t = torch.concatenate(path_tangent_list, dim=0)

    def compute_path_np_info(self):
        path_xyz_list, path_radius, path_tangent_list = [], [], []

        last_cell: Union[SegmentCell, ConnectorCell] = None
        table_xyzr = self.parent_cell.c_table[['x', 'y', 'z', 'radius']].values

        for i in range(len(self.cell_idxs)):
            if self.cell_types[i] == 'segment':
                cell = self.parent_cell.seg_dict[self.cell_idxs[i]]
            else:
                cell = self.parent_cell.connector_dict[self.cell_idxs[i]]

            if i == 0:
                if self.cell_types[i + 1] == 'segment':
                    next_cell = self.parent_cell.seg_dict[self.cell_idxs[i + 1]]
                else:
                    next_cell = self.parent_cell.connector_dict[self.cell_idxs[i + 1]]

                if cell.c_idxs[-1] in next_cell.c_idxs:
                    path_xyz_list.append(cell.compute_spline_np(table_xyzr)[:, :3])
                    path_tangent_list.append(cell.compute_tangent_np(table_xyzr))
                    path_radius.append(cell.radius)
                elif cell.c_idxs[0] in next_cell.c_idxs:
                    path_xyz_list.append(cell.compute_spline_np(table_xyzr)[:, :3][::-1])
                    path_tangent_list.append(cell.compute_tangent_np(table_xyzr) * -1.0)
                    path_radius.append(cell.radius[::-1])
                else:
                    raise ValueError(f"[ERROR]: Wrong path flags.")

            else:
                if cell.c_idxs[0] in last_cell.c_idxs:
                    path_xyz_list.append(cell.compute_spline_np(table_xyzr)[:, :3][1:])
                    path_tangent_list.append(cell.compute_tangent_np(table_xyzr)[1:])
                    path_radius.append(cell.radius[1:])
                elif cell.c_idxs[-1] in last_cell.c_idxs:
                    path_xyz_list.append(cell.compute_spline_np(table_xyzr)[:, :3][::-1][1:])
                    path_tangent_list.append(cell.compute_tangent_np(table_xyzr)[1:] * -1.0)
                    path_radius.append(cell.radius[::-1][1:])
                else:
                    raise ValueError(f"[ERROR]: Wrong path flags.")

            last_cell = cell

        return {
            'spline_xyz': np.concatenate(path_xyz_list, axis=0),
            'spline_tangent': np.concatenate(path_tangent_list, axis=0),
            'radius': np.concatenate(path_radius)
        }

    def init_cfg(self, cfg: Dict):
        self.update_cost_cfg(cfg['cost_setting'])

    def update_cost_cfg(self, cfg: Dict):
        self.cost_cfgs = cfg['cost_cfgs']
        self.element_cost_cfgs = cfg['element_cost_cfgs']

    def compute_set_cost(self, unit_length: float, with_default_weight: bool):
        cost_dict = {}
        for cfg in self.cost_cfgs:
            cost_tag = f"{self.path_name}/{self.idx}/{cfg['cost_type']}"
            cost_weight = 1.0 if with_default_weight else cfg['weight']

            if cfg['cost_type'] == 'self_conflict':
                spline_pcd_np = TorchUtils.tensor2np(self.spline_xyz_t)

                radius_conflict_pair = AuxCostUtil.detect_path_self_conflict(
                    spline_pcd_np, self.radius, unit_length
                )
                if radius_conflict_pair.shape[0] > 0:
                    require_dist = self.radius[radius_conflict_pair[:, 0]] + self.radius[radius_conflict_pair[:, 1]]
                    self_conflict_cost = CostUtils.compute_shape_conflict_cost(
                        self.spline_xyz_t[radius_conflict_pair[:, 0]], self.spline_xyz_t[radius_conflict_pair[:, 1]],
                        require_dist=require_dist, weight=1.0
                    )
                    cost_dict[cost_tag] = self_conflict_cost.mean() * cost_weight

            elif cfg['cost_type'] == 'crossSection_curvature':
                # ------ non-analytical tangent
                left_tangent = self.spline_xyz_t[1:-1] - self.spline_xyz_t[:-2]
                right_tangent = self.spline_xyz_t[2:] - self.spline_xyz_t[1:-1]
                spline_tangent_t = (left_tangent + right_tangent) * 0.5
                spline_tangent_t = spline_tangent_t / torch.norm(spline_tangent_t, p=2, dim=1, keepdim=True)

                spline_pcd_t = self.spline_xyz_t[1:-1]
                spline_radius = self.radius[1:-1]
                # ------

                # ------ analytical tangent
                # spline_tangent_t = self.spline_tangent_t
                # spline_pcd_t = self.spline_xyz_t
                # spline_radius = self.radius
                # ------

                spline_pcd_np = TorchUtils.tensor2np(spline_pcd_t)
                spline_tangent_np = TorchUtils.tensor2np(spline_tangent_t)
                cur_conflict_pair, res_pack = AuxCostUtil.detect_adjacent_curvature_conflict(
                    spline_pcd_np, spline_tangent_np, spline_radius,
                    is_project_mid_pcd=True, project_orig_pcd=False, radius_scale=3.0
                )

                if cur_conflict_pair.shape[0] > 0:
                    # ------ np process: Fail
                    inter_pcd_t = TorchUtils.np2tensor(res_pack['pcd_inter'], require_grad=False)
                    inter_vec_t = TorchUtils.np2tensor(res_pack['vec_inter'], require_grad=False)

                    # # ------ torch process: Fail
                    # left_pcd_t = path_pcd_t[cur_conflict_pair[:, 0], :]
                    # left_vec_t = path_vec_t[cur_conflict_pair[:, 0], :]
                    #
                    # right_pcd_t = path_pcd_t[cur_conflict_pair[:, 1], :]
                    # right_vec_t = path_vec_t[cur_conflict_pair[:, 1], :]
                    #
                    # res_pack_t = TorchMath.compute_intersect_pcd(
                    #     left_pcd_t, left_vec_t, right_pcd_t, right_vec_t,
                    #     is_project_mid_pcd=True, project_orig_pcd=False
                    # )
                    # inter_pcd_t = res_pack_t['pcd_inter']
                    # inter_vec_t = res_pack_t['vec_inter']

                    # ------ Orthogonal distance Fail
                    left2line_dist = TorchMath.compute_dist_point2line(
                        spline_pcd_t[cur_conflict_pair[:, 0]], inter_pcd_t, inter_vec_t
                    )
                    right2line_dist = TorchMath.compute_dist_point2line(
                        spline_pcd_t[cur_conflict_pair[:, 1]], inter_pcd_t, inter_vec_t
                    )
                    target_scale_t = TorchUtils.np2tensor(
                        (spline_radius[cur_conflict_pair[:, 0]] + spline_radius[cur_conflict_pair[:, 1]]) * 3.0,
                        require_grad=False
                    )

                    cost = (target_scale_t - (left2line_dist + right2line_dist)).mean()
                    cost_dict[cost_tag] = cost * cost_weight

            elif cfg['cost_type'] == 'tangent_diffusion':
                left_tangent = self.spline_xyz_t[1:-1] - self.spline_xyz_t[:-2]
                right_tangent = self.spline_xyz_t[2:] - self.spline_xyz_t[1:-1]

                cos_value = TorchMath.compute_cos_val(left_tangent, right_tangent)
                rad_value = torch.arccos(cos_value)
                cost_dict[cost_tag] = CostUtils.compute_distribution_cost(rad_value, cfg, cost_weight)

            else:
                raise ValueError(f"[ERROR]: Non-valid seg cost type {cfg['cost_type']}")

        return cost_dict

    def compute_element_cost(self, with_default_weight: bool):
        raise NotImplementedError

    def export_cfg(self):
        cfg = {
            'path_idx': self.idx,
            'path_name': self.path_name,
            'group_idx': self.group_idx,
            'cell_idxs': self.cell_idxs,
            'cell_types': self.cell_types,
        }
        return cfg

    @staticmethod
    def restore_cfg(cfg: Dict, parent_cell):
        path_cell = PathCell(
            cfg['path_name'], cfg['path_idx'], cfg['group_idx'], cfg['cell_idxs'], cfg['cell_types'], parent_cell
        )
        return path_cell


class PairCell(object):
    def __init__(
            self, idx: int, group_idx: int,
            left_idx: int, left_type: str, right_idx: int, right_type: str,
            parent_cell
    ):
        self.idx = idx
        self.group_idx = group_idx

        self.parent_cell: GroupCell = parent_cell
        self.left_idx = left_idx
        self.left_type = left_type
        self.right_idx = right_idx
        self.right_type = right_type

        self.cost_cfgs: List[Dict] = []

    def init_cfg(self, cfg: Dict):
        self.update_cost_cfg(cfg['cost_setting'])

    def update_cost_cfg(self, cfg: Dict):
        self.cost_cfgs = cfg['cost_cfgs']

    @staticmethod
    def find_conflict(
            unit_length: float,
            left_cell: Union[SegmentCell, ConnectorCell], right_cell: Union[SegmentCell, ConnectorCell]
    ):
        left_xyz = TorchUtils.tensor2np(left_cell.spline_xyz_t)
        left_radius = left_cell.radius
        right_xyz = TorchUtils.tensor2np(right_cell.spline_xyz_t)
        right_radius = right_cell.radius

        left_tree = KDTree(left_xyz)
        search_radius = np.max(left_radius) + np.max(right_radius) + unit_length * 2.0
        idxs_list, dist_list = left_tree.query_radius(right_xyz, search_radius, return_distance=True)

        left_idxs, right_idxs, require_radius = [], [], []
        for right_idx, (left_idx_list, dists) in enumerate(zip(idxs_list, dist_list)):
            if len(left_idx_list) == 0:
                continue

            for left_idx, dist in zip(left_idx_list, dists):
                detection_radius = left_radius[left_idx] + right_radius[right_idx] + unit_length * 2.0
                if dist <= detection_radius:
                    left_idxs.append(left_idx)
                    right_idxs.append(right_idx)
                    require_radius.append(left_radius[left_idx] + right_radius[right_idx])

        return np.array(left_idxs), np.array(right_idxs), np.array(require_radius)

    def compute_cost(self, unit_length: float, with_default_weight: bool):
        cost_dict = {}
        for cfg in self.cost_cfgs:
            cost_tag = f"{self.group_idx}/pair_segment/{self.idx}/{cfg['cost_type']}"
            cost_weight = 1.0 if with_default_weight else cfg['weight']

            if self.left_type == 'segment':
                left_cell = self.parent_cell.seg_dict[self.left_idx]
            elif self.left_type == 'connector':
                left_cell = self.parent_cell.connector_dict[self.left_idx]
            else:
                raise ValueError

            if self.right_type == 'segment':
                right_cell = self.parent_cell.seg_dict[self.right_idx]
            elif self.right_type == 'connector':
                right_cell = self.parent_cell.connector_dict[self.right_idx]
            else:
                raise ValueError

            if cfg['cost_type'] == 'pair_conflict':
                left_idxs, right_idxs, require_dist = self.find_conflict(unit_length, left_cell, right_cell)
                if left_idxs.shape[0] > 0:
                    costs = CostUtils.compute_shape_conflict_cost(
                        pcd0=left_cell.spline_xyz_t[left_idxs, :],
                        pcd1=right_cell.spline_xyz_t[right_idxs, :],
                        require_dist=require_dist,
                        weight=1.0
                    )
                    cost_dict[cost_tag] = costs.mean() * cost_weight

            else:
                raise ValueError(f"[ERROR]: Non-valid pair seg cost type {cfg['cost_type']}")

        return cost_dict


class GroupCell(object):
    def __init__(self, group_idx: int):
        self.group_idx = group_idx
        self.c_table: pd.DataFrame = None
        self.seg_dict: Dict[int, SegmentCell] = {}
        self.connector_dict: Dict[int, ConnectorCell] = {}
        self.path_dict: Dict[int, PathCell] = {}
        self.pair_seg_dict: Dict[int, PairCell] = {}
        self.table_xyz_t: torch.Tensor = None

    @staticmethod
    def convert_to_flag_df(aggr_df: pd.DataFrame):
        """two more columns: [is_unique, unique_idx]"""
        aggr_df['df_idx'] = np.arange(0, aggr_df.shape[0], 1)
        aggr_df.sort_values(by=['x', 'y', 'z'], inplace=True)
        aggr_xyz = aggr_df[['x', 'y', 'z']].values
        unique_bool = ~(np.sum(np.abs(aggr_xyz[1:] - aggr_xyz[:-1]), axis=1) < 1e-6)
        unique_bool = np.concatenate([np.array([True]), unique_bool], axis=0)
        aggr_df['is_unique'] = unique_bool.astype(int)
        aggr_df['unique_idx'] = aggr_df['is_unique'].cumsum() - 1
        aggr_df.sort_values(by=['df_idx'], inplace=True)
        aggr_df.drop(columns=['df_idx'], inplace=True)
        return aggr_df

    @staticmethod
    def encode_segment_tag(flags: List[int]):
        if len(flags) < 4:
            label_flags = flags
        else:
            label_flags = [flags[0], flags[1], flags[-2], flags[-1]]
        return tuple(np.sort(np.array(label_flags)))

    @staticmethod
    def refit_xyzr(c_xyzr: np.ndarray, seg_degree_min: int, seg_degree_max: int):
        seg_length = c_xyzr.shape[0]
        if seg_length < seg_degree_min:
            while seg_length < seg_degree_min:
                expand_xyzr = []
                for i in range(seg_length - 1):
                    expand_xyzr.append(c_xyzr[i])
                    expand_xyzr.append((c_xyzr[i] + c_xyzr[i + 1]) * 0.5)
                expand_xyzr.append(c_xyzr[-1])
                c_xyzr = np.array(expand_xyzr)
                seg_length = c_xyzr.shape[0]

        seg_length = c_xyzr.shape[0]
        if seg_length > seg_degree_max:
            select_idxs = np.arange(1, seg_length - 1, 1)  # 第1个与最后1个不用于采样
            select_idxs = np.random.choice(select_idxs, size=min(seg_degree_max, select_idxs.shape[0]), replace=False)
            select_idxs = np.sort(select_idxs)
            c_xyzr = np.concatenate([c_xyzr[:1, :], c_xyzr[select_idxs, :], c_xyzr[-1:, :]], axis=0)

        return c_xyzr

    def assemble_discrete_graph(
            self,
            path_dict: Dict[str, np.ndarray],
            assemble_cfg: Dict,
            aid_path_info: List[Dict],
    ):
        # ------ Step 1 construct graph
        tmp_graph_df = []
        for task_name, xyzr_np in path_dict.items():
            sub_df = pd.DataFrame(xyzr_np, columns=['x', 'y', 'z', 'radius'])
            sub_df['task_name'] = task_name
            tmp_graph_df.append(sub_df)
        tmp_graph_df = pd.concat(tmp_graph_df, axis=0)
        tmp_graph_df = self.convert_to_flag_df(tmp_graph_df)

        tmp_graph, tmp_node2xyzr = nx.Graph(), {}
        for (task_name,), sub_df in tmp_graph_df.groupby(by=['task_name']):
            last_flag = None
            for i, data_point in enumerate(sub_df[['x', 'y', 'z', 'radius', 'unique_idx']].values):
                flag = int(data_point[-1])
                tmp_node2xyzr[flag] = data_point[:4]
                if i > 0:
                    tmp_graph.add_edge(last_flag, flag)
                last_flag = flag

        # ------ Step 2 find all terminal nodes
        terminal_xyz2flags = {}
        for flag in tmp_graph.nodes:
            if tmp_graph.degree(flag) == 1:
                xyzr = tmp_node2xyzr[flag]
                terminal_xyz2flags[(xyzr[0], xyzr[1], xyzr[2])] = flag

        # ------ Step 3 travel all path to find all segments
        seg_book, connector_book, terminal_seg_book, path_contain_seg_dict = {}, [], {}, {}
        for path_info in aid_path_info:
            path_name = path_info['name']

            from_pos = path_info['from_position']
            from_flag = terminal_xyz2flags[(from_pos[0], from_pos[1], from_pos[2])]
            from_vector = path_info['from_vector']
            from_is_input = path_info['from_is_input']

            to_pos = path_info['to_position']
            to_flag = terminal_xyz2flags[(to_pos[0], to_pos[1], to_pos[2])]
            to_vector = path_info['to_vector']
            to_is_input = path_info['to_is_input']

            # ------ use to record path contain seg
            path_contain_seg_dict[path_name] = []

            path_flags_list = nx.all_simple_paths(tmp_graph, from_flag, to_flag)
            for path_flags in path_flags_list:
                seg_flag, path_contain_seg = [], []
                path_length = len(path_flags)
                seg_book_old_tags = list(seg_book.keys()).copy()

                for i, flag in enumerate(path_flags):
                    seg_flag.append(flag)
                    if tmp_graph.degree(flag) <= 2 and (i < path_length - 1):  # node is in the middle and degree = 2
                        continue

                    seg_flag_data = seg_flag.copy()
                    seg_tag = self.encode_segment_tag(seg_flag_data)
                    path_contain_seg.append(seg_tag)

                    # ------ if it is new segment then record it
                    if seg_tag not in seg_book.keys():
                        seg_book[seg_tag] = seg_flag_data

                    # ------ if more than 2 segment then generate connector
                    if len(path_contain_seg) > 1:
                        connector_book.append([path_contain_seg[-2], path_contain_seg[-1]])

                    seg_flag = seg_flag[-1:]  # save as begin of segment

                # ------ add clamp information but only once
                first_tag = path_contain_seg[0]
                if first_tag not in seg_book_old_tags:
                    if not from_is_input:
                        from_vector = from_vector * -1.0
                    terminal_seg_book[first_tag] = {'left_vector': from_vector, 'left_clamp': True}

                last_tag = path_contain_seg[-1]
                if last_tag not in seg_book_old_tags:
                    if to_is_input:
                        to_vector = to_vector * -1.0
                    terminal_seg_book[last_tag] = {'right_vector': to_vector, 'right_clamp': True}

                # ------ record path contain seg
                path_contain_seg_dict[path_name].append(path_contain_seg)

        # ------ Step 4 generate all segments & aggregation information
        aggr_df, tmp_seg_tag2idx = [], {}
        for seg_idx, (seg_tag, seg_flags) in enumerate(seg_book.items()):
            c_xyzr = np.array([tmp_node2xyzr[flag] for flag in seg_flags])
            assert assemble_cfg['seg_node_min'] >= 10
            c_xyzr = self.refit_xyzr(c_xyzr, assemble_cfg['seg_node_min'], assemble_cfg['seg_node_max'])
            df = pd.DataFrame(c_xyzr, columns=['x', 'y', 'z', 'radius'])
            df['seg_idx'] = seg_idx

            aggr_df.append(df)
            tmp_seg_tag2idx[seg_tag] = seg_idx

        aggr_df = pd.concat(aggr_df, axis=0, ignore_index=True)
        aggr_df = self.convert_to_flag_df(aggr_df)

        c_table = aggr_df.copy()
        c_table.drop_duplicates(subset=['unique_idx'], inplace=True, keep='last')
        c_table.sort_values(by=['unique_idx'], inplace=True)
        c_table['is_fix'] = 0
        self.c_table = c_table

        # ------ add segment cell
        for (seg_idx,), sub_df in aggr_df.groupby(by=['seg_idx']):
            self.seg_dict[seg_idx] = SegmentCell(seg_idx, self.group_idx, sub_df['unique_idx'].values, self)

        # ------ add connector cell, be careful not to repeat
        connector_dict = {}
        for tag0, tag1 in connector_book:
            seg_idx0, seg_idx1 = tmp_seg_tag2idx[tag0], tmp_seg_tag2idx[tag1]
            con_idx0, con_idx1 = min(seg_idx0, seg_idx1), max(seg_idx0, seg_idx1)
            con_flag = (con_idx0, con_idx1)
            if con_flag not in connector_dict.keys():
                connector_dict[con_flag] = (seg_idx0, seg_idx1)

        tmp_con_tag2idx = {}
        for connector_idx, (_, (seg_idx0, seg_idx1)) in enumerate(connector_dict.items()):
            self.connector_dict[connector_idx] = ConnectorCell(connector_idx, self.group_idx, seg_idx0, seg_idx1, self)
            tmp_con_tag2idx[(min(seg_idx0, seg_idx1), max(seg_idx0, seg_idx1))] = connector_idx

        # ------ add path contain segments information
        path_idx = 0
        for path_name, path_contain_seg_list in path_contain_seg_dict.items():
            for path_contain_seg in path_contain_seg_list:

                if len(path_contain_seg) > 1:
                    contain_idxs, contain_types = [], []
                    for i in range(len(path_contain_seg) - 1):
                        cell_0 = self.seg_dict[tmp_seg_tag2idx[path_contain_seg[i]]]
                        cell_1 = self.seg_dict[tmp_seg_tag2idx[path_contain_seg[i + 1]]]

                        if i == 0:
                            contain_idxs.append(cell_0.idx)
                            contain_types.append(cell_0.cell_type)

                        # ------ record connector
                        con = self.connector_dict[tmp_con_tag2idx[
                            (min(cell_0.idx, cell_1.idx), max(cell_0.idx, cell_1.idx))
                        ]]
                        contain_idxs.append(con.idx)
                        contain_types.append(con.cell_type)

                        # ------ record next cell
                        contain_idxs.append(cell_1.idx)
                        contain_types.append(cell_1.cell_type)

                    self.path_dict[path_idx] = PathCell(
                        path_name, path_idx, self.group_idx, contain_idxs, contain_types, self
                    )
                
                else:
                    cell = self.seg_dict[tmp_seg_tag2idx[path_contain_seg[0]]]
                    self.path_dict[path_idx] = PathCell(
                        path_name, path_idx, self.group_idx, [cell.idx], [cell.cell_type], self
                    )
                
                path_idx += 1

        # ------ add terminal information
        for tag, cfg in terminal_seg_book.items():
            seg_idx = tmp_seg_tag2idx[tag]
            if 'left_clamp' in cfg.keys():
                self.seg_dict[seg_idx].left_clamp = cfg['left_clamp']
                self.seg_dict[seg_idx].left_vector = cfg['left_vector']
            if 'right_clamp' in cfg.keys():
                self.seg_dict[seg_idx].right_clamp = cfg['right_clamp']
                self.seg_dict[seg_idx].right_vector = cfg['right_vector']

        # ------ update fix information
        for seg_idx, seg_cell in self.seg_dict.items():
            if seg_cell.left_clamp:
                self.c_table.loc[self.c_table.index[seg_cell.c_idxs[0]], 'is_fix'] = 1
            if seg_cell.right_clamp:
                self.c_table.loc[self.c_table.index[seg_cell.c_idxs[-1]], 'is_fix'] = 1

    def init_cfg(self, setting_cfg: Dict):
        for seg_cfg in setting_cfg['segments']:
            seg_idx = seg_cfg['seg_idx']
            self.seg_dict[seg_idx].init_cfg(seg_cfg)

        for con_cfg in setting_cfg['connectors']:
            con_idx = con_cfg['con_idx']
            self.connector_dict[con_idx].init_cfg(cfg=con_cfg, seg_dict=self.seg_dict)

        for path_cfg in setting_cfg['paths']:
            path_idx = path_cfg['path_idx']
            self.path_dict[path_idx].init_cfg(path_cfg)

    def init_cross_cell_cfg(self, setting_cfg: Dict):
        for pair_cfg in setting_cfg['pair_cells']:
            pair_idx = pair_cfg['pair_idx']
            self.pair_seg_dict[pair_idx] = PairCell(
                pair_idx, self.group_idx,
                left_idx=pair_cfg['left_idx'], left_type=pair_cfg['left_type'],
                right_idx=pair_cfg['right_idx'], right_type=pair_cfg['right_type'],
                parent_cell=self
            )
            self.pair_seg_dict[pair_idx].init_cfg(pair_cfg)

    def prepare_tensor(self):
        self.table_xyz_t = TorchUtils.np2tensor(self.c_table[['x', 'y', 'z']].values[:, :3], require_grad=True)
        for _, seg_cell in self.seg_dict.items():
            seg_cell.prepare_tensor()
            seg_cell.update_tensor(self.table_xyz_t)

        for _, con_seg in self.connector_dict.items():
            con_seg.prepare_tensor()
            con_seg.update_tensor(self.table_xyz_t)

        for _, path_cell in self.path_dict.items():
            path_cell.update_tensor(with_analytical_tangent=False)

    def update_state(self, with_tensor: bool, with_np: bool):
        if with_tensor:
            for _, seg_cell in self.seg_dict.items():
                seg_cell.update_tensor(self.table_xyz_t)
            for _, con_cell in self.connector_dict.items():
                con_cell.update_tensor(self.table_xyz_t)
            for _, path_cell in self.path_dict.items():
                path_cell.update_tensor(with_analytical_tangent=False)
        if with_np:
            self.c_table.loc[self.c_table.index, ['x', 'y', 'z']] = TorchUtils.tensor2np(self.table_xyz_t)

    def compute_loss(self, unit_length: float, with_default_weight: bool):
        cost_dict = {}
        for seg_idx, seg_cell in self.seg_dict.items():
            cost_dict.update(seg_cell.compute_cell_cost(self.table_xyz_t, with_default_weight))
        for con_idx, con_cell in self.connector_dict.items():
            cost_dict.update(con_cell.compute_cell_cost(self.table_xyz_t, with_default_weight))
        for path_idx, path_cell in self.path_dict.items():
            cost_dict.update(path_cell.compute_set_cost(unit_length, with_default_weight))
        # for path_idx, path_cell in self.path_dict.items():
        #     cost_dict.update(path_cell.compute_element_cost(with_default_weight))
        for pair_idx, pair_cell in self.pair_seg_dict.items():
            cost_dict.update(pair_cell.compute_cost(unit_length, with_default_weight))
        return cost_dict

    @property
    def is_fix(self) -> List[bool]:
        return (self.c_table[['is_fix']].values == 1).reshape(-1)

    def save(self, file: str = None, writer: h5py.File = None):
        close_auto = False
        if writer is None:
            assert file.endswith('.hdf5')
            writer = h5py.File(file, mode='w')
            close_auto = True

        group_layer = writer.create_group(f"group_{self.group_idx}")
        group_layer.create_dataset("group_idx", data=[self.group_idx])

        table_xyzr = self.c_table[['x', 'y', 'z', 'radius']].values
        group_layer.create_dataset("c_xyzr", data=table_xyzr)
        group_layer.create_dataset("c_info", data=self.c_table[['unique_idx', 'is_unique', 'is_fix']].values)

        seg_group_layer = group_layer.create_group('segment')
        for seg_idx, seg_cell in self.seg_dict.items():
            seg_layer = seg_group_layer.create_group(f"seg_{seg_cell.idx}")
            seg_layer.create_dataset("idx", data=[seg_cell.idx])
            seg_layer.create_dataset("cell_type", data=[seg_cell.cell_type])
            seg_layer.create_dataset("ts", data=seg_cell.ts)
            seg_layer.create_dataset("degree", data=[seg_cell.degree])
            seg_layer.create_dataset("fill_last_one", data=[seg_cell.fill_last_one])
            seg_layer.create_dataset("knot", data=seg_cell.knot)
            seg_layer.create_dataset("c_idxs", data=seg_cell.c_idxs)
            seg_layer.create_dataset("c_radius", data=seg_cell.c_radius)
            seg_layer.create_dataset("spline", data=seg_cell.compute_spline_np(table_xyzr))
            seg_layer.create_dataset("left_clamp", data=[seg_cell.left_clamp])
            seg_layer.create_dataset("right_clamp", data=[seg_cell.right_clamp])
            seg_layer.create_dataset("left_vector", data=seg_cell.left_vector)
            seg_layer.create_dataset("right_vector", data=seg_cell.right_vector)
            seg_layer.create_dataset("tangent_vector", data=seg_cell.compute_tangent_np(table_xyzr))

        con_group_layer = group_layer.create_group('connector')
        for con_idx, con_cell in self.connector_dict.items():
            con_layer = con_group_layer.create_group(f"con_{con_cell.idx}")
            con_layer.create_dataset("idx", data=[con_cell.idx])
            con_layer.create_dataset("cell_type", data=[con_cell.cell_type])
            con_layer.create_dataset("ts", data=con_cell.ts)
            con_layer.create_dataset("degree", data=[con_cell.degree])
            con_layer.create_dataset("fill_last_one", data=[con_cell.fill_last_one])
            con_layer.create_dataset("knot", data=con_cell.knot)
            con_layer.create_dataset("c_idxs", data=con_cell.c_idxs)
            con_layer.create_dataset("c_radius", data=con_cell.c_radius)
            con_layer.create_dataset("spline", data=con_cell.compute_spline_np(table_xyzr))
            con_layer.create_dataset("left_clamp", data=[con_cell.left_clamp])
            con_layer.create_dataset("right_clamp", data=[con_cell.right_clamp])
            con_layer.create_dataset("tangent_vector", data=con_cell.compute_tangent_np(table_xyzr))

        path_group_layer = group_layer.create_group('path')
        for path_idx, path_cell in self.path_dict.items():
            path_layer = path_group_layer.create_group(f"path_{path_cell.idx}")
            path_layer.create_dataset("idx", data=[path_cell.idx])
            path_layer.create_dataset("path_name", data=[path_cell.path_name])
            path_layer.create_dataset("cell_idxs", data=path_cell.cell_idxs)
            path_layer.create_dataset("cell_types", data=path_cell.cell_types)

        if close_auto:
            writer.close()

    def update_fix_term(self, c_idxs: List[int], is_fix_list: List[int]):
        for i in range(len(c_idxs)):
            self.c_table.loc[self.c_table.index[c_idxs[i]], 'is_fix'] = is_fix_list[i] > 0

    def update_xyzr_term(self, c_idxs: List[int], xyzr_list: List[int]):
        for i in range(len(c_idxs)):
            self.c_table.loc[self.c_table.index[c_idxs[i]], ['x', 'y', 'z', 'radius']] = xyzr_list[i]

    def draw_control_points(
            self, vis: VisUtils = None, with_seg=False, with_connector=False, with_fix=False, with_overlap=False
    ):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        table_xyzr = self.c_table[['x', 'y', 'z', 'radius']].values
        if with_seg:
            for seg_idx, seg_cell in self.seg_dict.items():
                seg_xyzr = table_xyzr[seg_cell.c_idxs, :]
                color = np.random.uniform(0.0, 1.0, size=(3,))
                if with_connector:
                    color = (0.0, 0.0, 1.0)

                line_set = VisUtils.create_line_set(np.arange(0, seg_xyzr.shape[0], 1))
                mesh = VisUtils.create_line(seg_xyzr[:, :3], line_set)
                vis.plot(mesh, color=color, point_size=5)

                if seg_cell.left_clamp:
                    vis.plotter.add_arrows(
                        seg_xyzr[:1, :3] - seg_cell.left_vector, np.array([seg_cell.left_vector]), mag=1.0
                    )
                if seg_cell.right_clamp:
                    vis.plotter.add_arrows(seg_xyzr[-1:, :3], np.array([seg_cell.right_vector]), mag=1.0)

        if with_connector:
            for con_idx, con_cell in self.connector_dict.items():
                con_xyzr = table_xyzr[con_cell.c_idxs, :]
                line_set = VisUtils.create_line_set(np.arange(0, con_xyzr.shape[0], 1))
                mesh = VisUtils.create_line(con_xyzr[:, :3], line_set)
                vis.plot(mesh, color=(0.0, 1.0, 0.0), point_size=5)

        for data_point in self.c_table.itertuples():
            if with_fix:
                if data_point.is_fix == 1:
                    mesh = VisUtils.create_sphere(np.array([data_point.x, data_point.y, data_point.z]), radius=0.25)
                    vis.plot(mesh, color=(1.0, 0.0, 0.0))

            if with_overlap:
                if data_point.is_unique == 0:
                    mesh = VisUtils.create_sphere(np.array([data_point.x, data_point.y, data_point.z]), radius=0.25)
                    vis.plot(mesh, color=(0.0, 1.0, 0.0))

        if show_plot:
            vis.show()

    def draw_spline(self, vis: VisUtils = None, is_tube=False):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        table_xyzr = self.c_table[['x', 'y', 'z', 'radius']].values
        for seg_idx, seg_cell in self.seg_dict.items():
            seg_xyzr = seg_cell.compute_spline_np(table_xyzr)
            line_set = VisUtils.create_line_set(np.arange(0, seg_xyzr.shape[0], 1))
            if is_tube:
                tube_mesh = VisUtils.create_deformation_tube(
                    seg_xyzr[:, :3], line_set, radius=None, scalars=seg_xyzr[:, -1]
                )
                vis.plot(tube_mesh, color=(0., 0., 1.), opacity=1.0)
            else:
                mesh = VisUtils.create_line(seg_xyzr[:, :3], line_set)
                # vis.plot(mesh, color=(0., 0., 1.), style='wireframe', point_size=0.1, line_width=2.0)
                vis.plot(mesh, color=(0., 0., 1.), point_size=5)

        for con_idx, con_cell in self.connector_dict.items():
            con_xyzr = con_cell.compute_spline_np(table_xyzr)
            line_set = VisUtils.create_line_set(np.arange(0, con_xyzr.shape[0], 1))
            if is_tube:
                tube_mesh = VisUtils.create_deformation_tube(
                    con_xyzr[:, :3], line_set, radius=None, scalars=con_xyzr[:, -1]
                )
                vis.plot(tube_mesh, color=(0., 1., 0.), opacity=1.0)
            else:
                mesh = VisUtils.create_line(con_xyzr[:, :3], line_set)
                # vis.plot(mesh, color=(0., 1., 0.), style='wireframe', point_size=0.1, line_width=2.0)
                vis.plot(mesh, color=(0., 1., 0.), point_size=5)

        if show_plot:
            vis.show()

    def draw_cell_tensor(self, vis: VisUtils = None):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        for seg_idx, seg_cell in self.seg_dict.items():
            seg_xyzr = TorchUtils.tensor2np(seg_cell.spline_xyz_t)
            line_set = VisUtils.create_line_set(np.arange(0, seg_xyzr.shape[0], 1))
            mesh = VisUtils.create_line(seg_xyzr[:, :3], line_set)
            vis.plot(mesh, color=(0., 0., 1.), style='wireframe', point_size=0.0, line_width=1.0)

        for con_idx, con_cell in self.connector_dict.items():
            color = (0., 1., 0.)
            con_xyzr = TorchUtils.tensor2np(con_cell.spline_xyz_t)
            line_set = VisUtils.create_line_set(np.arange(0, con_xyzr.shape[0], 1))
            mesh = VisUtils.create_line(con_xyzr[:, :3], line_set)
            vis.plot(mesh, color=color, style='wireframe', point_size=0.0, line_width=1.0)

        if show_plot:
            vis.show()

    def draw_curvature(self, vis: VisUtils = None):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        cell_list: List[Union[SegmentCell, ConnectorCell]] = []
        for seg_idx, seg_cell in self.seg_dict.items():
            cell_list.append(seg_cell)
        for con_idx, con_cell in self.connector_dict.items():
            cell_list.append(con_cell)

        table_xyzr = self.c_table[['x', 'y', 'z', 'radius']].values
        for cell in cell_list:
            xyzr = cell.compute_spline_np(table_xyzr)
            curvature = cell.compute_curvature_np(table_xyzr)
            mesh = pyvista.PointSet(xyzr[:, :3])
            mesh['scalars'] = curvature
            vis.plot(mesh, point_size=6, scalars='scalars', show_scalar_bar=True, cmap=plt.cm.rainbow)

        if show_plot:
            vis.show()

    def draw_path_info(
            self, vis: VisUtils = None, with_cross_section=True, with_arrow=False, with_conflict=False
    ):
        show_plot = False
        if vis is None:
            show_plot = True

        for path_idx, path_cell in self.path_dict.items():
            path_info_pack = path_cell.compute_path_np_info()
            spline_pcd: np.ndarray = path_info_pack['spline_xyz']
            spline_tangent: np.ndarray = path_info_pack['spline_tangent']
            spline_radius: np.ndarray = path_info_pack['radius']

            if show_plot:
                vis = VisUtils()

            pcd_mesh = VisUtils.create_point_cloud(spline_pcd)
            vis.plot(pcd_mesh, point_size=5, color=(0., 1., 0.))

            if with_cross_section:
                for xyz, vec, r in zip(spline_pcd, spline_tangent, spline_radius):
                    mesh = VisUtils.create_polygon(xyz, r, vec, n_sides=60)
                    vis.plot(mesh, color=(0., 0., 1.), opacity=1.0, style='wireframe')

            if with_arrow:
                vis.plotter.add_arrows(spline_pcd, spline_tangent, mag=0.05, show_scalar_bar=False)

            if with_conflict:
                radius_conflict_pair = AuxCostUtil.detect_path_self_conflict(spline_pcd, spline_radius)
                cur_conflict_pair, res_pack = AuxCostUtil.detect_adjacent_curvature_conflict(
                    spline_pcd, spline_tangent, spline_radius,
                    is_project_mid_pcd=True, project_orig_pcd=False, radius_scale=1.0
                )

                if radius_conflict_pair.shape[0] > 0:
                    radius_conflict_line_set = np.concatenate([
                        np.full((radius_conflict_pair.shape[0], 1), fill_value=2), radius_conflict_pair
                    ], axis=1)
                    radius_conflict_mesh = VisUtils.create_line(spline_pcd, radius_conflict_line_set)
                    vis.plot(radius_conflict_mesh, color=(1., 1., 0.), opacity=1.0, line_width=4)

                if cur_conflict_pair.shape[0] > 0:
                    cur_conflict_line_set = np.concatenate([
                        np.full((cur_conflict_pair.shape[0], 1), fill_value=2), cur_conflict_pair
                    ], axis=1)
                    cur_conflict_mesh = VisUtils.create_line(spline_pcd, cur_conflict_line_set)
                    vis.plot(cur_conflict_mesh, color=(1., 0., 0.), opacity=1.0, line_width=4)
                    vis.plot(VisUtils.create_point_cloud(res_pack['pcd_inter']), point_size=7, color=(0., 1., 1.))

            if show_plot:
                vis.show()

    def export_cfg(self):
        # self.c_table is pd.Dataframe, please convert it to dict whose key is column
        cfg = {
            'group_idx': self.group_idx,
            'c_table': self.c_table.to_dict(orient='list'),
        }
        return cfg

    @staticmethod
    def restore_cfg(cfg: Dict):
        group_cell = GroupCell(cfg['group_idx'])
        group_cell.c_table = pd.DataFrame(cfg['c_table'])
        return group_cell