# todo
#  1.添加动量项
#  2.增加不同分段之间的样式（分段之间的连接约束改为特定定义，不采用进出口定义全段）
#  3.增加半径修改再优化功能
#  4.增加半径曲率限制的自适应导出功能
#  5.添加CAD构造功能
#  6.不同分段之间更改为Bspline，使连接更流畅
#  7.添加优化中断与回续功能
#  8.不同分段之间添加半径平滑自适应
#  9.添加人工修改控制点位置再优化功能
#  10.提供自由固定控制点再优化功能
#  11.提供非固定半径优化功能
#  12.添加仿真功能
#  13.添加半径与控制点位置手动调整功能
#  14.曲率Cost添加sigmoid项
#  15.缺乏半径膨胀配置
#  16.拓展回退功能
#  17.添加自动优化功能
#  info.1 还是得重写，直觉上来说只要实际空间包含最优轨迹空间，那不管这个实际空间是什么样子，它都可以达到最优
#  info.2 connector由于共用控制点,可能自由度不足够
#  info.3 使用curvature分布优化 info.4 学习率使用近似 CosineAnnealingLR 的多个复合

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
import h5py

from scripts_py.version_9.mapf_pkg.bspline_utils import BsplineUtils
from scripts_py.version_9.mapf_pkg.torch_utils import TorchUtils, CostUtils, TorchMath
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils


class SegmentCell(object):
    def __init__(self, idx: int, group_idx: int, c_idxs: np.ndarray, parent_cell):
        self.idx = idx
        self.group_idx = group_idx
        self.c_idxs = c_idxs
        self.parent_cell: GroupPath = parent_cell
        self.cell_type: Literal["segment", "connector"] = 'segment'

        # ------ connector cfg
        self.left_clamp = False
        self.right_clamp = False
        self.left_vector: np.ndarray = np.array([0., 0., 0.])  # todo 这里仍然需要进出区分吗
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

        self.with_analytical_curvature = False

    def __str__(self):
        return f"segment_{self.idx}"

    def __len__(self):
        return self.c_idxs.shape[0]

    def init_cfg(self, cfg: Dict, parent_cell=None):
        self.degree = cfg['bspline_degree']
        self.density = cfg['bspline_density']
        self.with_analytical_curvature = cfg['with_analytical_curvature_computed']

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

            if self.with_analytical_curvature:
                mat_order1 = BsplineUtils.compute_first_order_mat(ts, n_controls, self.degree, knot)
                mat_order2 = BsplineUtils.compute_second_order_mat(ts, n_controls, self.degree, knot)
                self.mat_order1, self.mat_order2 = mat_order1, mat_order2

            print(f"[DEBUG]: Group[{self.group_idx}]/{self.cell_type}_{self.idx} "
                  f"length:{length:.2f} c_num:{n_controls} t_num:{ts.shape[0]} update")

            return True
        return False

    def prepare_tensor(self):
        self.spline_mat_t = TorchUtils.np2tensor(self.spline_mat, require_grad=False)
        if self.with_analytical_curvature:
            self.mat_order1_t = TorchUtils.np2tensor(self.mat_order1, require_grad=False)
            self.mat_order2_t = TorchUtils.np2tensor(self.mat_order2, require_grad=False)

    def update_tensor(self, table_xyz: torch.Tensor):
        c_xyz = table_xyz[self.c_idxs, :3]

        # ------ update spline position
        self.spline_xyz_t = self.spline_mat_t.matmul(c_xyz)

        if self.with_analytical_curvature:
            # ------ update spline curvature
            pcd_order1 = self.mat_order1_t.matmul(c_xyz)
            pcd_order2 = self.mat_order2_t.matmul(c_xyz)
            norm_vec = torch.cross(pcd_order1, pcd_order2, dim=1)
            a = torch.norm(norm_vec, p=2, dim=1)
            b = torch.pow(torch.norm(pcd_order1, p=2, dim=1), 3.0)

            if self.left_clamp and self.right_clamp:
                self.spline_curvature_t = (a / (b + 1e-8))[1:-1]
            elif self.left_clamp and (not self.right_clamp):
                self.spline_curvature_t = (a / (b + 1e-8))[1:]
            elif (not self.left_clamp) and self.right_clamp:
                self.spline_curvature_t = (a / (b + 1e-8))[:-1]
            else:
                self.spline_curvature_t = (a / (b + 1e-8))

    def compute_spline_np(self, table_xyzr: np.ndarray):
        c_xyz = table_xyzr[self.c_idxs, :4]
        return self.spline_mat.dot(c_xyz)

    def compute_curvature_np(self, table_xyzr: np.ndarray):
        return BsplineUtils.compute_curvature(self.ts, table_xyzr[self.c_idxs, :3], self.degree, self.knot)

    def compute_tensor_cost(self, table_xyz: torch.Tensor):
        cost_dict = {}
        for cfg in self.cost_cfgs:
            cost_tag = f"{self.group_idx}/{self.cell_type}_{self.idx}/{cfg['cost_type']}"

            if cfg['cost_type'] == 'length':
                if cfg['data_type'] == 'spline':
                    cost_dict[cost_tag] = CostUtils.compute_length_cost(self.spline_xyz_t, cfg)
                elif cfg['data_type'] == 'control_points':
                    cost_dict[cost_tag] = CostUtils.compute_length_cost(table_xyz[self.c_idxs, :3], cfg)
                else:
                    raise ValueError(f"[ERROR]: Non-valid data type {cfg['cost_type']}")

            elif cfg['cost_type'] == 'curvature':
                if cfg['curvature_method'] == 'approximate_curvature':
                    cfg['value'] = 1.0 / TorchUtils.np2tensor(self.radius[1:-1] * cfg['curvature_scale'], False)
                    cost_dict[cost_tag] = CostUtils.compute_curvature_cost(self.spline_xyz_t, cfg)

                elif cfg['curvature_method'] == 'analytical_curvature':
                    if self.left_clamp and self.right_clamp:
                        r_radius = self.radius[1:-1]
                    elif self.left_clamp and (not self.right_clamp):
                        r_radius = self.radius[1:]
                    elif (not self.left_clamp) and self.right_clamp:
                        r_radius = self.radius[:-1]
                    else:
                        r_radius = self.radius
                    cfg['value'] = 1.0 / TorchUtils.np2tensor(r_radius * cfg['curvature_scale'], False)
                    cost_dict[cost_tag] = CostUtils.compute_curvature_cost(self.spline_curvature_t, cfg)

            elif cfg['cost_type'] == 'length_distribution':
                if cfg['data_type'] == 'spline':
                    cost_dict[cost_tag] = CostUtils.compute_distribution_cost(self.spline_xyz_t, cfg)
                elif cfg['data_type'] == 'control_points':
                    cost_dict[cost_tag] = CostUtils.compute_distribution_cost(table_xyz[self.c_idxs, :3], cfg)
                else:
                    raise ValueError(f"[ERROR]: Non-valid data type {cfg['cost_type']}")

            elif cfg['cost_type'] == 'curvature_distribution':
                if cfg['curvature_method'] == 'approximate_curvature':
                    curvature = TorchMath.compute_approximate_curvature(self.spline_xyz_t)
                    cost_dict[cost_tag] = CostUtils.compute_distribution_cost(curvature, cfg)

                elif cfg['curvature_method'] == 'analytical_curvature':
                    cost_dict[cost_tag] = CostUtils.compute_distribution_cost(self.spline_curvature_t, cfg)

            elif cfg['cost_type'] == 'clamp_vector':
                if self.left_clamp:
                    idxs_range = max(math.floor(cfg['vector_length'] / self.density), 2)
                    cost_dict[cost_tag] = CostUtils.compute_vector_cost(
                        self.spline_xyz_t[:idxs_range], cfg, 'left2any'
                    )

                if self.right_clamp:
                    idxs_range = max(math.floor(cfg['vector_length'] / self.density), 2)
                    cost_dict[cost_tag] = CostUtils.compute_vector_cost(
                        self.spline_xyz_t[-idxs_range:], cfg, 'any2right'
                    )

            else:
                raise ValueError(f"[ERROR]: Non-valid seg cost type {cfg['cost_type']}")

        return cost_dict


class ConnectorCell(SegmentCell):
    def __init__(self, idx: int, group_idx: int, left_seg_idx: int, right_seg_idx: int, parent_cell):
        self.idx = idx
        self.group_idx = group_idx
        self.left_seg_idx = left_seg_idx
        self.right_seg_idx = right_seg_idx
        self.left_seg: SegmentCell = None
        self.right_seg: SegmentCell = None
        self.c_idxs: np.ndarray = None
        self.parent_cell: GroupPath = parent_cell
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

    def init_cfg(self, cfg: Dict, seg_dict: Dict[int, SegmentCell]):
        # todo 正反向还没做测试

        self.density = cfg['bspline_density']
        self.with_analytical_curvature = cfg['with_analytical_curvature_computed']
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
        c_radius = np.linspace(orig_radius[0], orig_radius[1], num=self.c_idxs.shape[0])
        return super().update_spline_NpCfg(table_xyzr, update_threshold, c_radius)


class GroupPath(object):
    def __init__(self, group_idx: int):
        self.group_idx = group_idx
        self.c_table: pd.DataFrame = None
        self.seg_dict: Dict[int, SegmentCell] = {}
        self.connector_dict: Dict[int, ConnectorCell] = {}
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
        seg_book, connector_book, terminal_seg_book = {}, [], {}
        for path_info in aid_path_info:
            from_pos = path_info['from_position']
            from_flag = terminal_xyz2flags[(from_pos[0], from_pos[1], from_pos[2])]
            from_vector = path_info['from_vector']
            from_is_input = path_info['from_is_input']

            to_pos = path_info['to_position']
            to_flag = terminal_xyz2flags[(to_pos[0], to_pos[1], to_pos[2])]
            to_vector = path_info['to_vector']
            to_is_input = path_info['to_is_input']

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

        for connector_idx, (_, (seg_idx0, seg_idx1)) in enumerate(connector_dict.items()):
            self.connector_dict[connector_idx] = ConnectorCell(connector_idx, self.group_idx, seg_idx0, seg_idx1, self)

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

    def prepare_tensor(self):
        self.table_xyz_t = TorchUtils.np2tensor(self.c_table[['x', 'y', 'z']].values[:, :3], require_grad=True)
        for _, seg_cell in self.seg_dict.items():
            seg_cell.prepare_tensor()
            seg_cell.update_tensor(self.table_xyz_t)

        for _, con_seg in self.connector_dict.items():
            con_seg.prepare_tensor()
            con_seg.update_tensor(self.table_xyz_t)

    def update_state(self, with_tensor: bool, with_np: bool):
        if with_tensor:
            for _, seg_cell in self.seg_dict.items():
                seg_cell.update_tensor(self.table_xyz_t)
            for _, con_seg in self.connector_dict.items():
                con_seg.update_tensor(self.table_xyz_t)
        if with_np:
            self.c_table.loc[self.c_table.index, ['x', 'y', 'z']] = TorchUtils.tensor2np(self.table_xyz_t)

    def compute_loss(self):
        cost_dict = {}
        for seg_idx, seg_cell in self.seg_dict.items():
            cost_dict.update(seg_cell.compute_tensor_cost(self.table_xyz_t))
        for con_idx, con_cell in self.connector_dict.items():
            cost_dict.update(con_cell.compute_tensor_cost(self.table_xyz_t))
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

        data_layer = group_layer.create_group('data')
        for seg_idx, seg_cell in self.seg_dict.items():
            seg_layer = data_layer.create_group(f"seg_{seg_cell.idx}")
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

        for con_idx, con_cell in self.connector_dict.items():
            con_layer = data_layer.create_group(f"con_{con_cell.idx}")
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

        if close_auto:
            writer.close()

    def restore(self):
        # todo non-complete
        raise NotImplementedError

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

    def draw_tensor(self, vis: VisUtils = None):
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
