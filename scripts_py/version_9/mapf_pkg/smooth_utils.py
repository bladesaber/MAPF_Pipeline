import math

import h5py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Union

import pyvista
import torch
import torch.nn.functional as F

from scripts_py.version_9.mapf_pkg.torch_utils import TorchUtils, MeanTestLearner
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils
from build import mapf_pipeline


# todo:
#  1.离散方法（数值化）:点表述方法
#  2.半连续方法 :类似Bspline方法，理论上可以通过无穷插值获得无穷精度
#  3.连续方法（参数化）:通常方法是构建表述函数族，并通过拟合求解，但这种表述方法只适合单路径。
#  难点:
#  1.曲率限制
#  2.无静态锚定点
#  3.多组路径考虑距离干涉
#  4.路径考虑障碍点干涉
#  5.路径有多分支，且分支不均衡，多分支用spline的前后端点重叠实现


class BsplineUtils(object):
    @staticmethod
    def create_knots(
            pcd: np.ndarray, degree: int, range_min=0.0, range_max=1.0, clamp=True,
            uniform_weight: float = -1.0, uniform_step: int = np.inf
    ):
        """
        pcd: control point cloud
        """
        knot = np.linspace(range_min, range_max, pcd.shape[0] - degree + 1)
        if uniform_weight > 0:
            assert 0.0 < uniform_weight < 1.0

            num = knot.shape[0] - 2
            step = min(math.floor(num * 0.5), uniform_step)

            for i in range(step):
                begin_loc = i + 1
                knot[begin_loc] += (knot[begin_loc + 1] - knot[begin_loc]) * uniform_weight

                end_loc = num - i
                knot[end_loc] -= (knot[end_loc] - knot[end_loc - 1]) * uniform_weight

        if clamp:
            knot = np.concatenate([
                np.full(shape=(degree,), fill_value=range_min), knot, np.full(shape=(degree,), fill_value=range_max)
            ])
        return knot

    @staticmethod
    def compute_Bip(t, i: int, degree: int, knot: np.ndarray):
        if degree == 0:
            if knot[i] <= t < knot[i + 1]:
                return 1.0
            else:
                return 0.0

        if knot[i + degree] == knot[i]:
            c1 = 0.0
        else:
            c1 = (t - knot[i]) / (knot[i + degree] - knot[i])
            c1 = c1 * BsplineUtils.compute_Bip(t, i, degree - 1, knot)

        if knot[i + degree + 1] == knot[i + 1]:
            c2 = 0.0
        else:
            c2 = (knot[i + degree + 1] - t) / (knot[i + degree + 1] - knot[i + 1])
            c2 = c2 * BsplineUtils.compute_Bip(t, i + 1, degree - 1, knot)

        return c1 + c2

    @staticmethod
    def compute_mat(t_list: np.ndarray, pcd: np.ndarray, degree: int, knot: np.ndarray):
        num = pcd.shape[0]
        b_mat = np.zeros(shape=(t_list.shape[0], num))
        for j, t in enumerate(t_list):
            for i in range(num):
                b_mat[j, i] = BsplineUtils.compute_Bip(t, i, degree, knot)
            if j == t_list.shape[0] - 1:
                b_mat[-1, -1] = 1.0
        return b_mat

    @staticmethod
    def compute_Bip_first_order(t, i: int, degree: int, knot: np.ndarray):
        if degree == 0:
            return 0.0

        if knot[i + degree] == knot[i]:
            c1 = 0.0
        else:
            c1 = degree / (knot[i + degree] - knot[i])
            c1 = c1 * BsplineUtils.compute_Bip(t, i, degree - 1, knot)

        if knot[i + degree + 1] == knot[i + 1]:
            c2 = 0.0
        else:
            c2 = -degree / (knot[i + degree + 1] - knot[i + 1])
            c2 = c2 * BsplineUtils.compute_Bip(t, i + 1, degree - 1, knot)
        return c1 + c2

    @staticmethod
    def compute_Bip_second_order(t, i: int, degree: int, knot: np.ndarray):
        if degree <= 1:
            return 0.0

        if knot[i + degree] == knot[i]:
            c1 = 0.0
        else:
            c1 = degree / (knot[i + degree] - knot[i])
            c1 = c1 * BsplineUtils.compute_Bip_first_order(t, i, degree - 1, knot)

        if knot[i + degree + 1] == knot[i + 1]:
            c2 = 0.0
        else:
            c2 = -degree / (knot[i + degree + 1] - knot[i + 1])
            c2 = c2 * BsplineUtils.compute_Bip_first_order(t, i + 1, degree - 1, knot)
        return c1 + c2

    @staticmethod
    def compute_relate_pcd_idxs(t: float, knot: np.ndarray, degree: int, pcd: np.ndarray):
        if t == 0:
            return 0
        elif t == 1.0:
            return pcd.shape[0]
        else:
            knot_unique = np.sort(np.unique(knot))
            i = int((t >= knot_unique).sum() - 1)
            return np.arange(i, i + degree + 1, 1)

    @staticmethod
    def compute_first_order_mat(t_list: np.ndarray, pcd: np.ndarray, degree: int, knot: np.ndarray):
        """
        Since I can't estimate the real distance of t_list, the curvature will be wrong estimate,
         value of curvature related to t_list
        """
        num = pcd.shape[0]
        b_mat = np.zeros(shape=(t_list.shape[0], num))
        for j, t in enumerate(t_list):
            for i in range(num):
                b_mat[j, i] = BsplineUtils.compute_Bip_first_order(t, i, degree, knot)
        return b_mat

    @staticmethod
    def compute_second_order_mat(t_list: np.ndarray, pcd: np.ndarray, degree: int, knot: np.ndarray):
        """
        Since I can't estimate the real distance of t_list, the curvature will be wrong estimate,
         value of curvature related to t_list
        """
        num = pcd.shape[0]
        b_mat = np.zeros(shape=(t_list.shape[0], num))
        for j, t in enumerate(t_list):
            for i in range(num):
                b_mat[j, i] = BsplineUtils.compute_Bip_second_order(t, i, degree, knot)
        return b_mat

    @staticmethod
    def compute_curvature(t_list: np.ndarray, pcd: np.ndarray, degree: int, knot: np.ndarray):
        mat1 = BsplineUtils.compute_first_order_mat(t_list, pcd, degree, knot)
        pcd_order1 = mat1.dot(pcd)
        mat2 = BsplineUtils.compute_second_order_mat(t_list, pcd, degree, knot)
        pcd_order2 = mat2.dot(pcd)
        a = np.linalg.norm(
            np.cross(pcd_order1, pcd_order2, axis=1).reshape((pcd_order1.shape[0], -1)),
            axis=1, ord=2
        )
        b = np.power(np.linalg.norm(pcd_order1, axis=1, ord=2), 3.0)
        return a / (b + 1e-8)

    @staticmethod
    def compute_menger_curvature(
            pcd: np.ndarray, start_vec: np.ndarray = None, end_vec: np.ndarray = None
    ):
        pcd_list = []
        if start_vec is not None:
            pcd_list.append((pcd[0] - start_vec).reshape((1, -1)))
        pcd_list.append(pcd)
        if end_vec is not None:
            pcd_list.append((pcd[-1] + end_vec).reshape((1, -1)))
        pcd = np.concatenate(pcd_list, axis=0)

        pcd_0 = pcd[:-2, :]  # x
        pcd_1 = pcd[1:-1, :]  # y
        pcd_2 = pcd[2:, :]  # z

        a_lengths = np.linalg.norm(pcd_1 - pcd_0, ord=2, axis=1)
        b_lengths = np.linalg.norm(pcd_2 - pcd_1, ord=2, axis=1)
        c_lengths = np.linalg.norm(pcd_0 - pcd_2, ord=2, axis=1)
        s = (a_lengths + b_lengths + c_lengths) * 0.5 + 1e-12
        A = np.sqrt(s * (s - a_lengths) * (s - b_lengths) * (s - c_lengths))
        radius = a_lengths * b_lengths * c_lengths * 0.25 / (A + 1e-16)
        return 1.0 / radius

    @staticmethod
    def compute_approximate_curvature(pcd: np.ndarray, start_vec: np.ndarray = None, end_vec: np.ndarray = None):
        pcd_list = []
        if start_vec is not None:
            pcd_list.append((pcd[0] - start_vec).reshape((1, -1)))
        pcd_list.append(pcd)
        if end_vec is not None:
            pcd_list.append((pcd[-1] + end_vec).reshape((1, -1)))
        pcd = np.concatenate(pcd_list, axis=0)

        vectors = pcd[1:, :] - pcd[:-1, :]
        lengths = np.linalg.norm(vectors, axis=1, ord=2)
        vector0, vector1 = vectors[:-1, :], vectors[1:, :]
        length0, length1 = lengths[:-1], lengths[1:]
        cos_theta = np.sum(vector0 * vector1, axis=1) / (length0 * length1)

        curvature = np.arccos(cos_theta) / np.minimum(length0, length1)  # real curvature
        # arc_theta = np.log((1.0 - cos_theta) * 11.0 + 1.0)
        # curvature = arc_theta / np.minimum(length0, length1)

        return curvature


class SegmentCell(object):
    """
    info:
        bspline_degree:       int
        bspline_num:   int
        costs: [
            {
                method:       spline_square_length_cost
                type:         ['value', 'auto']
                value:        float
                weight:       float
            },
            {
                method:       control_square_length_cost
                type:         ['value', 'auto']
                value:        float
                weight:       float
            }
        ]
    """

    def __init__(self, idx: int, flags: List[int], color: Tuple = (1.0, 0.0, 0.0)):
        self.idx = idx
        self.flags = flags
        self.tag = self.encode(self.flags)
        self.pcd_idxs: np.ndarray = None
        self.knot: np.ndarray = None
        self.ts: np.ndarray = None
        self.spline_mat: np.ndarray = None
        self.cost_info_list: List[Dict] = None

        self.spline_mat_tensor: torch.Tensor = None
        self.pcd_tensor: torch.Tensor = None
        self.radius_np: np.ndarray = None,
        self.color = color
        self.relax_conflict_ratio = 0.01
        self.min_relax_conflict_thickness = 0.01

        self.output_spline_mat: np.ndarray = None  # just used for more precision output

    @staticmethod
    def encode(flags: List[int]):
        if len(flags) < 4:
            label_flags = flags
        else:
            label_flags = [flags[0], flags[1], flags[-2], flags[-1]]
        return tuple(np.sort(np.array(label_flags)))

    def __str__(self):
        return f"segment_{self.idx}"

    def __len__(self):
        return len(self.flags)

    def update_spline_mat(self, xyzr: np.ndarray, degree: int, sample_num: int, uniform_weight=0.5, uniform_step=3):
        self.knot = BsplineUtils.create_knots(
            xyzr[self.pcd_idxs, :3], degree, uniform_step=uniform_step, uniform_weight=uniform_weight
        )
        self.ts = np.linspace(0.0, 1.0, sample_num)
        self.spline_mat = BsplineUtils.compute_mat(self.ts, xyzr[self.pcd_idxs, :3], degree, self.knot)
        self.radius_np = self.spline_mat.dot(xyzr[self.pcd_idxs, 3:4]).reshape(-1)

    def get_bspline_xyzr_np(self, xyzr: np.ndarray):
        return self.spline_mat.dot(xyzr[self.pcd_idxs, :])

    def prepare_tensor(self):
        self.spline_mat_tensor = TorchUtils.np2tensor(self.spline_mat, require_grad=False)

    def update_tensor(self, control_pcd: torch.Tensor):
        self.pcd_tensor = self.spline_mat_tensor.matmul(control_pcd[self.pcd_idxs, :3])

    def update_output_spline_mat(
            self, xyzr: np.ndarray, degree: int, sample_num: int, uniform_weight=0.5, uniform_step=3
    ):
        knot = BsplineUtils.create_knots(
            xyzr[self.pcd_idxs, :3], degree, uniform_step=uniform_step, uniform_weight=uniform_weight
        )
        ts = np.linspace(0.0, 1.0, sample_num)
        self.output_spline_mat = BsplineUtils.compute_mat(ts, xyzr[self.pcd_idxs, :3], degree, knot)

    def get_output_bspline_xyzr_np(self, xyzr: np.ndarray):
        return self.output_spline_mat.dot(xyzr[self.pcd_idxs, :])


class PathCell(object):
    """
    costs:[
            method:       curvature_cost:
            radius_scale: float
            weight:       float
        ]
    """

    def __init__(self, segment_list: List[SegmentCell], begin_vec: np.ndarray, end_vec: np.ndarray):
        self.segments = segment_list
        self.begin_vec = begin_vec
        self.end_vec = end_vec
        self.cost_info_list: List[Dict] = None

        self.pcd_tensor: torch.Tensor = None
        self.radius_np: np.ndarray = None

    def get_bspline_xyzr_np(self, control_xyzr: np.ndarray, with_terminate_vec=False):
        path = []
        for i, cell in enumerate(self.segments):
            xyzr = cell.get_bspline_xyzr_np(control_xyzr)
            if i > 0:
                xyzr = xyzr[1:, :]
            path.append(xyzr)
        path = np.concatenate(path, axis=0)

        if with_terminate_vec:
            begin_vec = np.zeros((4,))
            begin_vec[:3] = path[0, :3] - self.begin_vec
            begin_vec[3] = path[0, 3]

            end_vec = np.zeros((4,))
            end_vec[:3] = path[-1, :3] + self.end_vec
            end_vec[3] = path[-1, 3]

            path = np.concatenate([
                begin_vec.reshape((1, -1)),
                path,
                end_vec.reshape((1, -1)),
            ], axis=0)

        return path

    def get_output_bspline_xyzr_np(self, control_xyzr: np.ndarray, with_terminate_vec=False):
        path = []
        for i, cell in enumerate(self.segments):
            xyzr = cell.get_output_bspline_xyzr_np(control_xyzr)
            if i > 0:
                xyzr = xyzr[1:, :]
            path.append(xyzr)
        path = np.concatenate(path, axis=0)

        if with_terminate_vec:
            begin_vec = np.zeros((4,))
            begin_vec[:3] = path[0, :3] - self.begin_vec
            begin_vec[3] = path[0, 3]

            end_vec = np.zeros((4,))
            end_vec[:3] = path[-1, :3] + self.end_vec
            end_vec[3] = path[-1, 3]

            path = np.concatenate([
                begin_vec.reshape((1, -1)),
                path,
                end_vec.reshape((1, -1)),
            ], axis=0)

        return path

    def update_tensor(self):
        pcd_tensor, radius_np = [], []
        for i, segment_cell in enumerate(self.segments):
            if i == 0:
                begin_pos = TorchUtils.tensor2np(segment_cell.pcd_tensor[0, :])
                begin_vec_pos_tensor = TorchUtils.np2tensor(
                    (begin_pos - self.begin_vec).reshape((1, -1)), require_grad=False
                )
                pcd_tensor.append(begin_vec_pos_tensor)
                pcd_tensor.append(segment_cell.pcd_tensor)
                radius_np.append(segment_cell.radius_np)

            else:
                pcd_tensor.append(segment_cell.pcd_tensor[1:, :])
                radius_np.append(segment_cell.radius_np[1:])

        end_pos = TorchUtils.tensor2np(pcd_tensor[-1][-1, :])
        end_vec_pos_tensor = TorchUtils.np2tensor((end_pos + self.end_vec).reshape((1, -1)), require_grad=False)
        pcd_tensor.append(end_vec_pos_tensor)

        self.pcd_tensor = TorchUtils.concatenate(pcd_tensor, dim=0)
        self.radius_np = np.concatenate(radius_np, axis=-1)


class NetworkPath(object):
    def __init__(self, grid: mapf_pipeline.DiscreteGridEnv, path_dict: Dict[str, np.ndarray], group_idx: int):
        self.grid = grid
        self.group_idx = group_idx

        self.node2info: Dict[Union[int, str], Dict[str, Union[int, float, str]]] = {}
        self.segments: Dict[Union[Tuple, int], SegmentCell] = {}
        self.path_dict: Dict[str, List[PathCell]] = {}  # 多个工况路径

        self.graph = nx.Graph()
        self.control_xyzr_np: np.ndarray = None
        self.control_pcd_tensor: torch.Tensor = None
        self.fix_pcd_idxs: np.ndarray = None

        control_xyzr_np = []
        for task_name, xyzr_list in path_dict.items():
            last_flag = None
            for i, xyzr in enumerate(xyzr_list):
                flag = self.grid.xyz2flag(x=xyzr[0], y=xyzr[1], z=xyzr[2])
                if flag not in self.node2info.keys():
                    control_xyzr_np.append(xyzr)
                    self.node2info[flag] = {'pcd_idx': len(control_xyzr_np) - 1, 'fix': False, 'type': 'cell'}
                if i > 0:
                    self.graph.add_edge(last_flag, flag)
                last_flag = flag
        self.control_xyzr_np = np.array(control_xyzr_np)
        # print(f"[INFO]: insert path nodes:{self.control_xyzr_np.shape[0]}, network nodes:{len(self.graph.nodes)}")

    def add_path(
            self, name: str, begin_xyz: np.ndarray, end_xyz: np.ndarray, begin_vec: np.ndarray, end_vec: np.ndarray
    ):
        begin_flag = self.grid.xyz2flag(x=begin_xyz[0], y=begin_xyz[1], z=begin_xyz[2])
        self.node2info[begin_flag].update({'fix': True})

        end_flag = self.grid.xyz2flag(x=end_xyz[0], y=end_xyz[1], z=end_xyz[2])
        self.node2info[end_flag].update({'fix': True})

        path_flag_list = nx.all_simple_paths(self.graph, begin_flag, end_flag)
        # path_flag_list = [nx.shortest_path(self.graph, begin_flag, end_flag)]

        for path_flags in path_flag_list:
            segment_list, segment_flags = [], []
            path_length = len(path_flags)

            for i, flag in enumerate(path_flags):
                segment_flags.append(flag)
                if self.graph.degree(flag) <= 2 and (i < path_length - 1):
                    continue

                if i < path_length - 1:
                    self.node2info[flag].update({'type': 'connector'})

                segment_cell = SegmentCell(idx=len(self.segments), flags=segment_flags.copy())
                segment_cell = self.segments.setdefault(segment_cell.tag, segment_cell)

                segment_list.append(segment_cell)
                segment_flags = segment_flags[-1:]  # save as begin of segment

            self.path_dict.setdefault(name, []).append(PathCell(segment_list, begin_vec, end_vec))

    def refit_graph(self, degree: int):
        for key in list(self.segments.keys()):
            cell = self.segments.pop(key)
            self.segments[cell.idx] = cell

        new_node_id = int(np.max(list(self.node2info.keys()))) + 1
        for seg_idx, cell in self.segments.items():

            while len(cell) < degree:
                shift = 0
                new_flags = cell.flags.copy()
                for i in range(len(cell) - 1):
                    flag0, flag1 = cell.flags[i], cell.flags[i + 1]
                    self.graph.remove_edge(flag0, flag1)

                    info0, info1 = self.node2info[flag0], self.node2info[flag1]
                    xyzr0, xyzr1 = self.control_xyzr_np[info0['pcd_idx']], self.control_xyzr_np[info1['pcd_idx']]
                    new_xyzr = (xyzr0 + xyzr1) * 0.5
                    self.control_xyzr_np = np.concatenate([self.control_xyzr_np, new_xyzr.reshape((1, -1))], axis=0)
                    self.node2info[new_node_id] = {
                        'pcd_idx': self.control_xyzr_np.shape[0] - 1, 'fix': False, 'type': 'cell'
                    }

                    self.graph.add_edge(flag0, new_node_id)
                    self.graph.add_edge(new_node_id, flag1)
                    new_flags.insert(i + shift + 1, new_node_id)

                    shift += 1
                    new_node_id += 1

                cell.flags = new_flags

        for seg_idx, cell in self.segments.items():
            cell.pcd_idxs = np.array([self.node2info[node_idx]['pcd_idx'] for node_idx in cell.flags])

    def update_optimization_info(self, segment_infos: Dict, path_infos: Dict):
        for seg_idx, cell in self.segments.items():
            info = segment_infos[seg_idx]
            cell.update_spline_mat(self.control_xyzr_np, info['bspline_degree'], info['bspline_num'])
            cell.cost_info_list = info['costs']
            cell.color = info['color']
            cell.relax_conflict_ratio = info['relax_conflict_ratio']
            cell.min_relax_conflict_thickness = info['min_relax_conflict_thickness']
            cell.update_output_spline_mat(self.control_xyzr_np, info['bspline_degree'], info['output_bspline_num'])

        for name, path_list in self.path_dict.items():
            for i, path_cell in enumerate(path_list):
                path_cell.cost_info_list = path_infos[name][i]

        fix_idxs = []
        for node_idx, info in self.node2info.items():
            if info['fix']:
                fix_idxs.append(info['pcd_idx'])
        self.fix_pcd_idxs = np.array(fix_idxs)

    def prepare_tensor(self):
        self.control_pcd_tensor = TorchUtils.np2tensor(self.control_xyzr_np[:, :3], require_grad=True)

        for _, segment_cell in self.segments.items():
            segment_cell.prepare_tensor()
            segment_cell.update_tensor(self.control_pcd_tensor)

        for name, path_list in self.path_dict.items():
            for path_cell in path_list:
                path_cell.update_tensor()

    def update_state(self, with_tensor: bool, with_np: bool):
        if with_tensor:
            for _, segment_cell in self.segments.items():
                segment_cell.update_tensor(self.control_pcd_tensor)
            for name, path_list in self.path_dict.items():
                for path_cell in path_list:
                    path_cell.update_tensor()

        if with_np:
            self.control_xyzr_np[:, :3] = TorchUtils.tensor2np(self.control_pcd_tensor)

    def compute_segment_cost(self) -> Dict[str, torch.Tensor]:
        cost_dict = {}
        for _, cell in self.segments.items():
            for info in cell.cost_info_list:
                if info['method'] == 'spline_square_length_cost':
                    cost_dict.setdefault('spline_square_length_cost', 0.0)
                    seg_lengths = TorchUtils.compute_segment_length(cell.pcd_tensor)
                    length = seg_lengths.square().mean()

                    if info['type'] == 'auto':
                        cost_dict['spline_square_length_cost'] += length * info['weight']
                    elif info['type'] == 'value':
                        cost_dict['spline_square_length_cost'] += torch.pow(
                            length - info['target_length'], 2.0
                        ) * info['weight']

                if info['method'] == 'control_square_length_cost':
                    cost_dict.setdefault('control_square_length_cost', 0.0)
                    seg_lengths = TorchUtils.compute_segment_length(self.control_pcd_tensor[cell.pcd_idxs, :])
                    length = seg_lengths.square().mean()

                    if info['type'] == 'auto':
                        cost_dict['control_square_length_cost'] += length * info['weight']
                    elif info['type'] == 'value':
                        cost_dict['control_square_length_cost'] += torch.pow(
                            length - info['target_length'], 2.0
                        ) * info['weight']

        return cost_dict

    def compute_path_cost(self) -> Dict[str, torch.Tensor]:
        cost_dict = {}
        for name, path_list in self.path_dict.items():
            for path_cell in path_list:
                for info in path_cell.cost_info_list:

                    # 由于这是一个最小最大问题，无法直接参数优化，因此使用了离散曲率。但离散曲率会被优化模型攻击而失效，因此必须添加额外约束
                    if info['method'] == 'curvature_cost':
                        cost_dict.setdefault('curvature_cost', 0.0)
                        target_curvature = 1.0 / TorchUtils.np2tensor(
                            path_cell.radius_np * info['radius_scale'], False
                        )

                        curvature = TorchUtils.compute_approximate_curvature(path_cell.pcd_tensor)
                        cost_dict['curvature_cost'] += info['radius_weight'] * (
                                (F.relu(curvature - target_curvature) / target_curvature).max()
                                + (F.relu(curvature - target_curvature) / target_curvature).mean()
                        )

                        cost_dict.setdefault('cos_cost', 0.0)
                        cos_val = TorchUtils.compute_cos_val(path_cell.pcd_tensor)
                        cost_dict['cos_cost'] += info['cos_weight'] * (
                            torch.pow(F.relu(info['cos_threshold'] - cos_val), info['cos_exponent']).max()
                            + torch.pow(F.relu(info['cos_threshold'] - cos_val), info['cos_exponent']).mean()
                        )

                    if info['method'] == 'connector_control_cos_cost':
                        cost_dict.setdefault('connector_control_cos_cost', 0.0)

                        segment_length = len(path_cell.segments)
                        for i in range(segment_length - 1):
                            seg_cell0 = path_cell.segments[i]
                            vec0 = (
                                    self.control_pcd_tensor[seg_cell0.pcd_idxs[-1], :]
                                    - self.control_pcd_tensor[seg_cell0.pcd_idxs[-2], :]
                            )

                            seg_cell1 = path_cell.segments[i + 1]
                            vec1 = (
                                    self.control_pcd_tensor[seg_cell1.pcd_idxs[1], :]
                                    - self.control_pcd_tensor[seg_cell1.pcd_idxs[0], :]
                            )

                            cos_val = (vec0 * vec1).sum() / torch.norm(vec0, p=2) / torch.norm(vec1, p=2)
                            cost_dict['connector_control_cos_cost'] += (1.0 - cos_val) * info['weight']

        return cost_dict

    def update_control_pcd(self, loss: torch.Tensor, lr: float):
        loss.backward()
        with torch.no_grad():
            self.control_pcd_tensor.grad[self.fix_pcd_idxs, :] = 0.0
            self.control_pcd_tensor -= self.control_pcd_tensor.grad * lr
            self.control_pcd_tensor.grad.zero_()

    def draw_network(self):
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(self.graph, pos, edge_color='black', width=1)
        nx.draw_networkx_labels(self.graph, pos, font_size=12)
        plt.show()

    def draw_segment(self, with_control=True, with_spline=False, vis: VisUtils = None):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        for seg_idx, cell in self.segments.items():
            color = np.random.uniform(0.0, 1.0, size=(3,))
            if with_control:
                control_pcd = self.control_xyzr_np[cell.pcd_idxs, :3]
                line_set = VisUtils.create_line_set(np.arange(0, cell.pcd_idxs.shape[0], 1))
                mesh = VisUtils.create_line(control_pcd, line_set)
                vis.plot(mesh, color=color, point_size=4)

            if with_spline:
                xyzr = cell.get_bspline_xyzr_np(self.control_xyzr_np)
                line_set = VisUtils.create_line_set(np.arange(0, xyzr.shape[0], 1))
                mesh = VisUtils.create_line(xyzr[:, :3], line_set)
                vis.plot(
                    mesh, color=color, style='wireframe',
                    point_size=0.1, line_width=2.0
                )

        if show_plot:
            vis.show()

    def draw_path(self, is_tensor=False):
        for name, path_list in self.path_dict.items():
            for path_cell in path_list:
                if is_tensor:
                    pcd = TorchUtils.tensor2np(path_cell.pcd_tensor)
                else:
                    pcd = path_cell.get_bspline_xyzr_np(self.control_xyzr_np)[:, :3]
                line_set = VisUtils.create_line_set(np.arange(0, pcd.shape[0], 1))
                mesh = VisUtils.create_line(pcd, line_set)
                mesh.plot()


def main():
    grid_env = mapf_pipeline.DiscreteGridEnv(
        size_of_x=51, size_of_y=51, size_of_z=1,
        x_init=0.0, y_init=0.0, z_init=0.0,
        x_grid_length=1.0, y_grid_length=1.0, z_grid_length=1.0
    )

    segment_dict = {
        'segment1': np.array([
            [15.0, 15.0, 0.0, 5.0],
            [15.0, 16.0, 0.0, 5.0],
            [15.0, 17.0, 0.0, 5.0],
            [15.0, 18.0, 0.0, 5.0],
            [15.0, 19.0, 0.0, 5.0],
            [15.0, 20.0, 0.0, 5.0],
            [15.0, 21.0, 0.0, 5.0]
        ]),
        'segment2': np.array([
            [15.0, 21.0, 0.0, 5.0],
            [14.0, 21.0, 0.0, 5.0],
            [13.0, 21.0, 0.0, 5.0],
            [12.0, 21.0, 0.0, 5.0],
            [11.0, 21.0, 0.0, 5.0],
            [10.0, 21.0, 0.0, 5.0],
        ]),
        'segment3': np.array([
            [15.0, 21.0, 0.0, 5.0],
            [16.0, 21.0, 0.0, 5.0],
            [17.0, 21.0, 0.0, 5.0],
            [18.0, 21.0, 0.0, 5.0],
            [19.0, 21.0, 0.0, 5.0],
            [20.0, 21.0, 0.0, 5.0]
        ]),
    }
    path_list = [
        {
            'name': 'path1',
            'begin_xyz': np.array([15.0, 15.0, 0.0]), 'end_xyz': np.array([10.0, 21.0, 0.0]),
            'begin_vec': np.array([0., 1., 0.]), 'end_vec': np.array([-1., 0., 0.])
        },
        {
            'name': 'path2',
            'begin_xyz': np.array([15.0, 15.0, 0.0]), 'end_xyz': np.array([20.0, 21.0, 0.0]),
            'begin_vec': np.array([0., 1., 0.]), 'end_vec': np.array([1., 0., 0.])
        }
    ]
    opt_info = {
        'segments': {
            0: {
                'bspline_degree': 3,
                'bspline_num': 40,
                'costs': [
                    {
                        "method": "control_square_length_cost",
                        "type": "auto",
                        "target_length": 0.0,
                        "weight": 0.1
                    }
                ],
                "color": [1.0, 0.0, 0.0]
            },
            1: {
                'bspline_degree': 3,
                'bspline_num': 40,
                'costs': [
                    {
                        'method': 'control_square_length_cost',
                        'type': 'auto',
                        'weight': 0.1
                    }
                ],
                "color": [0.0, 1.0, 0.0]
            },
            2: {
                'bspline_degree': 3,
                'bspline_num': 40,
                'costs': [
                    {
                        'method': 'control_square_length_cost',
                        'type': 'auto',
                        'weight': 0.1
                    }
                ],
                "color": [0.0, 0.0, 1.0]
            },
        },
        'paths': {
            'path1': [
                [
                    {
                        "method": "curvature_cost",
                        "radius_scale": 3.0,
                        "radius_weight": 1.0,
                        "cos_threshold": 0.95,
                        "cos_exponent": 1.5,
                        "cos_weight": 1.0
                    },
                    # {
                    #     "method": "connector_control_cos_cost",
                    #     "weight": 10.0
                    # }
                ]
            ],
            'path2': [
                [
                    {
                        "method": "curvature_cost",
                        "radius_scale": 3.0,
                        "radius_weight": 1.0,
                        "cos_threshold": 0.95,
                        "cos_exponent": 1.5,
                        "cos_weight": 1.0
                    },
                    # {
                    #     "method": "connector_control_cos_cost",
                    #     "weight": 10.0
                    # }
                ]
            ]
        }
    }

    net = NetworkPath(grid_env, segment_dict, group_idx=0)
    for path in path_list:
        net.add_path(
            path['name'], begin_xyz=path['begin_xyz'], end_xyz=path['end_xyz'],
            begin_vec=path['begin_vec'], end_vec=path['end_vec']
        )

    net.refit_graph(degree=4)
    net.update_optimization_info(opt_info['segments'], opt_info['paths'])
    net.prepare_tensor()

    # net.draw_network()
    # net.draw_segment(with_spline=False, with_control=True)
    # net.draw_path(is_tensor=False)

    lr = 0.02
    learner = MeanTestLearner(init_lr=lr)
    ii = 0

    while True:
        ii += 1

        loss_info = {}
        loss_info.update(net.compute_segment_cost())
        loss_info.update(net.compute_path_cost())

        loss = 0.0
        for cost_name, cost in loss_info.items():
            loss += cost

        log_txt = f'step:{ii} ' + \
                  ' '.join([f"{cost_name}:{cost.detach().numpy():.6f}" for cost_name, cost in loss_info.items()])
        loss_np = loss.detach().numpy()
        log_txt += f" loss:{loss_np:.6f} lr:{lr}"
        print(log_txt)

        info = learner.get_lr(loss_np)
        if info['state']:
            break
        lr = info['lr']

        loss.backward()
        with torch.no_grad():
            grad = net.control_pcd_tensor.grad
            grad[net.fix_pcd_idxs, :] = 0.0
            grad = grad / (torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-8)
            net.control_pcd_tensor -= grad * lr
            net.control_pcd_tensor.grad.zero_()

        net.update_state(with_tensor=True, with_np=False)

    # plt.plot(loss_record)
    # plt.show()

    net.update_state(with_tensor=False, with_np=True)

    # net.draw_segment(with_control=True, with_spline=True)
    with h5py.File('/home/admin123456/Desktop/work/path_examples/s5/debug/record.h5df', 'w') as f:
        for name, path_list in net.path_dict.items():
            for path_cell in path_list:
                xyzr = path_cell.get_bspline_xyzr_np(net.control_xyzr_np)
                f.create_dataset(name=name, data=xyzr)


if __name__ == '__main__':
    main()
