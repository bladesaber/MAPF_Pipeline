import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Union

import pyvista
import torch
import torch.nn.functional as F

from scripts_py.version_9.mapf_pkg.torch_utils import TorchUtils, CostUtils
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils
from scripts_py.version_9.mapf_pkg.bspline_utils import BsplineUtils


class SegmentCell(object):
    def __init__(self, idx: Union[int, str], flags: List[int], color: Tuple = (1.0, 0.0, 0.0)):
        self.idx = idx
        self.color = color
        self.flags = flags
        self.tag = self.encode_tag(flags)
        self.pcd_idxs: np.ndarray = None

        self.spline_info = {
            'sample_num': 0, 'degree': 0, 'length': 0.0, 'density': 0.0,
            'knot': np.array([]), 'ts_list': np.array([]), 'spline_mat': np.array([]),
            'radius': np.array([]), 'update_length_threshold': np.inf
        }
        self.tensor_info = {
            'spline_mat_t': torch.Tensor(),
            'path_pcd_t': torch.Tensor(),
        }
        self.cost_cfgs: List[Dict] = []

        self.spring_radius_conflict_ratio = 0.01
        self.min_conflict_thickness = 0.01

    @staticmethod
    def encode_tag(flags: List[int]):
        if len(flags) < 4:
            label_flags = flags
        else:
            label_flags = [flags[0], flags[1], flags[-2], flags[-1]]
        return tuple(np.sort(np.array(label_flags)))

    def __str__(self):
        return f"segment_{self.idx}"

    def __len__(self):
        return len(self.flags)

    def update_spline_mat(self, c_pcd: np.ndarray, degree: int, sample_num: int):
        knot = BsplineUtils.create_knots(c_pcd[:, :3].shape[0], degree)
        ts_list = np.linspace(0.0, 1.0, sample_num)
        spline_mat = BsplineUtils.compute_mat(ts_list, c_pcd[:, :3].shape[0], degree, knot)
        self.spline_info.update({'knot': knot, 'ts_list': ts_list, 'spline_mat': spline_mat})
        self.spline_info['radius'] = self.spline_info['spline_mat'].dot(c_pcd[:, 3:4]).reshape(-1)

    def update_spline_info(
            self, c_pcd: np.ndarray, degree: int, density: float, update_threshold=0.1, tag=''
    ):
        length = BsplineUtils.compute_spline_length(c_pcd[self.pcd_idxs, :3], degree)
        length_change_ratio = np.abs(length / (self.spline_info['length'] + 1e-6) - 1.0)

        if length_change_ratio > update_threshold:
            sample_num = max(math.ceil(length / density), 5)
            self.update_spline_mat(c_pcd[self.pcd_idxs, :], degree, sample_num)
            self.spline_info.update({
                'sample_num': sample_num, 'degree': degree, 'length': length, 'density': density,
                'update_length_threshold': update_threshold
            })
            print(f"[DEBUG]: {tag} Segment idx:{self.idx} length:{length} update")
            return True
        return False

    def prepare_tensor(self):
        self.tensor_info['spline_mat_t'] = TorchUtils.np2tensor(self.spline_info['spline_mat'], require_grad=False)

    def update_tensor(self, control_pcd: torch.Tensor):
        """
        params:
            control_pcd: (x, y, z)[n x 3]
        """
        self.tensor_info['path_pcd_t'] = self.tensor_info['spline_mat_t'].matmul(control_pcd[self.pcd_idxs, :3])

    def get_spline_xyzr_np(self, xyzr: np.ndarray):
        return self.spline_info['spline_mat'].dot(xyzr[self.pcd_idxs, :])


class PathCell(object):
    def __init__(self, segment_list: List[SegmentCell], begin_vec: np.ndarray, end_vec: np.ndarray):
        self.segments = segment_list
        self.begin_vec = begin_vec
        self.end_vec = end_vec
        self.tensor_info = {
            'paths_pcd_t': [],
            'radius': []
        }
        self.cost_cfgs: List[Dict] = []

    def update_tensor(self, fill_terminate_vec=True):
        self.tensor_info['paths_pcd_t'].clear()
        self.tensor_info['radius'].clear()

        for i, seg_cell in enumerate(self.segments):
            path_pcd_t = seg_cell.tensor_info['path_pcd_t']

            seg_tensor_list = []
            if i == 0:
                if fill_terminate_vec:
                    pos0 = TorchUtils.tensor2np(path_pcd_t[0, :])
                    pos0_tensor = TorchUtils.np2tensor((pos0 - self.begin_vec).reshape((1, -1)), require_grad=False)
                    seg_tensor_list.append(pos0_tensor)
                seg_tensor_list.append(path_pcd_t)
                self.tensor_info['radius'].append(seg_cell.spline_info['radius'])

            elif i == len(self.segments) - 1:
                seg_tensor_list.append(path_pcd_t[1:, :])
                self.tensor_info['radius'].append(seg_cell.spline_info['radius'][1:])

                if fill_terminate_vec:
                    pos1 = TorchUtils.tensor2np(path_pcd_t[-1, :])
                    pos1_tensor = TorchUtils.np2tensor((pos1 + self.end_vec).reshape((1, -1)), require_grad=False)
                    seg_tensor_list.append(pos1_tensor)

            else:
                seg_tensor_list.append(path_pcd_t[1:, :])
                self.tensor_info['radius'].append(seg_cell.spline_info['radius'][1:])

            if len(seg_tensor_list) > 1:
                seg_pcd_t = TorchUtils.concatenate(seg_tensor_list, dim=0)
            else:
                seg_pcd_t = seg_tensor_list[0]

            self.tensor_info['paths_pcd_t'].append(seg_pcd_t)

    def get_spline_xyzr_np(self, control_xyzr: np.ndarray, fill_terminate_vec=False, is_split=False):
        path = []
        for i, cell in enumerate(self.segments):
            xyzr = cell.get_spline_xyzr_np(control_xyzr)
            if i == 0 and fill_terminate_vec:
                begin_terminate = xyzr[0:1, :].copy()
                begin_terminate[0, :3] = begin_terminate[0, :3] - self.begin_vec
                xyzr = np.concatenate([begin_terminate, xyzr], axis=0)
            elif i == len(self.segments) - 1 and fill_terminate_vec:
                end_terminate = xyzr[-1:, :].copy()
                end_terminate[0, :3] = end_terminate[0, :3] + self.end_vec
                xyzr = np.concatenate([xyzr, end_terminate], axis=0)

            if not is_split:
                if i > 0:
                    xyzr = xyzr[1:, :]
            path.append(xyzr)

        if not is_split:
            path = np.concatenate(path, axis=0)
        return path


class NetworkPath(object):
    def __init__(self, path_dict: Dict[str, np.ndarray], group_idx: int, grid_cfg: Dict):
        """
        params:
            path_dict: Dict(path_name, xyzr[n x 4]), xyz should be integer path
        """
        self.group_idx = group_idx
        self.encode_cfg = self.get_encode_cfg(grid_cfg)

        self.graph = nx.Graph()
        self.node2info: Dict[Union[int, str], Dict[str, Union[int, float, str]]] = {}

        control_xyzr_np = []
        for task_name, xyzr_list in path_dict.items():
            last_flag = None
            for i, xyzr in enumerate(xyzr_list):
                flag = self.encode_xyz(xyzr[:3], self.encode_cfg)
                if flag not in self.node2info.keys():
                    control_xyzr_np.append(xyzr)
                    self.node2info[flag] = {'pcd_idx': len(control_xyzr_np) - 1, 'fix': False, 'type': 'cell'}
                if i > 0:
                    self.graph.add_edge(last_flag, flag)
                last_flag = flag

        # # ------ debug connect graph
        # degree_count = {}
        # for node_name in self.graph.nodes:
        #     node_degree = self.graph.degree(node_name)
        #     if node_degree not in degree_count:
        #         degree_count[node_degree] = 0
        #     degree_count[node_degree] += 1
        # print('degree_count:', degree_count, ' shape:', len(self.graph))
        # self.draw_network()
        # # ------

        self.control_xyzr_np = np.array(control_xyzr_np)
        # print(f"[INFO]: insert path nodes:{self.control_xyzr_np.shape[0]}, network nodes:{len(self.graph.nodes)}")

        self.segments: Dict[Union[Tuple, int], SegmentCell] = {}
        self.path_dict: Dict[str, List[PathCell]] = {}  # 多个工况路径

        self.control_pcd_tensor: torch.Tensor = None
        self.fix_pcd_idxs: np.ndarray = None

    @staticmethod
    def get_encode_cfg(grid_cfg: Dict):
        size_of_xyz = np.array([
            grid_cfg['size_of_x'] + 1, grid_cfg['size_of_y'] + 1, grid_cfg['size_of_z'] + 1
        ])
        xyz_init = np.array([
            grid_cfg['grid_min'][0], grid_cfg['grid_min'][1], grid_cfg['grid_min'][2]
        ])
        xyz_grid_length = np.array([
            grid_cfg['x_grid_length'], grid_cfg['y_grid_length'], grid_cfg['z_grid_length']
        ])
        z_code = size_of_xyz[0] * size_of_xyz[1]
        y_code = size_of_xyz[0]

        encode_cfg = {
            'size_of_xyz': size_of_xyz,
            'xyz_init': xyz_init,
            'xyz_grid_length': xyz_grid_length,
            'z_code': int(z_code),
            'y_code': int(y_code)
        }
        return encode_cfg

    @staticmethod
    def encode_xyz(xyz, encode_cfg):
        xyz_grid = np.round((xyz - encode_cfg['xyz_init']) / encode_cfg['xyz_grid_length'])
        xyz_grid = xyz_grid.astype(int)
        return xyz_grid[2] * encode_cfg['z_code'] + xyz_grid[1] * encode_cfg['y_code'] + xyz_grid[0]

    def add_path(
            self, name: str, begin_xyz: np.ndarray, end_xyz: np.ndarray, begin_vec: np.ndarray, end_vec: np.ndarray
    ):
        begin_flag = self.encode_xyz(begin_xyz, self.encode_cfg)
        end_flag = self.encode_xyz(end_xyz, self.encode_cfg)
        self.node2info[begin_flag].update({'fix': True, 'type': 'terminal'})
        self.node2info[end_flag].update({'fix': True, 'type': 'terminal'})

        path_flag_list = nx.all_simple_paths(self.graph, begin_flag, end_flag)
        # path_flag_list = [nx.shortest_path(self.graph, begin_flag, end_flag)]

        self.path_dict[name] = []
        for path_flags in path_flag_list:
            segment_list, segment_flags, path_length = [], [], len(path_flags)
            for i, flag in enumerate(path_flags):
                segment_flags.append(flag)
                if self.graph.degree(flag) <= 2 and (i < path_length - 1):  # node is in the middle and degree = 2
                    continue
                if i < path_length - 1:
                    self.node2info[flag].update({'type': 'connector'})

                seg_cell = SegmentCell(idx=len(self.segments), flags=segment_flags.copy())
                # if segment exist, return exist segment
                seg_cell = self.segments.setdefault(seg_cell.tag, seg_cell)

                segment_list.append(seg_cell)
                segment_flags = segment_flags[-1:]  # save as begin of segment

            self.path_dict[name].append(PathCell(segment_list, begin_vec, end_vec))

    def assemble_and_expand_graph(self, degree: int = 0):
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
                    info0, info1 = self.node2info[flag0], self.node2info[flag1]
                    xyzr0, xyzr1 = self.control_xyzr_np[info0['pcd_idx']], self.control_xyzr_np[info1['pcd_idx']]

                    new_xyzr = (xyzr0 + xyzr1) * 0.5
                    self.control_xyzr_np = np.concatenate([self.control_xyzr_np, new_xyzr.reshape((1, -1))], axis=0)
                    self.node2info[new_node_id] = {
                        'pcd_idx': self.control_xyzr_np.shape[0] - 1, 'fix': False, 'type': 'cell'
                    }

                    self.graph.remove_edge(flag0, flag1)
                    self.graph.add_edge(flag0, new_node_id)
                    self.graph.add_edge(new_node_id, flag1)
                    new_flags.insert(i + shift + 1, new_node_id)  # be careful the shift para since new nodes inserted

                    shift += 1
                    new_node_id += 1

                cell.flags = new_flags

        for seg_idx, cell in self.segments.items():
            cell.pcd_idxs = np.array([self.node2info[node_idx]['pcd_idx'] for node_idx in cell.flags])

    def update_fix_idxs(self):
        fix_idxs = []
        for node_idx, info in self.node2info.items():
            if info['fix']:
                fix_idxs.append(info['pcd_idx'])
        self.fix_pcd_idxs = np.array(fix_idxs)

    def update_optimize_info(self, segment_infos: Dict, path_infos: Dict):
        scale_np = np.zeros(self.control_xyzr_np.shape[0])
        update_np = np.zeros(self.control_xyzr_np.shape[0], dtype=bool)

        for seg_idx, seg_cell in self.segments.items():
            seg_info = segment_infos[seg_idx]
            seg_cell.color = seg_info['color']
            seg_cell.cost_cfgs = seg_info['costs']
            seg_cell.spring_radius_conflict_ratio = seg_info['spring_radius_conflict_ratio']
            seg_cell.min_conflict_thickness = seg_info['min_conflict_thickness']
            scale_np[seg_cell.pcd_idxs] = np.maximum(scale_np[seg_cell.pcd_idxs], seg_info['radius_scale'])
            update_np[seg_cell.pcd_idxs] = True
        self.control_xyzr_np[update_np, 3] = self.control_xyzr_np[update_np, 3] * scale_np[update_np]

        for seg_idx, seg_cell in self.segments.items():
            seg_info = segment_infos[seg_idx]
            seg_cell.update_spline_info(
                self.control_xyzr_np, seg_info['bspline_degree'],
                seg_info['bspline_density'], seg_info['update_bspline_length_threshold'],
                tag=f"Group id:{self.group_idx}"
            )

        for name, path_list in self.path_dict.items():
            for i, path_cell in enumerate(path_list):
                path_cell.cost_cfgs = path_infos[name][i]

        self.update_fix_idxs()

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

    def compute_segment_cost(self):
        cost_dict = {}
        for _, seg_cell in self.segments.items():
            for cfg in seg_cell.cost_cfgs:
                if cfg['method'] == 'spring_length':
                    cost = CostUtils.compute_spring_length_cost(seg_cell.tensor_info['path_pcd_t'], cfg)
                elif cfg['method'] == 'control_distribute_constraint':
                    cost = CostUtils.compute_segment_constraint(self.control_pcd_tensor[seg_cell.pcd_idxs, :], cfg)
                elif cfg['method'] == 'spline_distribute_constraint':
                    cost = CostUtils.compute_segment_constraint(seg_cell.tensor_info['path_pcd_t'], cfg)
                elif cfg['method'] == 'curvature':
                    radius_np = seg_cell.spline_info['radius']
                    cfg['value'] = 1.0 / TorchUtils.np2tensor(radius_np[1:-1] * cfg['radius_scale'], False)
                    cost = CostUtils.compute_curvature_cost(seg_cell.tensor_info['path_pcd_t'], cfg)
                elif cfg['method'] == 'control_spring_length':
                    cost = CostUtils.compute_spring_length_cost(self.control_pcd_tensor[seg_cell.pcd_idxs, :], cfg)
                elif cfg['method'] == 'control_spring_variance':
                    cost = CostUtils.compute_spring_variance_cost(self.control_pcd_tensor[seg_cell.pcd_idxs, :], cfg)
                else:
                    raise ValueError
                cost_dict.update({f"{cfg['method']}_{seg_cell.idx}": cost})
        return cost_dict

    def compute_path_cost(self):
        cost_dict = {}
        for path_name, path_list in self.path_dict.items():
            for path_j, path_cell in enumerate(path_list):
                for cfg in path_cell.cost_cfgs:
                    if cfg['method'] == 'connector_curvature':
                        path_pcd_list = path_cell.tensor_info['paths_pcd_t']
                        radius_list = path_cell.tensor_info['radius']

                        seg_num = len(path_pcd_list)
                        for i in range(seg_num - 1):
                            left_idx, right_idx = path_cell.segments[i].idx, path_cell.segments[i + 1].idx
                            tag = f"{cfg['method']}_{min(left_idx, right_idx)}_{max(left_idx, right_idx)}"
                            if tag in cost_dict.keys():
                                continue

                            left_pcd, right_pcd = path_pcd_list[i], path_pcd_list[i + 1]
                            left_radius, right_radius = radius_list[i], radius_list[i + 1]
                            radius_cat = np.concatenate([
                                left_radius[-cfg['left_num']:], right_radius[:cfg['right_num']]
                            ], axis=-1)
                            cfg['value'] = 1.0 / TorchUtils.np2tensor(radius_cat[1:-1] * cfg['radius_scale'], False)

                            cost = CostUtils.compute_connector_curvature_cost(left_pcd, right_pcd, cfg)
                            cost_dict.update({tag: cost})
        return cost_dict

    def update_control_pcd(self, loss: torch.Tensor, lr: float):
        loss.backward()
        with torch.no_grad():
            self.control_pcd_tensor.grad[self.fix_pcd_idxs, :] = 0.0
            self.control_pcd_tensor -= self.control_pcd_tensor.grad * lr
            self.control_pcd_tensor.grad.zero_()

    def draw_network(self, graph):
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(graph, pos, edge_color='black', width=1)
        nx.draw_networkx_labels(graph, pos, font_size=12)
        plt.show()

    def draw_segment(self, with_control=True, with_spline=False, with_tube=False, vis: VisUtils = None):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        print(f"[INFO]: Group idx:{self.group_idx} Segment num:{len(self.segments)}")
        for seg_idx, seg_cell in self.segments.items():
            print(f"[INFO] Segment id:{seg_idx}, num:{seg_cell.pcd_idxs.shape[0]}")
            color = np.random.uniform(0.0, 1.0, size=(3,))
            if with_control:
                control_pcd = self.control_xyzr_np[seg_cell.pcd_idxs, :3]
                line_set = VisUtils.create_line_set(np.arange(0, seg_cell.pcd_idxs.shape[0], 1))
                mesh = VisUtils.create_line(control_pcd, line_set)
                vis.plot(mesh, color=color, point_size=6)

            if with_spline:
                xyzr = seg_cell.get_spline_xyzr_np(self.control_xyzr_np)
                line_set = VisUtils.create_line_set(np.arange(0, xyzr.shape[0], 1))
                mesh = VisUtils.create_line(xyzr[:, :3], line_set)
                vis.plot(mesh, color=color, style='wireframe', point_size=0.1, line_width=2.0)

            if with_tube:
                xyzr = seg_cell.get_spline_xyzr_np(self.control_xyzr_np)
                line_set = VisUtils.create_line_set(np.arange(0, xyzr.shape[0], 1))
                tube_mesh = VisUtils.create_deformation_tube(xyzr[:, :3], line_set, radius=None, scalars=xyzr[:, -1])
                vis.plot(tube_mesh, color, opacity=1.0)

        if show_plot:
            vis.show()

    def draw_points(self, with_point_type=False, with_point_fix=False, vis: VisUtils = None):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        print(f"[INFO]: Group idx:{self.group_idx} Segment num:{len(self.segments)}")
        for seg_idx, seg_cell in self.segments.items():
            pcd_idxs = np.array([self.node2info[node_idx]['pcd_idx'] for node_idx in seg_cell.flags])
            color = np.random.uniform(0.0, 1.0, size=(3,))
            control_pcd = self.control_xyzr_np[pcd_idxs, :3]
            line_set = VisUtils.create_line_set(np.arange(0, pcd_idxs.shape[0], 1))
            mesh = VisUtils.create_line(control_pcd, line_set)
            vis.plot(mesh, color=color, point_size=4)

        if with_point_type and (not with_point_fix):
            for flag, info in self.node2info.items():
                xyzr = self.control_xyzr_np[info['pcd_idx'], :]
                if info['type'] == 'terminal':
                    color = np.array([1., 0., 0.])
                elif info['type'] == 'connector':
                    color = np.array([0., 0., 1.])
                else:
                    color = np.array([0., 1., 0.])
                mesh = VisUtils.create_sphere(xyzr[:3], radius=0.15)
                vis.plot(mesh, color=color)

        if with_point_fix and (not with_point_type):
            for flag, info in self.node2info.items():
                xyzr = self.control_xyzr_np[info['pcd_idx'], :]
                if info['fix']:
                    color = np.array([1., 0., 0.])
                else:
                    color = np.array([0., 0., 1.])
                mesh = VisUtils.create_sphere(xyzr[:3], radius=0.15)
                vis.plot(mesh, color=color)

        if show_plot:
            vis.show()

    def draw_path(self, with_control_points=False, with_spline=False, is_tensor=False):
        for name, path_list in self.path_dict.items():
            for path_cell in path_list:
                if with_spline:
                    if is_tensor:
                        merge_path_t = TorchUtils.concatenate(path_cell.tensor_info['paths_pcd_t'], dim=0)
                        pcd = TorchUtils.tensor2np(merge_path_t)
                    else:
                        pcd = path_cell.get_spline_xyzr_np(self.control_xyzr_np)[:, :3]
                    line_set = VisUtils.create_line_set(np.arange(0, pcd.shape[0], 1))
                    mesh = VisUtils.create_line(pcd, line_set)
                    mesh.plot()

                if with_control_points:
                    vis = VisUtils()
                    for seg_cell in path_cell.segments:
                        pcd = self.control_xyzr_np[seg_cell.pcd_idxs, :3]
                        line_set = VisUtils.create_line_set(np.arange(0, pcd.shape[0], 1))
                        mesh = VisUtils.create_line(pcd, line_set)
                        color = np.random.uniform(0.0, 1.0, size=3)
                        vis.plot(mesh, color)
                    vis.show()


class NetworkSimplePath(NetworkPath):
    def __init__(self, group_idx: int, grid_cfg: Dict):
        """
        params:
            path_dict: Dict(path_name, xyzr[n x 4]), xyz should be integer path
        """
        self.group_idx = group_idx
        self.encode_cfg = self.get_encode_cfg(grid_cfg)

        self.node_graph: nx.Graph = None
        self.control_xyzr_np: np.ndarray = None
        self.node2info: Dict[Union[int, str], Dict[str, Union[int, float, str, np.ndarray]]] = None

        self.node_graph: nx.Graph = None
        self.segments: Dict[Union[Tuple, int, str], SegmentCell] = None
        self.path_dict: Dict[str, List[PathCell]] = {}  # 多个工况路径

        self.control_pcd_tensor: torch.Tensor = None
        self.fix_pcd_idxs: np.ndarray = None

    @staticmethod
    def encode_segment_tag(flags: List[int]):
        if len(flags) < 4:
            label_flags = flags
        else:
            label_flags = [flags[0], flags[1], flags[-2], flags[-1]]
        return tuple(np.sort(np.array(label_flags)))

    def assemble_graph(self, path_dict: Dict[str, np.ndarray], graph_cfg: Dict):
        # ------ Step 1 find all segments
        tmp_graph, tmp_node2xyzr = nx.Graph(), {}
        for task_name, xyzr_np in path_dict.items():
            last_flag = None
            for i, xyzr in enumerate(xyzr_np):
                flag = self.encode_xyz(xyzr[:3], self.encode_cfg)
                tmp_node2xyzr[flag] = xyzr
                if i > 0:
                    tmp_graph.add_edge(last_flag, flag)
                last_flag = flag

        # ------ travel all path to generate all segments
        terminal_nodes = []
        for node in tmp_graph.nodes:
            if tmp_graph.degree(node) == 1:
                terminal_nodes.append(node)

        seg_book = {}
        begin_flag = terminal_nodes[0]
        for end_flag in terminal_nodes[1:]:
            path_flag_list = nx.all_simple_paths(tmp_graph, begin_flag, end_flag)
            for path_flags in path_flag_list:
                seg_flag, seg_xyzr = [], []
                path_length = len(path_flags)
                for i, flag in enumerate(path_flags):
                    seg_flag.append(flag)
                    seg_xyzr.append(tmp_node2xyzr[flag])
                    if tmp_graph.degree(flag) <= 2 and (i < path_length - 1):  # node is in the middle and degree = 2
                        continue

                    seg_flag_data = seg_flag.copy()
                    seg_xyzr_data = np.array(seg_xyzr.copy())
                    seg_tag = self.encode_segment_tag(seg_flag_data)
                    if seg_tag not in seg_book.keys():
                        seg_book[seg_tag] = {
                            'flags': seg_flag_data,
                            'xyzr_np': seg_xyzr_data
                        }

                    seg_flag = seg_flag[-1:]  # save as begin of segment
                    seg_xyzr = seg_xyzr[-1:]

        # ------ Step 2 refit graph
        node_graph = nx.Graph()
        control_xyzr_np = []
        node2info = {}
        segments = {}
        segment_degree_min, segment_degree_max = graph_cfg['segment_degree_min'], graph_cfg['segment_degree_max']
        assert segment_degree_min > 4
        extend_flag_idx = np.max(list(tmp_graph.nodes)) + 1

        for seg_idx, (_, seg_book) in enumerate(seg_book.items()):
            seg_idx = f"seg_{seg_idx}"
            seg_length = len(seg_book['flags'])
            xyzr_np = seg_book['xyzr_np']
            seg_flags = seg_book['flags']

            if seg_length < segment_degree_min:
                while seg_length < segment_degree_min:
                    expand_xyzr, expand_flags = [], []
                    for i in range(xyzr_np.shape[0] - 1):
                        expand_xyzr.append(xyzr_np[i])
                        expand_xyzr.append((xyzr_np[i] + xyzr_np[i + 1]) * 0.5)

                        expand_flags.append(seg_flags[i])
                        expand_flags.append(extend_flag_idx)
                        extend_flag_idx += 1

                    expand_xyzr.append(xyzr_np[-1])
                    expand_flags.append(seg_flags[-1])

                    xyzr_np = np.array(expand_xyzr)
                    seg_flags = expand_flags.copy()
                    seg_length = len(seg_flags)

            elif seg_length > segment_degree_max:
                important_bool = np.zeros(len(seg_flags), dtype=bool)
                for j, flag in enumerate(seg_flags):
                    if tmp_graph.degree(flag) == 2:
                        important_bool[j] = True

                all_idxs = np.arange(0, seg_length, 1)
                select_idxs = all_idxs[important_bool]
                select_idxs = np.random.choice(
                    select_idxs, size=min(segment_degree_max, select_idxs.shape[0]), replace=False
                )
                select_idxs = np.concatenate([all_idxs[~important_bool], select_idxs], axis=-1)
                select_idxs = np.sort(select_idxs)

                xyzr_np = xyzr_np[select_idxs, :]
                seg_flags = np.array(seg_flags)[select_idxs]

            last_flag = None
            for i, (xyzr, flag) in enumerate(zip(xyzr_np, seg_flags)):
                if flag not in node2info.keys():
                    control_xyzr_np.append(xyzr)
                    node2info[flag] = {
                        'pcd_idx': len(control_xyzr_np) - 1,
                        'fix': False,
                        'type': 'cell',
                        'seg_idx': set([seg_idx])
                    }
                else:
                    seg_idx_list = list(node2info[flag]['seg_idx'])
                    seg_idx_list.append(seg_idx)
                    node2info[flag].update({'seg_idx': set(seg_idx_list)})

                if i > 0:
                    node_graph.add_edge(last_flag, flag)
                last_flag = flag

            segments[seg_idx] = SegmentCell(seg_idx, seg_flags)

        for flag in node_graph.nodes:
            if node_graph.degree(flag) == 1:
                node2info[flag].update({'type': 'terminal', 'fix': True})
            elif node_graph.degree(flag) > 2:
                node2info[flag].update({'type': 'connector'})

        self.node_graph = node_graph
        self.segments = segments
        self.node2info = node2info
        self.control_xyzr_np = np.array(control_xyzr_np)

        for _, cell in self.segments.items():
            cell.pcd_idxs = np.array([self.node2info[node_idx]['pcd_idx'] for node_idx in cell.flags])

        # ------ debug connect graph
        print(f"[INFO group_{self.group_idx}]: "
              f"control nodes:{self.control_xyzr_np.shape[0]}, network nodes:{len(self.node_graph.nodes)}")
        degree_count = {}
        for node_name in self.node_graph.nodes:
            node_degree = self.node_graph.degree(node_name)
            if node_degree not in degree_count:
                degree_count[node_degree] = 0
            degree_count[node_degree] += 1
        print(f'[INFO group_{self.group_idx}]: degree_count:', degree_count, ' shape:', len(self.node_graph))
        # self.draw_network(self.node_graph)
        # ------

    def add_path(
            self, name: str, begin_xyz: np.ndarray, end_xyz: np.ndarray, begin_vec: np.ndarray, end_vec: np.ndarray
    ):
        self.path_dict[name] = []
        begin_flag = self.encode_xyz(begin_xyz, self.encode_cfg)
        end_flag = self.encode_xyz(end_xyz, self.encode_cfg)

        path_flag_list = nx.all_simple_paths(self.node_graph, begin_flag, end_flag)
        # path_flag_list = [nx.shortest_path(self.node_graph, begin_flag, end_flag)]

        for path_flags in path_flag_list:
            seg_list = []
            for flag in path_flags:
                if self.node_graph.degree(flag) == 2:
                    seg_idx_list = list(self.node2info[flag]['seg_idx'])
                    assert len(seg_idx_list) == 1
                    seg_idx = seg_idx_list[0]

                    if len(seg_list) == 0:
                        seg_list.append(seg_idx)
                    else:
                        if seg_idx != seg_list[-1]:
                            seg_list.append(seg_idx)

            seg_cell_list = []
            for seg_name in seg_list:
                seg_cell_list.append(self.segments[seg_name])
            self.path_dict[name].append(PathCell(seg_cell_list, begin_vec, end_vec))

    def assemble_and_expand_graph(self, degree: int = 0):
        raise NotImplementedError
