import os

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict
import matplotlib.pyplot as plt
import pyvista

from scripts_py.version_9.mapf_pkg.bspline_utils import BsplineUtils
from scripts_py.version_9.mapf_pkg.torch_utils import TorchMath, TorchUtils
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils

from scripts_py.version_9.mapf_pkg.cad_utils import OcctUtil, OcctVis, OcctIO, OcctAux, OcctOperation, OcctProfile


class CellRadiusAdjuster(object):
    def __init__(
            self,
            group_idx: int, idx: int, cell_type: str,
            c_idxs: np.ndarray, ts: np.ndarray, knot: np.ndarray,
            degree: int, fill_last_one: bool, c_radius_orig: np.ndarray, table_xyz: np.ndarray,
            left_clamp: bool, right_clamp: bool, left_vector: np.ndarray = None, right_vector: np.ndarray = None
    ):
        self.idx = idx
        self.group_idx = group_idx
        self.cell_type = cell_type
        self.ts = ts
        self.degree = degree
        self.knot = knot
        self.c_idxs = c_idxs
        self.c_xyz = table_xyz[self.c_idxs, :]
        self.n_control = self.c_idxs.shape[0]
        self.fill_last_one = fill_last_one
        self.left_clamp = left_clamp
        self.right_clamp = right_clamp
        self.left_vector = left_vector
        self.right_vector = right_vector

        self.mat = BsplineUtils.compute_mat(self.ts, self.n_control, self.degree, self.knot, self.fill_last_one)
        self.spline_xyz = self.mat.dot(self.c_xyz)

        mat_order1 = BsplineUtils.compute_first_order_mat(self.ts, self.n_control, self.degree, self.knot)
        self.tangent_vec = mat_order1.dot(self.c_xyz)
        if self.cell_type == 'segment':
            if self.left_clamp:
                self.tangent_vec[0] = self.left_vector
            if self.right_clamp:
                self.tangent_vec[-1] = self.right_vector
        self.tangent_vec = self.tangent_vec / np.linalg.norm(self.tangent_vec, ord=2, axis=1, keepdims=True)

        self.mat_t = TorchUtils.np2tensor(self.mat, require_grad=False)

        self.limit_c_radius = c_radius_orig
        self.limit_c_radius_t = TorchUtils.np2tensor(c_radius_orig, require_grad=False)

        self.inter_info = self.compute_radius_condition()

    def __len__(self):
        return self.n_control

    def __str__(self):
        return f"{self.group_idx}_{self.cell_type}_{self.idx}"

    def compute_radius_np(self, table_radius: np.ndarray):
        return self.mat.dot(table_radius[self.c_idxs].reshape((-1, 1))).reshape(-1)

    @staticmethod
    def compute_distance_point2line(pcd: np.ndarray, line_pcd: np.ndarray, line_vec: np.ndarray):
        line_vec = line_vec / np.linalg.norm(line_vec, ord=2, axis=1, keepdims=True)
        line2point = pcd - line_pcd
        line2point_length = np.linalg.norm(line2point, ord=2, axis=1)
        return np.sqrt(np.power(line2point_length, 2.0) - np.power(np.sum(line2point * line_vec, axis=1), 2.0))

    def compute_radius_condition(self):
        center_xyz = self.spline_xyz
        # vec_xyz = np.concatenate([
        #     self.spline_xyz[1:, :] - self.spline_xyz[:-1, :],
        #     self.spline_xyz[-1:, :] - self.spline_xyz[-2:-1, :]
        # ], axis=0)
        # vec_xyz = vec_xyz / np.linalg.norm(vec_xyz, ord=2, axis=1, keepdims=True)
        vec_xyz = self.tangent_vec

        vec_inter_group = np.cross(vec_xyz[:, :, np.newaxis], vec_xyz, axisa=1)
        vec_inter_group = vec_inter_group / np.linalg.norm(vec_inter_group, ord=2, axis=2, keepdims=True)

        inter_info = {}
        for i in range(center_xyz.shape[0]):
            vec_inter = vec_inter_group[i]
            valid_idxs = np.arange(0, center_xyz.shape[0], 1)

            valid_idxs = valid_idxs[np.bitwise_and(
                ~np.isnan(np.sum(vec_inter, axis=1)),
                np.sum(vec_inter, axis=1) != 0.0
            )]
            if valid_idxs.shape[0] == 0:
                continue

            xyz0, vec0 = center_xyz[i, :], vec_xyz[i, :]
            vec_inter_v = vec_inter[valid_idxs, :]
            vec_xyz_v = vec_xyz[valid_idxs, :]
            xyz_v = center_xyz[valid_idxs, :]

            A = np.concatenate([
                np.tile(vec0[np.newaxis, np.newaxis, np.newaxis, :], (1, vec_xyz_v.shape[0], 1, 1)),
                np.tile(vec_xyz_v[np.newaxis, :, np.newaxis, :], (1, 1, 1, 1)),
                np.tile(vec_inter_v[np.newaxis, :, np.newaxis, :], (1, 1, 1, 1))
            ], axis=-2)
            b = np.concatenate([
                np.tile(
                    np.sum(vec0 * xyz0)[np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                    (1, vec_xyz_v.shape[0], 1, 1)
                ),
                np.tile(
                    np.sum(vec_xyz_v * xyz_v, axis=1)[np.newaxis, :, np.newaxis, np.newaxis],
                    (1, 1, 1, 1)
                ),
                np.zeros((1, vec_xyz_v.shape[0], 1, 1))
            ], axis=-2)
            inter_ps = np.linalg.solve(A, b)[0, :, :, 0]

            dist0 = self.compute_distance_point2line(xyz0, inter_ps, vec_inter_v)
            dist1 = self.compute_distance_point2line(xyz_v, inter_ps, vec_inter_v)
            inter_info[i] = {'center_idx': i, 'valid_idxs': valid_idxs, 'dist0': dist0, 'dist1': dist1}

        return inter_info

    def compute_tensor_cost(self, table_radius: torch.Tensor, cfg: Dict):
        cost_dict = {}
        c_radius = table_radius[self.c_idxs]
        radius = self.mat_t.matmul(table_radius[self.c_idxs].reshape((-1, 1))).reshape(-1)

        lower_bound_cost = F.relu(-c_radius).sum()
        upper_bound_cost = F.relu(c_radius - self.limit_c_radius_t).sum()
        bound_cost = lower_bound_cost + upper_bound_cost
        cost_dict[f'{str(self)}_boundCost'] = bound_cost * cfg['bound_weight']

        scale_cost = 0.0
        for i, info in self.inter_info.items():
            radius_i = radius[i]
            radius_j = radius[info['valid_idxs']]
            limit_radius = info['dist0'] + info['dist1']
            limit_radius_t = TorchUtils.np2tensor(limit_radius, require_grad=False)
            scale_cost += F.relu(radius_i + radius_j - limit_radius_t + 0.1).sum()
        cost_dict[f'{str(self)}_scaleCost'] = scale_cost * cfg['scale_weight']

        # balance_cost = torch.pow(c_radius - c_radius.mean(), 2.0).sum()
        balance_cost = torch.pow(c_radius[1:] - c_radius[:-1], 2.0).sum()
        cost_dict[f'{str(self)}_balanceCost'] = balance_cost * cfg['balance_weight']

        keep_cost = torch.pow(c_radius - self.limit_c_radius_t, 2.0).sum()
        cost_dict[f'{str(self)}_keepCost'] = keep_cost * cfg['keep_weight']

        return cost_dict


class GroupRadiusAdjuster(object):
    def __init__(self, group_idx: int, table_xyz: np.ndarray, table_radius: np.ndarray):
        self.group_idx = group_idx
        self.table_xyz = table_xyz
        self.table_radius = table_radius
        self.table_radius_t = TorchUtils.np2tensor(table_radius, require_grad=True)

        self.cell_dict: Dict[str, CellRadiusAdjuster] = {}

    def add_cell(
            self, idx: int, group_idx: int, cell_type: str,
            ts: np.ndarray, degree: int, knot: np.ndarray,
            c_idxs: np.ndarray, fill_last_one: bool, c_radius: np.ndarray,
            left_clamp: bool, right_clamp: bool, left_vector: np.ndarray, right_vector: np.ndarray
    ):
        cell = CellRadiusAdjuster(
            group_idx, idx, cell_type, c_idxs, ts, knot, degree, fill_last_one, c_radius, self.table_xyz,
            left_clamp, right_clamp, left_vector, right_vector
        )
        self.cell_dict[str(cell)] = cell

    def compute_tensor_cost(self, cost_cfg: Dict):
        cost_dict = {}
        for name, cell in self.cell_dict.items():
            cost_dict.update(cell.compute_tensor_cost(self.table_radius_t, cost_cfg))
        return cost_dict

    def update_state(self):
        self.table_radius = TorchUtils.tensor2np(self.table_radius_t)

    def draw_wireframe(self, vis: VisUtils = None):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        for cell_name, cell in self.cell_dict.items():
            color = np.random.uniform(0.0, 1.0, size=(3,))
            line_set = VisUtils.create_line_set(np.arange(0, cell.spline_xyz.shape[0], 1))
            radius = cell.compute_radius_np(self.table_radius)

            # tube_mesh = VisUtils.create_deformation_tube(cell.spline_xyz, line_set, radius=None, scalars=radius)
            # vis.plot(tube_mesh, color=color, opacity=1.0, style='wireframe')

            # line_mesh = VisUtils.create_line(cell.spline_xyz, line_set)
            # vis.plot(line_mesh, color=color, opacity=1.0, style='wireframe')

            pcd_mesh = VisUtils.create_point_cloud(cell.spline_xyz)
            vis.plot(pcd_mesh, color=color, opacity=1.0)

            for xyz, vec, r in zip(cell.spline_xyz, cell.tangent_vec, radius):
                mesh = VisUtils.create_polygon(xyz, r, vec, n_sides=60)
                vis.plot(mesh, color=color, opacity=1.0, style='wireframe')

            vis.plotter.add_arrows(cell.spline_xyz, cell.tangent_vec, mag=0.1)

        if show_plot:
            vis.show()


class RadiusOptimizer(object):
    def __init__(self):
        self.group_dict: Dict[int, GroupRadiusAdjuster] = {}

    def add_cell(self, adjuster: GroupRadiusAdjuster):
        self.group_dict[adjuster.group_idx] = adjuster

    def optimize(self, optimize_cfg: Dict, cost_cfg: Dict):
        optimizer = TorchUtils.get_optimizer(
            [group_cell.table_radius_t for _, group_cell in self.group_dict.items()], lr=optimize_cfg['lr']
        )

        scheduler = TorchUtils.get_lr_scheduler(
            optimizer, factor=optimize_cfg['factor'], patience=optimize_cfg['patience'],
            cooldown=optimize_cfg['cooldown'], min_lr=optimize_cfg['min_lr']
        )

        loss_record, loss_info = [], {}
        iteration = 0
        while True:
            iteration += 1
            lr = scheduler.get_last_lr()[-1]

            loss_info.clear()
            for group_idx, group_cell in self.group_dict.items():
                loss_info.update(group_cell.compute_tensor_cost(cost_cfg))

            loss: torch.Tensor = 0.0
            for _, cost_value in loss_info.items():
                loss += cost_value

            # ------ log information
            if (iteration % 100 == 0) or (iteration == 1):
                loss_record.append(loss.detach().numpy())
                print(f"[iter:{iteration}/lr:{lr:.5f}] loss:{loss_record[-1]:.6f}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            # ------ prepare to break
            if iteration > optimize_cfg['max_iter']:
                break

            if lr <= optimize_cfg['min_lr']:
                break

        for _, group_cell in self.group_dict.items():
            group_cell.update_state()

        loss_info_np = {}
        for key in list(loss_info.keys()):
            val = loss_info.pop(key)
            if isinstance(val, torch.Tensor):
                loss_info_np[key] = val.detach().numpy()
            else:
                loss_info_np[key] = val

        return loss_record, loss_info_np

    def draw_wireframe(self, vis: VisUtils = None):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        for group_idx, group_cell in self.group_dict.items():
            group_cell.draw_wireframe(vis)

        if show_plot:
            vis.show()


class StepExporter(object):
    @staticmethod
    def to_step(group_cell: GroupRadiusAdjuster, save_dir, with_vis=False):
        if with_vis:
            display, start_display, _, _ = OcctVis.init_plotter()

        solid_list = []
        for name, cell in group_cell.cell_dict.items():
            spline_xyz = cell.spline_xyz
            radius_list = cell.compute_radius_np(group_cell.table_radius)
            spline_vec = cell.tangent_vec

            wire_list = []
            for xyz, vec, radius in zip(spline_xyz, spline_vec, radius_list):
                circle = OcctProfile.make_circle(xyz, vec, radius)
                circle_wire = OcctUtil.make_wire(OcctUtil.make_edge(circle))
                wire_list.append(circle_wire)

            solid = OcctProfile.make_thruSections(wire_list, is_solid=True, ruled=True)
            solid_list.append(solid)

            if with_vis:
                for wire in wire_list:
                    display.DisplayShape(wire, update=True, color='RED')
                # display.DisplayShape(solid, update=True, color='BLUE')

            # OcctIO.write_STEP(os.path.join(save_dir, f"solid_{cell.name}.STEP"), solid)

        merge_solid = solid_list[0]
        for solid in solid_list[1:]:
            merge_solid = OcctOperation.fuse(merge_solid, solid)

        if with_vis:
            display.DisplayShape(merge_solid, update=True, color='BLUE')

        OcctIO.write_STEP(os.path.join(save_dir, f"solid_{group_cell.group_idx}.STEP"), merge_solid)
