import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict
import matplotlib.pyplot as plt
import pyvista

from scripts_py.version_9.mapf_pkg.bspline_utils import BsplineUtils
from scripts_py.version_9.mapf_pkg.torch_utils import TorchMath, TorchUtils
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils


class CellRadiusAdjuster(object):
    def __init__(
            self, name: str, ts: np.ndarray, degree: int, knot: np.ndarray,
            c_xyz: np.ndarray, fill_last_one: bool, c_radius: np.ndarray
    ):
        self.name = name
        self.ts = ts
        self.degree = degree
        self.knot = knot
        self.c_xyz = c_xyz
        self.fill_last_one = fill_last_one
        self.c_radius = c_radius

        self.mat = BsplineUtils.compute_mat(self.ts, self.c_xyz.shape[0], self.degree, self.knot, self.fill_last_one)
        self.mat_t = TorchUtils.np2tensor(self.mat, require_grad=False)
        self.spline_xyz = self.mat.dot(c_xyz)

        self.curvature_radius = 1.0 / (BsplineUtils.compute_curvature(
            self.ts, self.c_xyz, self.degree, self.knot
        ) + 1e-2)
        self.curvature_radius_t = TorchUtils.np2tensor(self.curvature_radius, require_grad=False)

        self.c_radius_t = TorchUtils.np2tensor(self.c_radius, require_grad=True)
        self.target_radius_t = TorchUtils.np2tensor(self.compute_radius(), require_grad=False)

    def update_state(self):
        self.c_radius = TorchUtils.tensor2np(self.c_radius_t)

    def compute_radius(self):
        return self.mat.dot(self.c_radius).reshape(-1)

    def compute_cost(self, cfg: Dict):
        cost_dict = {}
        radius_t = self.mat_t.matmul(self.c_radius_t).reshape(-1)

        scale_error = F.relu(radius_t - self.curvature_radius_t)
        cost_dict[f'{self.name}_scaleCost'] = scale_error.sum() * cfg['scale_weight']

        cost_dict[f'{self.name}_balanceCost'] = torch.pow(radius_t - radius_t.mean(), 2.0).sum() * cfg['balance_weight']
        cost_dict[f'{self.name}_keepCost'] = torch.pow(radius_t - self.target_radius_t, 2.0).sum() * cfg['keep_weight']
        return cost_dict


class GroupRadiusAdjuster(object):
    def __init__(self):
        self.cell_dict: Dict[str, CellRadiusAdjuster] = {}

    def add_cell(
            self, name: str, ts: np.ndarray, degree: int, knot: np.ndarray,
            c_xyz: np.ndarray, fill_last_one: bool, c_radius: np.ndarray
    ):
        self.cell_dict[name] = CellRadiusAdjuster(name, ts, degree, knot, c_xyz, fill_last_one, c_radius)

    def optimize(self, optimize_cfg: Dict, cost_cfg: Dict):
        optimizer = TorchUtils.get_optimizer(
            [cell.c_radius_t for cell_name, cell in self.cell_dict.items()], lr=optimize_cfg['lr']
        )

        scheduler = TorchUtils.get_lr_scheduler(
            optimizer, factor=optimize_cfg['factor'], patience=optimize_cfg['patience'],
            cooldown=optimize_cfg['cooldown'], min_lr=optimize_cfg['min_lr']
        )

        loss_record, loss_info = [], {}
        iteration = -1
        while True:
            lr = scheduler.get_last_lr()[-1]

            loss_info.clear()
            for cell_name, cell in self.cell_dict.items():
                loss_info.update(cell.compute_cost(cost_cfg))

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

        for cell_name, cell in self.cell_dict.items():
            cell.update_state()

        loss_info_np = {}
        for key in list(loss_info.keys()):
            val = loss_info.pop(key)
            if isinstance(val, torch.Tensor):
                loss_info_np[key] = val.detach().numpy()
            else:
                loss_info_np[key] = val

        return loss_record, loss_info_np

    def draw_error_check(self, vis: VisUtils = None):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        for cell_name, cell in self.cell_dict.items():
            radius = cell.compute_radius()
            curvature_error = np.maximum(radius - cell.curvature_radius, 0.0)
            mesh = pyvista.PointSet(cell.spline_xyz)
            mesh['scalars'] = curvature_error
            vis.plot(mesh, point_size=6, scalars='scalars', show_scalar_bar=True, cmap=plt.cm.rainbow)

        if show_plot:
            vis.show()
