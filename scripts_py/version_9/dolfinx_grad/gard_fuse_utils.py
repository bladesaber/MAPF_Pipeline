import numpy as np
from functools import partial
from scipy.optimize import minimize
from typing import Callable, Dict, Union, Tuple, Literal
import torch.nn.functional as F
import torch
import pyvista
from tqdm import tqdm

from scripts_py.version_9.mapf_pkg.torch_utils import TorchUtils
from scripts_py.version_9.dolfinx_grad.deform_mesh_utils import SpringMeshObj
from scripts_py.version_9.dolfinx_grad.vis_utils import VisUtils


# todo 1.引导梯度不是关键，平滑表面以及保持拓扑结构才是，要提取表面连接关系以及内部连接关系，尝试用netwrokx处理
#  2.保留拓扑关系依赖胡克弹性均衡并不能解决网格拉长变形问题，因此需要角度扭矩弹性(或者添加正则项，不允许过长的边)，引导梯度和障碍物想象为外部力


class GradSpringFusionObj(object):
    def __init__(self, mesh: SpringMeshObj, obs_pcd: np.ndarray = None, obs_r=0.0):
        self.grad_dict: Dict[str, np.ndarray] = {}
        self.mesh = mesh

        self.with_obs = obs_pcd is not None
        self.obs_pcd = obs_pcd
        self.obs_r = obs_r

        self.pro_dict: Dict[str, Union[torch.Tensor, float, np.ndarray]] = {
            'geo_pcd': 0.0, 'x': 0.0, 'y': 0.0
        }
        self.loss_dict: Dict[str, torch.Tensor] = {}
        self.log_dict = {}

    def update_grad(self, name, grad: np.ndarray):
        self.grad_dict[name] = grad

    def concat_var(self, var: np.ndarray):
        if self.mesh.topology_dim == 2:
            var = np.concatenate([var, np.zeros(shape=(var.shape[0], 1))], axis=1)
        return var

    def assemble_problem(self, min_scale=1e-4, max_scale=0.05):
        assert 1.0 > max_scale > min_scale >= 1e-6
        self.pro_dict.update({
            'min_grad_scale': min_scale,
            'max_grad_scale': max_scale
        })

        grad_fused = 0.0
        for _, grad in self.grad_dict.items():
            grad_fused += grad[:, :self.mesh.topology_dim]
        grad_fused = grad_fused / float(len(self.grad_dict))

        grad_norm = np.linalg.norm(grad_fused, axis=1, ord=2)
        grad_fused = grad_fused / np.max(grad_norm) * max_scale

        # if self.debug_mode:
        #     self.pro_dict.update({'grad_orig': grad_fused})

        grad_norm = np.linalg.norm(grad_fused, axis=1, ord=2)
        grad_fused[grad_norm <= min_scale, :] = 0.0
        grad_norm[grad_norm <= min_scale] = 0.0

        self.pro_dict.update({
            'geo_pcd': TorchUtils.np2tensor(self.mesh.points[:, :self.mesh.topology_dim], require_grad=False),
            'grad_x': TorchUtils.np2tensor(
                grad_fused + np.random.uniform(-1., 1., size=grad_fused.shape) * min_scale * 1e-1,
                require_grad=True
            ),
            'grad_y': TorchUtils.np2tensor(grad_fused, require_grad=False),
            'grad_norm': grad_norm,
        })

        # ------ 这里测试表明grad_orig与grad_y效果几乎相同
        # if self.debug_mode:
        #     vis = VisUtils()
        #     vis.plot(self.mesh.grid, style='wireframe')
        #     vis.plot_grad(self.mesh.points, self.concat_var(self.pro_dict['grad_orig']), mag=2.0, color=(1., 0., 0.))
        #     vis.plot_grad(
        #         self.mesh.points,
        #         self.concat_var(TorchUtils.tensor2np(self.pro_dict['grad_y'])),
        #         mag=2.0, color=(0., 0., 1.)
        #     )
        #     vis.show()

    def cal_constant_node_loss(self):
        geo_pcd: torch.Tensor = self.pro_dict['geo_pcd']
        geo_pcd_new: torch.Tensor = geo_pcd + self.pro_dict['grad_x']
        loss = torch.norm(
            geo_pcd_new[self.mesh.bc_fix_dofs, :] - geo_pcd[self.mesh.bc_fix_dofs, :], p=2, dim=1
        )
        loss = torch.mean(loss) + torch.max(loss)
        self.loss_dict.update({'loss_constant': loss})

    def cal_obs_loss(self, unit_lr: float):
        conf_mesh_idxs, conf_obs_idxs = self.mesh.detect_obstacle(self.obs_pcd, self.obs_r, unit_lr)
        if conf_mesh_idxs.shape[0] > 0:
            relate_obs_pcd = self.obs_pcd[conf_obs_idxs, :self.mesh.topology_dim]
            obs_pcd_t = TorchUtils.np2tensor(relate_obs_pcd, require_grad=False)

            require_dist = np.full_like(conf_mesh_idxs, fill_value=self.obs_r, dtype=float)
            require_dist_t = TorchUtils.np2tensor(require_dist.reshape(-1), require_grad=False)

            geo_pcd_new: torch.Tensor = self.pro_dict['geo_pcd'] + self.pro_dict['grad_x']
            dist_t = torch.norm(geo_pcd_new[conf_mesh_idxs, :] - obs_pcd_t, p=2, dim=1)

            loss = F.relu(require_dist_t - dist_t)
            loss = torch.mean(loss) + torch.max(loss)
            self.loss_dict.update({'loss_obs': loss})

    def cal_guide_loss(self, scale_tol=0.025, angle_tol=1e-3):
        # todo fail
        bc_grad_x: torch.Tensor = self.pro_dict['grad_x'][self.mesh.bc_free_dofs, :]
        bc_grad_y: torch.Tensor = self.pro_dict['grad_y'][self.mesh.bc_free_dofs, :]
        bc_grad_norm = self.pro_dict['grad_norm'][self.mesh.bc_free_dofs]

        zero_bool = bc_grad_norm == 0.
        non_zero_bool = ~zero_bool

        x1 = bc_grad_x[non_zero_bool, :]
        y1 = bc_grad_y[non_zero_bool, :]
        scale_x1 = torch.norm(x1, p=2, dim=1, keepdim=False)
        scale_y1 = torch.norm(y1, p=2, dim=1, keepdim=False)
        cos_theta = torch.sum(x1 * y1, dim=1) / (scale_x1 * scale_y1)

        loss_p1_angle = 1.0 - cos_theta
        loss_p1_angle = F.relu(loss_p1_angle - angle_tol)
        loss_p1_angle = torch.mean(loss_p1_angle) + torch.max(loss_p1_angle)

        # ------ 使用ratio scale很容易过于敏感
        # loss_p1_scale = torch.sqrt(torch.pow(scale_x1 / scale_y1 - 1.0, 2.0))
        # loss_p1_scale = F.relu(loss_p1_scale - scale_tol)
        # loss_p1_scale = torch.mean(loss_p1_scale) + torch.max(loss_p1_scale)

        loss_p1_scale = torch.sqrt(torch.pow(scale_x1 - scale_y1, 2.0))
        loss_p1_scale = torch.mean(loss_p1_scale) + torch.max(loss_p1_scale)

        x2 = bc_grad_x[zero_bool, :]
        # y2 = bc_grad_y[zero_bool, :]
        loss_p2 = torch.sqrt(torch.pow(x2, 2.0))
        loss_p2 = torch.mean(loss_p2) + torch.max(loss_p2)

        self.loss_dict.update({
            'loss_p1_angle': loss_p1_angle,
            'loss_p1_scale': loss_p1_scale,
            'loss_p2': loss_p2
        })

    def cal_geo_loss(self, tol=0.25):
        # todo fail 这里不能够充分表述一个网格结构，它不能表述冲突的问题

        geo_pcd = self.pro_dict['geo_pcd']
        geo_pcd_new = geo_pcd + self.pro_dict['grad_x']

        pair_list = [[0, 1], [1, 2], [0, 2]]
        loss = 0.0
        for pair_i, pair_j in pair_list:
            con_i, con_j = self.mesh.connectivity[:, pair_i], self.mesh.connectivity[:, pair_j]
            len_ref = torch.norm(geo_pcd[con_i, :] - geo_pcd[con_j, :], p=2, dim=1)
            len_new = torch.norm(geo_pcd_new[con_i, :] - geo_pcd_new[con_j, :], p=2, dim=1)

            # ------ 使用ratio scale很容易过于敏感
            len_dif = torch.sqrt(torch.pow(len_new / len_ref - 1.0, 2.0))
            len_dif = F.relu(len_dif - tol)
            loss += torch.mean(len_dif) + torch.max(len_dif)

            # len_dif = torch.sqrt(torch.pow(len_new - len_ref, 2.0))
            # loss += torch.mean(len_dif) + torch.max(len_dif)

        self.loss_dict.update({'loss_geo': loss})

    def compute_loss(self, weight_dict):
        self.cal_constant_node_loss()
        self.cal_geo_loss()
        self.cal_guide_loss()
        self.cal_obs_loss(self.pro_dict['max_grad_scale'])

        loss = 0.0
        for name, loss_t in self.loss_dict.items():
            loss += loss_t * weight_dict[name]
            self.log_dict.update({name: self.loss_dict[name].detach().numpy() * weight_dict[name]})

        return loss

    def fit_grad(self, max_iter=1000, with_debug=False):
        debug_bar = tqdm()

        # ------ debug grad visualization
        if with_debug:
            vis = VisUtils()
            vis.plot(self.mesh.grid, style='wireframe')
            # grad_y_np = TorchUtils.tensor2np(self.pro_dict['grad_y'])
            # vis.plot_grad(self.mesh.points, self.concat_var(grad_y_np), mag=1.0, color=(1., 0., 0.))
        # -------------------------------

        lr = self.pro_dict['min_grad_scale']
        min_lr = lr * 0.1
        optimizer = TorchUtils.get_optimizer([self.pro_dict['grad_x']], lr=lr)
        scheduler = TorchUtils.get_lr_scheduler(optimizer, factor=0.25, patience=100, cooldown=100, min_lr=min_lr)

        loss_first = None
        iteration = -1
        while True:
            iteration += 1
            loss = self.compute_loss(
                {
                    'loss_constant': 1000.0,  # loss_constant=1.0，不动点移动距离为1e-4
                    'loss_obs': 1000.0,  # loss_obs=1.0，与障碍物重叠距离为1e-4
                    'loss_p1_angle': 10.0,  # 当loss_p1_angle=1.0，误差为3度
                    'loss_p1_scale': 10.0,  # loss_p1_scale=1.0，scale变幅为0.1
                    'loss_p2': 10.0,  # loss_p2=1.0，0梯度位置出现1e-4的随机梯度
                    'loss_geo': 10.0  # loss_geo=1.0，边的距离变动比例为0.2
                }
            )

            # ------ debug vis
            desc = f'[iter:{iteration}/lr:{scheduler.get_last_lr()[-1]:.5f}] '
            for loss_n, loss_f in self.log_dict.items():
                desc += f'{loss_n}:{loss_f:.3f} '
            loss_cur = np.sum(list(self.log_dict.values()))
            if loss_first is None:
                loss_first = loss_cur
            desc += f'loss_rho:{loss_cur / loss_first:.2f}.'

            debug_bar.set_description(desc)
            debug_bar.update(iteration)
            # print(desc)
            # ------

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            if iteration > max_iter:
                break

            if lr <= min_lr:
                break

        grad_res = TorchUtils.tensor2np(self.pro_dict['grad_x'], with_copy=True)

        # ------ debug grad visualization
        if with_debug and ('loss_obs' in list(self.loss_dict.keys())):
            vis.plot_grad(self.mesh.points, self.concat_var(grad_res), mag=1.0, color=(0., 0., 1.))
            vis.plot((pyvista.PointSet(self.obs_pcd)))

            grid_new = self.mesh.grid.copy()
            grid_new.points = grid_new.points + self.concat_var(grad_res)
            vis.plot(grid_new, style='wireframe', color=(0., 1., 0.))

            vis.show()
        # -----------

        return grad_res
