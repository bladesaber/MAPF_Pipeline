import numpy as np
from typing import Callable, Dict, Union, Tuple, Literal, List
import torch
import pyvista

from scripts_py.version_9.dolfinx_grad.vis_utils import VisUtils
from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshObj
from scripts_py.version_9.dolfinx_grad.spring_mesh_utils import SpringFeaMeshObj, SpringTorchMeshObj


class GradRawFusion(object):
    def __init__(self, mesh: MeshObj):
        self.mesh: Union[MeshObj, SpringFeaMeshObj] = mesh
        self.grad_dict: Dict[str, np.ndarray] = {}

    @property
    def style(self):
        return 'GradRawFusion'

    def add_grad(self, name, grad: np.ndarray):
        self.grad_dict[name] = grad

    def solve_disp(self, max_step: float):
        grad_agg = 0.0
        for _, grad in self.grad_dict.items():
            grad_agg += grad[:, :self.mesh.topology_dim]
        grad_agg = grad_agg / float(len(self.grad_dict))
        grad_norm = np.linalg.norm(grad_agg, axis=1, ord=2)
        grad_agg = grad_agg / np.max(grad_norm) * max_step
        return grad_agg

    def draw_problem(self, disp: np.ndarray = None, with_new_grid=True, vis: VisUtils = None):
        show_vis = False
        if vis is None:
            vis = VisUtils()
            show_vis = True
        vis.plot(self.mesh.grid, style='wireframe', color=(1., 0., 0.))
        if disp is not None:
            if disp.shape[1] == 2:
                disp = np.concatenate([disp, np.zeros(shape=(disp.shape[0], 1))], axis=1)
            vis.plot_grad(self.mesh.grid.points, disp, mag=1.0, color=(0., 0., 1.), opacity=0.5)

            if with_new_grid:
                deformed_grid = self.mesh.grid.copy()
                deformed_grid.points += disp
                vis.plot(deformed_grid, style='wireframe', color=(0., 1., 0.))
        if show_vis:
            vis.show()


class GradSpringTorchFusion(GradRawFusion):
    def __init__(self, mesh: SpringTorchMeshObj, obs_pcd: np.ndarray = None, obs_r=0.0):
        super().__init__(mesh=mesh)
        self.with_obs = obs_pcd is not None
        self.obs_pcd = obs_pcd
        self.obs_r = obs_r
        self.run_cfg: Dict[str, Union[float, np.ndarray, torch.Tensor]] = {}

    @property
    def style(self):
        return 'GradSpringTorchFusion'

    def cal_guide_force(self, max_step: float):
        assert 1.0 > max_step >= 1e-2
        self.run_cfg.update({'max_scale': max_step})
        grad_agg = 0.0
        for _, grad in self.grad_dict.items():
            grad_agg += grad[:, :self.mesh.topology_dim]
        grad_agg = grad_agg / float(len(self.grad_dict))

        grad_norm = np.linalg.norm(grad_agg, axis=1, ord=2)
        grad_agg = grad_agg / np.max(grad_norm) * max_step
        force_guide = grad_agg * self.mesh.stiff_cof
        return force_guide

    def cal_obs_force(self, field_weight: float):
        if self.with_obs:
            conf_mesh_idxs, conf_obs_idxs = self.mesh.detect_obstacle(
                obs_pcd=self.obs_pcd,
                obs_radius=self.obs_r,
                unit_lr=0.0
            )
            if conf_mesh_idxs.shape[0] > 0:
                vec_raw = self.mesh.points[conf_mesh_idxs, :self.mesh.topology_dim] - \
                          self.obs_pcd[conf_obs_idxs, :self.mesh.topology_dim]
                vec_len = np.linalg.norm(vec_raw, ord=2, axis=1)
                vec_norm = vec_raw / vec_len[:, np.newaxis]
                force_obs = (self.obs_r - vec_len) * field_weight * vec_norm
                force_inv = np.zeros(shape=(self.mesh.num_points, self.mesh.topology_dim))
                for conf_mesh_idx in np.unique(conf_mesh_idxs):
                    inv_agg = np.sum(force_obs[conf_mesh_idxs == conf_mesh_idx], axis=0)
                    force_inv[conf_mesh_idx, :] = inv_agg
                return {
                    'is_conflict': True, 'conf_mesh_idxs': conf_mesh_idxs,
                    'conf_obs_idxs': conf_obs_idxs, 'force_inv': force_inv
                }
        return {'is_conflict': False}

    def solve_disp(self, lr=2e-6, max_step=0.1, field_weight=10.0, max_iter=10000):
        force_guide = self.cal_guide_force(max_step)
        force_internal = self.mesh.cal_internal_force(with_internal=True, with_bc=True) * -1.0
        force_agg = force_internal + force_guide
        # 指引力只保留边界部分
        force_agg[self.mesh.bc_fix_dofs, :] = 0.0
        force_agg[self.mesh.internal_dofs, :] = 0.0
        inv_cfg = self.cal_obs_force(field_weight=field_weight)
        if inv_cfg['is_conflict']:
            force_agg += inv_cfg['force_inv']
        self.mesh.assemble_problem(force=force_agg)
        disp = self.mesh.fit_disp(
            min_lr_ratio=0.05, max_iter=1000,
            loss_weight={
                'loss_fix': 1000.0,  # loss_fix=1.0，不动点移动距离为1e-4
                'loss_force_bc': 100.0,  # loss_force=1.0，力的移动距离均衡残差1e-4
                'loss_force_internal': 100.0,  # loss_force=1.0，力的移动距离均衡残差1e-4
            }
        )
        return disp

    def draw_problem(
            self,
            disp: np.ndarray = None, with_new_grid=True,
            with_force_internal=False, with_obs=True, vis: VisUtils = None
    ):
        show_vis = False
        if vis is None:
            vis = VisUtils()
            show_vis = True
        super().draw_problem(disp=disp, with_new_grid=with_new_grid, vis=vis)
        if with_obs:
            vis.plot((pyvista.PointSet(self.obs_pcd)))
        if with_force_internal:
            force_internal = self.mesh.cal_internal_force()
            if force_internal.shape[1] == 2:
                force_internal = np.concatenate([force_internal, np.zeros(shape=(force_internal.shape[0], 1))], axis=1)
            vis.plot_grad(self.mesh.points, force_internal, mag=1.0, color=(1., 0., 0.), opacity=0.5)
        if show_vis:
            vis.show()


class GradSpringFeaFusion(GradRawFusion):
    def __init__(self, mesh: SpringFeaMeshObj, obs_pcd: np.ndarray = None, obs_r=0.0):
        super().__init__(mesh=mesh)
        self.with_obs = obs_pcd is not None
        self.obs_pcd = obs_pcd
        self.obs_r = obs_r
        self.run_cfg: Dict[str, Union[float, np.ndarray, torch.Tensor]] = {}

    @property
    def style(self):
        return 'GradSpringFeaFusion'

    def cal_obs_var(self, max_step: float, var_agg: np.ndarray):
        if self.with_obs:
            conf_mesh_idxs, conf_obs_idxs = self.mesh.detect_obstacle(
                obs_pcd=self.obs_pcd, obs_radius=self.obs_r, unit_lr=max_step
            )
            if conf_mesh_idxs.shape[0] > 0:
                vec_raw = self.mesh.points[conf_mesh_idxs, :self.mesh.topology_dim] - \
                          self.obs_pcd[conf_obs_idxs, :self.mesh.topology_dim]
                vec_len = np.linalg.norm(vec_raw, ord=2, axis=1)
                vec_norm = vec_raw / vec_len[:, np.newaxis]

                # ------ method 1
                # 当vec_len=self.obs_r时，反向距离恰好与最大最大距离相斥为0，但这样是不够，除了抵消正向力还需要提供反向力
                # scale_offset = np.linalg.norm(var_agg[conf_mesh_idxs, :], ord=2, axis=1)
                # scale_inv = (max_step - (vec_len - self.obs_r)) * 1.0
                # obs_var = vec_norm * (scale_offset + scale_inv)[:, np.newaxis]

                # ------ method 2
                # cos_length = np.abs(np.minimum(np.sum(var_agg[conf_mesh_idxs, :] * vec_norm, axis=1), 0.0))
                # vec_offset = var_agg[conf_mesh_idxs, :] + cos_length[:, np.newaxis] * vec_norm
                # scale_inv = (max_step - (vec_len - self.obs_r)) * 1.0
                # obs_var = vec_norm * scale_inv[:, np.newaxis] + vec_offset

                # ------ method 3
                scale_inv = (max_step - (vec_len - self.obs_r)) * 1.0
                obs_var = vec_norm * scale_inv[:, np.newaxis]

                var_inv = np.zeros(shape=(self.mesh.num_points, self.mesh.topology_dim))
                for conf_mesh_idx in np.unique(conf_mesh_idxs):
                    var_agg = np.sum(obs_var[conf_mesh_idxs == conf_mesh_idx], axis=0)
                    var_inv[conf_mesh_idx, :] = var_agg / np.linalg.norm(var_agg, ord=2) * max_step

                return {
                    'is_conflict': True, 'conf_mesh_idxs': conf_mesh_idxs,
                    'conf_obs_idxs': conf_obs_idxs, 'var_inv': var_inv
                }
        return {'is_conflict': False}

    def cal_guide_var(self, max_step):
        self.run_cfg.update({'max_step': max_step})
        grad_agg = 0.0
        for _, grad in self.grad_dict.items():
            grad_agg += grad[:, :self.mesh.topology_dim]
        grad_agg = grad_agg / float(len(self.grad_dict))
        grad_norm = np.linalg.norm(grad_agg, axis=1, ord=2)
        grad_agg = grad_agg / np.max(grad_norm) * max_step
        return grad_agg

    def solve_disp(
            self,
            method: Literal['linear_elastic', 'poisson_eq', 'linear_elastic_obs'],
            kwargs: Dict,
            max_step: float,
            with_debug=False,
    ):
        var_guide = self.cal_guide_var(max_step)
        inv_cfg = self.cal_obs_var(max_step=max_step, var_agg=var_guide)
        if method == 'linear_elastic':
            var_agg = var_guide + inv_cfg['var_inv'] if inv_cfg['is_conflict'] else var_guide
            var_agg[self.mesh.bc_fix_dofs, :] = 0.0
            var_agg[self.mesh.internal_dofs, :] = 0.0
            force_agg = var_agg * kwargs['stiff_cof']
            kwargs.update({'force': force_agg})
            is_valid, disp = self.mesh.fit_disp(method=method, with_debug=with_debug, kwargs=kwargs)
            disp_norm = np.linalg.norm(disp, axis=1, ord=2)
            disp = disp / np.max(disp_norm) * max_step

        elif method == 'poisson_eq':
            var_agg = var_guide + inv_cfg['var_inv'] if inv_cfg['is_conflict'] else var_guide
            var_agg[self.mesh.bc_fix_dofs, :] = 0.0
            var_agg[self.mesh.internal_dofs, :] = 0.0
            kwargs.update({'displacement': var_agg})
            is_valid, disp = self.mesh.fit_disp(method=method, with_debug=with_debug, kwargs=kwargs)

        elif method == 'linear_elastic_obs':
            # todo how to find suitable stiff_cof
            is_valid = True
            if inv_cfg['is_conflict']:
                force_inv = inv_cfg['var_inv'] * kwargs['stiff_cof']
                force_inv[self.mesh.internal_dofs, :] = 0.0
                kwargs.update({'obs_force': force_inv})
                is_valid, var_inv = self.mesh.fit_disp(method=method, with_debug=with_debug, kwargs=kwargs)
                if is_valid:
                    disp = var_guide + var_inv
                    disp = disp / np.max(np.linalg.norm(disp)) * max_step
                else:
                    disp = None
            else:
                disp = var_guide

        else:
            raise ValueError(f'[Error]: Non-valid method {method}.')

        # # ------ debug vis
        # if inv_cfg['is_conflict']:
        #     self.draw_problem(
        #         disp=disp,
        #         var_guide=var_guide,
        #         # var_agg=var_agg,
        #         var_inv=var_inv,
        #         with_obs=True,
        #         inv_cfg=inv_cfg,
        #         with_new_grid=False
        #     )
        # # ------

        return is_valid, disp

    def draw_problem(
            self,
            disp: np.ndarray = None, with_new_grid=True,
            var_guide: np.ndarray = None, var_inv: np.ndarray = None, var_agg: np.ndarray = None,
            with_obs=True, inv_cfg: Dict = None, vis: VisUtils = None
    ):
        show_vis = False
        if vis is None:
            vis = VisUtils()
            show_vis = True
        super().draw_problem(disp=disp, with_new_grid=with_new_grid, vis=vis)
        if var_guide is not None:
            if var_guide.shape[1] == 2:
                var_guide = np.concatenate([var_guide, np.zeros(shape=(var_guide.shape[0], 1))], axis=1)
            vis.plot_grad(self.mesh.grid.points, var_guide, mag=1.0, color=(1., 0., 0.), opacity=0.5)
        if var_agg is not None:
            if var_agg.shape[1] == 2:
                var_agg = np.concatenate([var_agg, np.zeros(shape=(var_agg.shape[0], 1))], axis=1)
            vis.plot_grad(self.mesh.grid.points, var_agg, mag=1.0, color=(1., 0., 0.), opacity=0.5)
        if with_obs:
            vis.plot(pyvista.PointSet(self.obs_pcd))
            if inv_cfg is not None:
                if inv_cfg['is_conflict']:
                    vis.plot(pyvista.PointSet(self.obs_pcd[inv_cfg['conf_obs_idxs'], :]), color=(0., 1., 1.))
                    vis.plot(pyvista.PointSet(self.mesh.points[inv_cfg['conf_mesh_idxs'], :]), color=(0., 1., 1.))
        if var_inv is not None:
            if var_inv.shape[1] == 2:
                var_inv = np.concatenate([var_inv, np.zeros(shape=(var_inv.shape[0], 1))], axis=1)
            vis.plot_grad(self.mesh.grid.points, var_inv, mag=1.0, color=(0., 1., 1.))
        if show_vis:
            vis.show()
