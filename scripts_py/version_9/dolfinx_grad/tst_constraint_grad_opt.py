import numpy as np
from functools import partial
from scipy.optimize import minimize
from typing import Callable, Dict, Union, Tuple, Literal
import torch
import pyvista
from tqdm import tqdm

from scripts_py.version_9.dolfinx_grad.tst_shape_ffd_opt_fluid import ShapeFfdFluidOptimizer
from scripts_py.version_9.mapf_pkg.torch_utils import TorchUtils
from scripts_py.version_9.dolfinx_grad.deform_geometry import AreaBspline, VolumeBspline


class ConstraintGradScipy(object):
    def __init__(
            self,
            fix_var_dofs: np.ndarray, fix_var: np.ndarray,
            current_cof: np.ndarray, target_cof_grad: np.ndarray, compute_var: Callable,
            grad_norm_scale=0.5
    ):
        self.fix_var_dofs = fix_var_dofs
        self.fix_var = fix_var

        self.target_cof_grad = target_cof_grad / np.linalg.norm(target_cof_grad) * grad_norm_scale
        # ------ 解耦方向会由于部分梯度为0而出现问题
        # self.cof_scale = np.linalg.norm(target_cof_grad, ord=2, axis=-1, keepdims=True) + 1e-8
        # self.cof_vec = target_cof_grad / self.cof_scale

        self.compute_var = partial(
            self.var_func,
            current_cof=current_cof,
            compute_var=compute_var
        )
        self.compute_loss = partial(
            self.loss_func,
            target_x=self.target_cof_grad
        )
        self.compute_constraint = partial(
            self.constraint_func,
            fix_var_dofs=self.fix_var_dofs,
            fix_var=self.fix_var,
            compute_var=self.compute_var
        )

    @staticmethod
    def var_func(x: np.ndarray, current_cof: np.ndarray, compute_var: Callable):
        x = x.reshape(current_cof.shape)
        return compute_var(current_cof + x)

    @staticmethod
    def loss_func(x: np.ndarray, target_x: np.ndarray):
        x = x.reshape(target_x.shape)
        return (np.sqrt((np.power(x - target_x, 2.0)))).mean()

    @staticmethod
    def constraint_func(
            x: np.ndarray,
            fix_var_dofs: np.ndarray, fix_var: np.ndarray,
            compute_var: Callable
    ):
        var_new = compute_var(x)
        dif = var_new[fix_var_dofs, :] - fix_var
        return (np.sum(np.power(dif, 2.0), axis=1)).flatten()

    def run(self, constraint_tol=1e-4):
        x0 = self.target_cof_grad.flatten()

        constraint = {'type': 'eq', 'fun': self.compute_constraint}
        # res = minimize(self.compute_loss, x0, constraints=[constraint], method='SLSQP')
        res = minimize(self.compute_loss, x0, constraints=[constraint], method='trust-constr')

        constraint_loss = np.max(np.abs(self.compute_constraint(res.x)))
        print(f'[Info]: fun_loss:{res.fun} constraint_loss:{constraint_loss}.')

        x = res.x.reshape(self.target_cof_grad.shape)
        return {'is_solved': constraint_loss < constraint_tol, 'res': x}


class ConstraintGradOptimizer(object):
    def __init__(self):
        self.shape_dict: Dict[str, Dict[str, ShapeFfdFluidOptimizer]] = {}

    def add_model(self, name, model: ShapeFfdFluidOptimizer, deform_scale: float):
        self.shape_dict[name] = {
            'model': model,
            'fix_dofs': model.mesh.fix_dofs,
            'fix_var': TorchUtils.np2tensor(model.mesh.fix_var, require_grad=False),
            'var_func': AreaBspline.get_pcd_torch if model.mesh.topology_dim == 2 else VolumeBspline.get_pcd_torch,
            'mat': TorchUtils.np2tensor(model.mesh.ffd_obj.mat, require_grad=False),
            'deform_scale': deform_scale
        }

    def assemble_problem(self):
        for name, cfg in self.shape_dict.items():
            model: ShapeFfdFluidOptimizer = cfg['model']
            var_grad = model.var_grad
            var_grad = var_grad / np.max(np.abs(var_grad)) * cfg['deform_scale']  # var_grad的分布是很极端的

            x = var_grad + np.random.normal(loc=0.0, scale=1e-2, size=var_grad.shape)  # 不能与y完全相等,否则会有nan梯度
            y = var_grad

            cfg.update({
                'x_grad': TorchUtils.np2tensor(x, require_grad=True),
                'y': TorchUtils.np2tensor(y, require_grad=False),
                'cur_x': TorchUtils.np2tensor(model.mesh.ffd_obj.coef, require_grad=False),
            })

    def compute_loss(self, logger: Dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
        loss_grad, loss_fix = 0.0, 0.0
        for name, cfg in self.shape_dict.items():
            shape_logger = None
            if logger is not None:
                shape_logger = {}

            loss_grad += self.compute_loss_grad(
                x=cfg['x_grad'], y=cfg['y'],
                weight=1.0, logger=shape_logger
            )
            loss_fix += self.compute_loss_fix(
                x_grad=cfg['x_grad'],
                cur_x=cfg['cur_x'], mat=cfg['mat'],
                fix_var_dofs=cfg['fix_dofs'], fix_var=cfg['fix_var'], var_func=cfg['var_func'],
                weight=1e4, logger=shape_logger
            )

            logger[name] = shape_logger
        return loss_grad, loss_fix

    # @staticmethod
    # def compute_loss_grad(
    #         x: torch.Tensor, y: torch.Tensor,
    #         weight: float, logger: Dict = None
    # ):
    #     x_scale = torch.norm(x, p=2, dim=-1, keepdim=True)
    #     x_vec = x / (x_scale + 1e-8)  # 防止零除
    #     y_scale = torch.norm(y, p=2, dim=-1, keepdim=True)
    #     y_vec = y / (y_scale + 1e-8)  # 防止零除
    #
    #     cos_value = torch.sum(x_vec * y_vec, dim=-1).reshape(-1)
    #     # cos_value = (torch.sum(x_vec * y_vec, dim=-1) * relu_layer).reshape(-1)
    #
    #     loss_cos = 1.0 - cos_value
    #     loss_cos = (torch.log(loss_cos) - np.log(1e-8)) / np.log(2.0)
    #     loss_cos = torch.mean(loss_cos) + torch.max(loss_cos)
    #
    #     # scale_dif = (x_scale / y_scale - 1.0).reshape(-1)
    #     # loss_scale = torch.sqrt(torch.pow(scale_dif, 2.0))  # 由于y_scale太小会过于敏感
    #
    #     scale_dif = (x_scale - y_scale).reshape(-1)
    #     # scale_dif = ((x_scale - y_scale) * relu_layer).reshape(-1)
    #
    #     loss_scale = torch.sqrt(torch.pow(scale_dif, 2.0))
    #     loss_scale = torch.mean(loss_scale) + torch.max(loss_scale)
    #
    #     loss = (loss_cos + loss_scale) * weight
    #     if logger is not None:
    #         logger.update({'loss_cos': loss_cos.detach().numpy(), 'loss_scale': loss_scale.detach().numpy()})
    #     return loss

    @staticmethod
    def compute_loss_grad(
            x: torch.Tensor, y: torch.Tensor,
            weight: float, logger: Dict = None
    ):
        loss = torch.sqrt(torch.pow(x - y, 2.0))
        loss = (torch.mean(loss) + torch.max(loss)) * weight
        if logger is not None:
            logger.update({'loss_grad': loss.detach().numpy()})
        return loss

    @staticmethod
    def compute_loss_fix(
            x_grad: torch.Tensor,
            cur_x: torch.Tensor, mat: torch.Tensor,
            fix_var_dofs: np.ndarray, fix_var: torch.Tensor,
            var_func: Callable,
            weight,
            logger: Dict = None
    ):
        var_new = var_func(mat, cur_x + x_grad)
        loss = torch.sum(torch.pow(var_new[fix_var_dofs, :] - fix_var, 2.0), dim=1)
        loss = torch.sum(loss) * weight
        if logger is not None:
            logger.update({'loss_fix': loss.detach().numpy()})
        return loss

    def run(
            self,
            max_shape_iter=200,
            max_deformation=0.05,
            min_deformation=0.01,
            state_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            adjoint_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            gradient_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            lr_grad=0.1, min_lr_grad=0.01, max_grad_iter=1000,
            with_debug=False
    ):
        step, min_loss = 0, np.inf
        while True:
            loss_sum = 0.0
            for _, cfg in self.shape_dict.items():
                model: ShapeFfdFluidOptimizer = cfg['model']
                is_solved, grads_dict = model.compute_gradient(
                    state_mat_assemble_method=state_mat_assemble_method,
                    adjoint_mat_assemble_method=adjoint_mat_assemble_method,
                    gradient_mat_assemble_method=gradient_mat_assemble_method,
                    with_debug=with_debug
                )
                if not is_solved:
                    return False
                model.fuse_grad(grads_dict)
                model.log(step)

                loss_dict = model.evaluate_cost(update_state=False)
                loss_sum += np.sum(list(loss_dict.values()))
                model.log(step)
            print(f"[Debug]: Step_{step} loss:{loss_sum:.3f}")

            self.assemble_problem()
            self.refit_grad(lr=lr_grad, min_lr=min_lr_grad, max_iter=max_grad_iter)

            for _, cfg in self.shape_dict.items():
                model: ShapeFfdFluidOptimizer = cfg['model']
                is_valid = model.update_control(
                    min_deform=min_deformation,
                    max_deform=max_deformation,
                    line_search_step=5,
                )
                if not is_valid:
                    return False

            step += 1
            if step > max_shape_iter:
                break

            min_loss = np.minimum(min_loss, loss_sum)
            if loss_sum > min_loss * 1.05:
                break

        loss_sum = 0.0
        for _, cfg in self.shape_dict.items():
            model: ShapeFfdFluidOptimizer = cfg['model']
            loss_dict = model.evaluate_cost(update_state=False)
            loss_sum += np.sum(list(loss_dict.values()))
            model.log(step)

        print(f"[Debug]: Step_{step} loss:{loss_sum:.3f}")
        return True

    def refit_grad(self, lr=0.1, min_lr=0.01, max_iter=1000):
        debug_bar = tqdm()
        optimizer = TorchUtils.get_optimizer([cfg['x_grad'] for _, cfg in self.shape_dict.items()], lr=lr)
        scheduler = TorchUtils.get_lr_scheduler(optimizer, factor=0.25, patience=100, cooldown=100, min_lr=min_lr)
        iteration = -1
        while True:
            iteration += 1
            logger: Dict[str, Dict[str, float]] = {}

            loss_grad, loss_fix = self.compute_loss(logger=logger)
            loss = loss_grad + loss_fix

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            # ------ debug vis
            desc = f'[iter:{iteration}/lr:{scheduler.get_last_lr()[-1]:.5f}]'
            for name, sub_logger in logger.items():
                sub_desc = f' {name}:'
                for cost_name, cost_value in sub_logger.items():
                    sub_desc += f'{cost_name}:{cost_value:.3f} '
                desc += sub_desc
            debug_bar.set_description(desc)
            debug_bar.update(iteration)
            # print(desc)
            # ------

            if iteration > max_iter:
                break

            if lr <= min_lr:
                break

            # if iteration > 10:
            #     raise ValueError

        for name, cfg in self.shape_dict.items():
            model: ShapeFfdFluidOptimizer = cfg['model']
            model.var_grad = TorchUtils.tensor2np(cfg['x_grad'])

    def update_control(self, grad_dict: Dict[str, np.ndarray]):
        for name, displacement in grad_dict.items():
            model: ShapeFfdFluidOptimizer = self.shape_dict[name]['model']
            model.update_control()


def debug():
    optim_shape = ShapeFfdFluidOptimizer(
        mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example03/model.msh',
        dim=2,
        result_dir='/home/admin123456/Desktop/work/topopt_exps/new_test/example03/cache',
        ffd_num=[30, 30], ffd_degree=[3, 3]
    )
    optim_shape.add_regularization(
        reg_type='volume_regularization',
        target_volume_rho=1.0,
        weight=2000.0
    )
    optim_shape.define_control_wall({
        'inlet': {'marker': 13},
        'outlet': {'marker': 14},
        'wallFixed': {'marker': 15},
    })
    # optim.mesh.draw_mesh()

    optim_shape.add_problem(
        problem_name='cond1',
        inlet_cfg={
            'inlet': {
                'marker': 13,
                'value': [3., 0., 0.]
            }
        },
        outlet_cfg={
            'outlet': {
                'marker': 14,
                'value': 0.0
            }
        },
        wall_cfg={
            'wallFixed': {
                'marker': 15
            },
            'wallFree': {
                'marker': 16
            }
        },
        constant_nu=0.01,
        fluid_type='stoke',
        loss_cfgs=[
            {
                'style': 'minimum_dissipation',
                'weight': 1.0
            }
        ],
        su2_cfg={
            'run_iter': 1000,
            'constant_viscosity': 0.01,
            'refit_by_fenics': False
        }
    )

    optim = ConstraintGradOptimizer()
    optim.add_model(
        name='test', model=optim_shape, deform_scale=0.1
    )
    optim.run(
        max_shape_iter=200,
        max_deformation=0.1,
        min_deformation=0.01,
        state_mat_assemble_method='lift',
        adjoint_mat_assemble_method='lift',
        gradient_mat_assemble_method='Identity_row',  # 必须使用Identity_row否则边界条件不正确
        with_debug=False,
        lr_grad=0.02, min_lr_grad=0.001,
        max_grad_iter=1000
    )


if __name__ == '__main__':
    debug()
