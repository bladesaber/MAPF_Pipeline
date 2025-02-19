import numpy as np
from functools import partial
from scipy.optimize import minimize
from typing import Callable, Dict, Union, Tuple, Literal
import pyvista


class FixConstraintGradScipy(object):
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
