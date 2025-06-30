import dolfinx
import numpy as np
import ufl
import ctypes
from typing import Union, Tuple, List, Set, Literal

from scripts_py.version_9.dolfinx_grad.dolfinx_utils import AssembleUtils


class IntegralFunctional(object):
    def __init__(
            self,
            name: str,
            domain: dolfinx.mesh.Mesh,
            form: ufl.Form,
            weight: float = 1.0  # todo 后期改为场函数
    ):
        self.name = name
        self.domain = domain
        self.weight = dolfinx.fem.Constant(self.domain, weight)
        self.form = self.weight * form
        self.form_dolfin = dolfinx.fem.form(self.form)

    def evaluate(self):
        return AssembleUtils.assemble_scalar(self.form_dolfin)

    def derivative(
            self,
            argument: [ufl.Coefficient],
            direction: Union[dolfinx.fem.Function, ufl.Argument]
    ):
        return ufl.derivative(self.form, argument, direction)

    def coefficients(self):
        coeffs: Tuple[dolfinx.fem.Function] = self.form.coefficients()
        return coeffs

    def update(self):
        pass

    def update_scale(self, weight: float):
        self.weight.value = weight


class ScalarTrackingFunctional(object):
    def __init__(
            self,
            name: str,
            domain: dolfinx.mesh.Mesh,
            form: ufl.Form,
            tracking_goal: float,
            weight: float = 1.0,
            method: Literal['absolute_sub', 'percentage_div'] = 'absolute_sub'
    ):
        self.name = name
        self.domain = domain
        self.method = method
        self.weight = dolfinx.fem.Constant(self.domain, weight)

        self.form = form
        self.form_dolfin = dolfinx.fem.form(self.form)

        self.tracking_goal: float = tracking_goal
        # self.integrand_value = dolfinx.fem.Constant(domain, 0.0)
        # self.goal_value = dolfinx.fem.Constant(domain, self.tracking_goal_value)
        self.dif_value = dolfinx.fem.Constant(domain, 0.0)

    def evaluate(self):
        # self.goal_value.value = self.tracking_goal
        cur_value = AssembleUtils.assemble_scalar(self.form_dolfin)
        if self.method == 'absolute_sub':
            val = 0.5 * np.power(cur_value - self.tracking_goal, 2)
        elif self.method == 'percentage_div':
            val = np.sqrt(np.power(cur_value / self.tracking_goal - 1.0, 2.0))
        else:
            raise ValueError
        return val * self.weight.value

    def coefficients(self):
        coeffs: Tuple[dolfinx.fem.Function] = self.form.coefficients()
        return coeffs

    def derivative(
            self,
            argument: [ufl.Coefficient],
            direction: Union[dolfinx.fem.Function, ufl.Argument],
    ):
        # self.goal_value.value = self.tracking_goal
        derivative = ufl.derivative(self.weight * self.dif_value * self.form, argument, direction)
        # derivative = ufl.derivative(
        #     self.weight * (self.integrand_value - self.goal_value) * self.form, argument, direction
        # )
        return derivative

    def update(self):
        cur_val: float = AssembleUtils.assemble_scalar(self.form_dolfin)
        # self.integrand_value.value = val
        if self.method == 'absolute_sub':
            val = cur_val - self.tracking_goal
        elif self.method == 'percentage_div':
            val = cur_val / self.tracking_goal - 1.0
        else:
            raise ValueError
        self.dif_value.value = val

    def update_scale(self, weight: float):
        self.weight.value = weight


class MinMaxFunctional(object):
    def __init__(
            self,
            name: str,
            domain: dolfinx.mesh.Mesh,
            form: ufl.Form,
            lower_bound: float = None,
            upper_bound: float = None,
            weight: float = 1.0
    ):
        self.name = name
        self.domain = domain
        self.weight = dolfinx.fem.Constant(self.domain, weight)

        self.form = form
        self.form_dolfin = dolfinx.fem.form(self.form)

        self.bound_dif_form: ufl.Form = 0.0

        self.lower_bound = lower_bound
        if self.lower_bound is not None:
            self.lower_bound_dif = dolfinx.fem.Constant(domain, 0.0)
            self.bound_dif_form += self.lower_bound_dif

        self.upper_bound = upper_bound
        if self.upper_bound is not None:
            self.upper_bound_dif = dolfinx.fem.Constant(domain, 0.0)
            self.bound_dif_form += self.upper_bound_dif

    def evaluate(self):
        cur_val = AssembleUtils.assemble_scalar(self.form_dolfin)
        cost = 0.0
        if self.lower_bound is not None:
            cost += 0.5 * np.power(np.minimum(0.0, cur_val - self.lower_bound), 2)
        if self.upper_bound is not None:
            cost += 0.5 * np.power(np.maximum(0.0, cur_val - self.upper_bound), 2)
        return self.weight.value * cost

    def coefficients(self):
        coeffs: Tuple[dolfinx.fem.Function] = self.form.coefficients()
        return coeffs

    def derivative(
            self,
            argument: [ufl.Coefficient],
            direction: Union[dolfinx.fem.Function, ufl.Argument],
    ):
        derivative = ufl.derivative(self.weight * self.bound_dif_form * self.form, argument, direction)
        return derivative

    def update(self):
        cur_val = AssembleUtils.assemble_scalar(self.form_dolfin)

        if self.lower_bound is not None:
            cost = np.power(np.minimum(0.0, cur_val - self.lower_bound), 2) * 0.5
            self.lower_bound_dif.value = cost

        if self.upper_bound is not None:
            cost = np.power(np.maximum(0.0, cur_val - self.upper_bound), 2) * 0.5
            self.upper_bound_dif.value = cost

    def update_scale(self, weight: float):
        self.weight.value = weight


# ------ Define Lagrangian Functional
class LagrangianFunction(object):
    def __init__(
            self,
            cost_functional_list: List[Union[IntegralFunctional, ScalarTrackingFunctional, MinMaxFunctional]],
            state_forms: List[ufl.Form],
    ):
        """
        lagrangian乘数法：此处state_form(状态方程)为弱形式，因此TestFunction作为拉格朗日乘子
        lagrangian_func = cost_functional_list.sum() + summed_state_forms
        """
        self.cost_functional_list = cost_functional_list
        self.state_forms = state_forms
        self.summed_state_forms = self.state_forms[0]
        for form in self.state_forms[1:]:
            self.summed_state_forms += form

    def derivative(self, argument: ufl.core.expr.Expr, direction: ufl.core.expr.Expr) -> ufl.Form:
        cost_derivative = self.cost_functional_list[0].derivative(argument, direction)
        for functional in self.cost_functional_list[1:]:
            cost_derivative += functional.derivative(argument, direction)

        state_forms_derivative = ufl.derivative(self.summed_state_forms, argument, direction)
        derivative = cost_derivative + state_forms_derivative

        return derivative

    def coefficients(self) -> Set[dolfinx.fem.Function]:
        state_coeffs = set(self.summed_state_forms.coefficients())
        functional_coeffs = [set(functional.coefficients()) for functional in self.cost_functional_list]
        coeffs = set().union(*functional_coeffs)
        coeffs.union(state_coeffs)
        return coeffs
