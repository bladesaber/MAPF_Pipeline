import dolfinx
import numpy as np
import ufl
import ctypes
from typing import Union, Tuple, List, Set

from ..dolfinx_utils import AssembleUtils


class IntegralFunction(object):
    def __init__(
            self,
            domain: dolfinx.mesh.Mesh,
            form: ufl.Form,
            weight: float = 1.0,
            name: str = 'IntegralFunction'
    ):
        self.name = name
        self.domain = domain
        self.weight_value = weight
        self.weight = dolfinx.fem.Constant(self.domain, weight)

        self.orig_form = form
        self.form = self.weight * self.orig_form
        self.form_dolfin = dolfinx.fem.form(self.form)

    def evaluate(self):
        val: float = AssembleUtils.assemble_scalar(self.form_dolfin)
        return val

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
        self.weight_value = weight
        self.weight.value = weight


class ScalarTrackingFunctional(object):
    def __init__(
            self,
            domain: dolfinx.mesh.Mesh,
            integrand_form: ufl.Form,
            tracking_goal: Union[float, ctypes.c_float, ctypes.c_double],
            weight: float = 1.0,
            name: str = 'ScalarTracking'
    ):
        self.name = name
        self.domain = domain
        self.weight_value = weight
        self.weight = dolfinx.fem.Constant(self.domain, weight)

        self.form_orig = integrand_form
        self.form = self.form_orig
        self.form_dolfin = dolfinx.fem.form(self.form)

        self.tracking_goal = tracking_goal
        if isinstance(tracking_goal, (ctypes.c_float, ctypes.c_double)):
            self.tracking_goal_value = tracking_goal.value
        else:
            self.tracking_goal_value = tracking_goal

        self.integrand_value = dolfinx.fem.Constant(domain, 0.0)
        self.goal_value = dolfinx.fem.Constant(domain, self.tracking_goal_value)
        self.derivative_form = self.weight * (self.integrand_value - self.goal_value) * self.form

    def evaluate(self):
        if isinstance(self.tracking_goal, (ctypes.c_float, ctypes.c_double)):
            self.tracking_goal_value = self.tracking_goal.value
            self.goal_value.value = self.tracking_goal_value

        val: float = np.power(AssembleUtils.assemble_scalar(self.form_dolfin) - self.tracking_goal_value, 2) * 0.5
        val = val * self.weight_value
        return val

    def coefficients(self):
        coeffs: Tuple[dolfinx.fem.Function] = self.form.coefficients()
        return coeffs

    def derivative(
            self,
            argument: [ufl.Coefficient],
            direction: Union[dolfinx.fem.Function, ufl.Argument],
    ):
        if isinstance(self.tracking_goal, (ctypes.c_float, ctypes.c_double)):
            self.tracking_goal_value = self.tracking_goal.value
            self.goal_value.value = self.tracking_goal_value

        derivative = ufl.derivative(self.derivative_form, argument, direction)
        return derivative

    def update(self):
        val: float = AssembleUtils.assemble_scalar(self.form_dolfin)
        self.integrand_value.value = val

    def update_scale(self, weight: float):
        self.weight_value = weight
        self.weight.value = weight


class MinMaxFunctional(object):
    def __init__(
            self,
            domain: dolfinx.mesh.Mesh,
            integrand_form: ufl.Form,
            lower_bound: float = None,
            upper_bound: float = None,
            weight: float = 1.0,
            name: str = 'MinMax'
    ):
        assert (lower_bound is not None) or (upper_bound is not None)

        self.name = name
        self.domain = domain
        self.weight_value = weight
        self.weight = dolfinx.fem.Constant(self.domain, weight)

        self.form_orig = integrand_form
        self.form = self.form_orig
        self.form_dolfin = dolfinx.fem.form(self.form)

        bound_dif_form: ufl.Form = 0.0

        self.lower_bound = lower_bound
        if self.lower_bound is not None:
            self.lower_bound_dif = dolfinx.fem.Constant(domain, 0.0)
            bound_dif_form += self.lower_bound_dif

        self.upper_bound = upper_bound
        if self.upper_bound is not None:
            self.upper_bound_dif = dolfinx.fem.Constant(domain, 0.0)
            bound_dif_form += self.upper_bound_dif

        self.derivative_form = self.weight * bound_dif_form * self.form

    def evaluate(self):
        val = AssembleUtils.assemble_scalar(self.form_dolfin)

        cost = 0.0
        if self.lower_bound is not None:
            cost += np.power(np.minimum(0.0, val - self.lower_bound), 2) * 0.5

        if self.upper_bound is not None:
            cost += np.power(np.maximum(0.0, val - self.upper_bound), 2) * 0.5

        cost = self.weight_value * cost
        return cost

    def coefficients(self):
        coeffs: Tuple[dolfinx.fem.Function] = self.form.coefficients()
        return coeffs

    def derivative(
            self,
            argument: [ufl.Coefficient],
            direction: Union[dolfinx.fem.Function, ufl.Argument],
    ):
        derivative = ufl.derivative(self.derivative_form, argument, direction)
        return derivative

    def update(self):
        val: float = AssembleUtils.assemble_scalar(self.form_dolfin)
        if self.lower_bound is not None:
            cost = np.power(np.minimum(0.0, val - self.lower_bound), 2) * 0.5
            self.lower_bound_dif.value = cost

        if self.upper_bound is not None:
            cost = np.power(np.maximum(0.0, val - self.upper_bound), 2) * 0.5
            self.upper_bound_dif.value = cost

    def update_scale(self, weight: float):
        self.weight_value = weight
        self.weight.value = weight


# ------ Define Lagrangian Functional
CostFunctional_types = Union[
    IntegralFunction, ScalarTrackingFunctional, MinMaxFunctional
]


class LagrangianFunction(object):
    def __init__(
            self,
            cost_functional_list: List[CostFunctional_types],
            state_forms: List[ufl.Form],
    ):
        self.cost_functional_list = cost_functional_list
        self.state_forms = state_forms

        self.summed_state_forms = self.state_forms[0]
        for form in self.state_forms[1:]:
            self.summed_state_forms += form

    def derivative(self, argument: ufl.core.expr.Expr, direction: ufl.core.expr.Expr) -> ufl.Form:
        cost_functional_derivative = self.cost_functional_list[0].derivative(argument, direction)

        for functional in self.cost_functional_list[1:]:
            cost_functional_derivative += functional.derivative(argument, direction)

        state_forms_derivative = ufl.derivative(self.summed_state_forms, argument, direction)

        derivative = cost_functional_derivative + state_forms_derivative
        return derivative

    def coefficients(self) -> Set[dolfinx.fem.Function]:
        state_coeffs = set(self.summed_state_forms.coefficients())
        functional_coeffs = [set(functional.coefficients()) for functional in self.cost_functional_list]
        coeffs = set().union(*functional_coeffs)
        coeffs.union(state_coeffs)
        return coeffs
