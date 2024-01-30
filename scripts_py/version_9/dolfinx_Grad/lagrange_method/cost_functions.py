import dolfinx
import numpy as np
import ufl
from typing import Union, Tuple, List, Set

from ..dolfinx_utils import AssembleUtils


class IntegralFunction(object):
    def __init__(self, form: ufl.Form):
        self.form = form

    def evaluate(self):
        val: float = AssembleUtils.assemble_scalar(dolfinx.fem.form(self.form))
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

    def scale(self, scaling_factor: float):
        self.form = scaling_factor * self.form

    def update(self):
        pass


class ScalarTrackingFunctional(object):
    def __init__(
            self,
            domain: dolfinx.mesh.Mesh,
            integrand_form: ufl.Form,
            tracking_goal: float
    ):
        self.domain = domain
        self.integrand_form = integrand_form
        self.integrand_dolfinx = dolfinx.fem.form(self.integrand_form)
        self.tracking_goal = tracking_goal

        self.integrand_value = dolfinx.fem.Constant(domain, 0.0)
        self.goal_value = dolfinx.fem.Constant(domain, tracking_goal)
        self.derivative_form = (self.integrand_value - self.goal_value) * self.integrand_form

    def evaluate(self):
        val: float = np.power(AssembleUtils.assemble_scalar(self.integrand_dolfinx) - self.tracking_goal, 2) * 0.5
        return val

    def coefficients(self):
        coeffs: Tuple[dolfinx.fem.Function] = self.integrand_form.coefficients()
        return coeffs

    def derivative(
            self,
            argument: [ufl.Coefficient],
            direction: Union[dolfinx.fem.Function, ufl.Argument],
    ):
        derivative = ufl.derivative(self.derivative_form, argument, direction)
        return derivative

    def update(self):
        val: float = AssembleUtils.assemble_scalar(self.integrand_dolfinx)
        self.integrand_value.value = val


# ------ Define Lagrangian Functional
CostFunctional_types = Union[
    IntegralFunction, ScalarTrackingFunctional
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
