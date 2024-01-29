import dolfinx
import numpy
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


# ------ Define Lagrangian Functional
CostFunctional_types = Union[
    IntegralFunction,
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
