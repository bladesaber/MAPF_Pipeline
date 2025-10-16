from typing import Set, Literal

import dolfinx
import ufl
import ufl.core.expr

import numpy as np

from scripts_py.dolfinx_utils import AssembleUtils


class CostFunctional(object):
    def __init__(self, name: str):
        self.name = name
    
    def derivative(
        self, argument: ufl.Coefficient, direction: dolfinx.fem.Function | ufl.Argument
    ) -> ufl.Form:
        raise NotImplementedError

    def coefficients(self) -> Set[dolfinx.fem.Function]:
        raise NotImplementedError

    def evaluate(self) -> float:
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def update_weight(self, weight: float):
        raise NotImplementedError


class LagrangianFunction(object):
    def __init__(
        self,
        cost_functional_list: list[CostFunctional],
        state_weak_forms: list[ufl.Form],
    ):
        """
        lagrangian multiplier method:
            state_form(state equation) is weak form comes from finite element method, so TestFunction is Lagrangian multiplier
        object_function J = cost_functional_list.sum()
        lagrangian multiplier * equation constraint = state_weak_forms.sum()
        be careful, state_weak_forms is weak form, so it already has TestFunction as Lagrangian multiplier
        """
        self.cost_functional_list = cost_functional_list
        self.state_weak_forms = state_weak_forms
        self.state_weak_forms_sum: ufl.Form = self.state_weak_forms[0]
        for form in self.state_weak_forms[1:]:
            self.state_weak_forms_sum += form

    def derivative(
        self, argument: ufl.core.expr.Expr, direction: ufl.core.expr.Expr
    ) -> ufl.Form:
        # d(lagrangian)/d(control) = d(object_function)/d(control) + d(lagrangian multiplier * equation constraint)/d(control)

        # d(object_function)/d(control)
        cost_derivative = self.cost_functional_list[0].derivative(argument, direction)  # type: ignore
        for functional in self.cost_functional_list[1:]:
            cost_derivative += functional.derivative(argument, direction)  # type: ignore

        # d(lagrangian multiplier * equation constraint)/d(control)
        state_weak_forms_derivative = ufl.derivative(
            self.state_weak_forms_sum, argument, direction
        )
        arg_derivative = cost_derivative + state_weak_forms_derivative

        return arg_derivative

    def coefficients(self) -> Set[dolfinx.fem.Function]:
        state_coeffs = set(self.state_weak_forms_sum.coefficients())  # type: ignore
        functional_coeffs = [
            set(functional.coefficients()) for functional in self.cost_functional_list
        ]
        coeffs = set().union(*functional_coeffs)
        coeffs.union(state_coeffs)
        return coeffs


class IntegralFunctional(CostFunctional):
    """
    object_function J = integral(cost_form)
    """

    def __init__(
        self,
        name: str,
        domain: dolfinx.mesh.Mesh,
        cost_form: ufl.Form,
        weight: float = 1.0,  # todo 后期改为场函数
    ):
        super().__init__(name)
        self.domain = domain
        self.weight = dolfinx.fem.Constant(self.domain, weight)  # type: ignore
        self.cost_form = self.weight * cost_form
        self.cost_form_dolfin = dolfinx.fem.form(self.cost_form)

    def evaluate(self):
        return AssembleUtils.assemble_scalar(self.cost_form_dolfin)  # type: ignore

    def derivative(
        self,
        argument: ufl.Coefficient,
        direction: dolfinx.fem.Function | ufl.Argument,
    ):
        return ufl.derivative(self.cost_form, argument, direction)

    def coefficients(self):
        return self.cost_form.coefficients()

    def update(self):
        pass

    def update_weight(self, weight: float):
        self.weight.value = weight


class ScalarTrackingFunctional(CostFunctional):
    """
    object_function J = tracking_goal - integral(cost_form)
    """

    def __init__(
        self,
        name: str,
        domain: dolfinx.mesh.Mesh,
        form: ufl.Form,
        tracking_goal: float,
        weight: float = 1.0,
        method: Literal["abs_sub", "pct_div"] = "abs_sub",
        **kwargs,
    ):
        '''
        Args:
            method:
                abs_sub: absolute subtraction
                pct_div: percentage division
            kwargs:
                max_bound: max bound of percentage division
        '''
        super().__init__(name)
        self.domain = domain
        self.method = method
        if self.method == "pct_div":
            self.max_bound = kwargs["max_bound"]

        self.weight = dolfinx.fem.Constant(self.domain, weight)  # type: ignore
        self.cost_form = self.weight * form
        self.cost_form_dolfin = dolfinx.fem.form(self.cost_form)

        self.tracking_goal: float = tracking_goal
        self.derivative_value = dolfinx.fem.Constant(domain, 0.0)  # type: ignore

    def evaluate(self):
        integral_value: float = AssembleUtils.assemble_scalar(self.cost_form_dolfin)  # type: ignore
        if self.method == "abs_sub":
            residual_value = 0.5 * np.power(integral_value - self.tracking_goal, 2)
        elif self.method == "pct_div":
            residual_value = np.min(
                integral_value / self.tracking_goal - 1.0, self.max_bound
            )
        else:
            raise ValueError(f"method {self.method} not supported")
        return residual_value * self.weight.value

    def coefficients(self):
        return self.cost_form.coefficients()

    def derivative(
        self,
        argument: ufl.Coefficient,
        direction: dolfinx.fem.Function | ufl.Argument,
    ):
        derivative = ufl.derivative(self.weight * self.derivative_value * self.cost_form, argument, direction)  # type: ignore
        return derivative

    def update(self):
        integral_value: float = AssembleUtils.assemble_scalar(self.cost_form_dolfin)  # type: ignore
        if self.method == "abs_sub":
            self.derivative_value.value = integral_value - self.tracking_goal
        elif self.method == "pct_div":
            self.derivative_value.value = np.min(
                integral_value / self.tracking_goal - 1.0, self.max_bound
            )
        else:
            raise ValueError

    def update_weight(self, weight: float):
        self.weight.value = weight


class MinMaxFunctional(CostFunctional):
    def __init__(
            self,
            name: str,
            domain: dolfinx.mesh.Mesh,
            cost_form: ufl.Form,
            lower_bound: float | None,
            upper_bound: float | None,
            weight: float = 1.0
    ):
        assert lower_bound is not None or upper_bound is not None, "lower_bound and upper_bound cannot be None at the same time"
        super().__init__(name)
        self.domain = domain
        self.weight = dolfinx.fem.Constant(self.domain, weight)  # type: ignore
        self.cost_form = self.weight * cost_form
        self.cost_form_dolfin = dolfinx.fem.form(self.cost_form)

        self.residual_form = 0.0
        self.lower_bound = lower_bound
        if self.lower_bound is not None:
            self.lower_bound_dolfin = dolfinx.fem.Constant(domain, 0.0)  # type: ignore
            self.residual_form += self.lower_bound_dolfin  # type: ignore

        self.upper_bound = upper_bound
        if self.upper_bound is not None:
            self.upper_bound_dolfin = dolfinx.fem.Constant(domain, 0.0)  # type: ignore
            self.residual_form += self.upper_bound_dolfin  # type: ignore

    def evaluate(self):
        integral_value: float = AssembleUtils.assemble_scalar(self.cost_form_dolfin)  # type: ignore
        cost = 0.0
        if self.lower_bound is not None:
            cost += 0.5 * np.power(np.maximum(0.0, self.lower_bound - integral_value), 2)
        if self.upper_bound is not None:
            cost += 0.5 * np.power(np.maximum(0.0, integral_value - self.upper_bound), 2)
        return self.weight.value * cost

    def coefficients(self):
        return self.cost_form.coefficients()

    def derivative(
            self,
            argument: ufl.Coefficient,
            direction: dolfinx.fem.Function | ufl.Argument,
    ):
        derivative = ufl.derivative(self.weight * self.residual_form * self.cost_form, argument, direction)  # type: ignore
        return derivative

    def update(self):
        integral_value: float = AssembleUtils.assemble_scalar(self.cost_form_dolfin)  # type: ignore

        if self.lower_bound is not None:
            if integral_value < self.lower_bound:
                self.lower_bound_dolfin.value = self.lower_bound - integral_value
            else:
                self.lower_bound_dolfin.value = 0.0

        if self.upper_bound is not None:
            if integral_value > self.upper_bound:
                self.upper_bound_dolfin.value = integral_value - self.upper_bound
            else:
                self.upper_bound_dolfin.value = 0.0

    def update_weight(self, weight: float):
        self.weight.value = weight
