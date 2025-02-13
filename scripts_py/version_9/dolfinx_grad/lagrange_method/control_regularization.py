import dolfinx
import numpy as np
import ufl
from typing import Union, List, Literal, Dict

from scripts_py.version_9.dolfinx_grad.dolfinx_utils import AssembleUtils
from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils


class ControlIntegralRegularization(object):
    """control变量的积分形式正则项"""

    def __init__(
            self,
            name: str,
            domain: dolfinx.mesh.Mesh,
            integral_form: Union[ufl.Form, dolfinx.fem.Function],
            control: dolfinx.fem.Function,
            weight: float,
            target_value: float,
            method: Literal['absolute_sub', 'percentage_div'] = 'absolute_sub',
    ):
        self.name = name
        self.weight = weight
        self.target_value = target_value
        self.integral_form = integral_form
        self.domain = domain
        self.control = control
        self.is_active = True if self.weight > 0.0 else False

        self.dx = MeshUtils.define_dx(self.domain)
        self.control_integral_form: ufl.Form = integral_form * self.dx

        self.test_v = ufl.TestFunction(self.control.function_space)
        self.value_dif = dolfinx.fem.Constant(self.domain, 0.0)
        self.method = method

    def _compute_integral_value(self) -> float:
        return AssembleUtils.assemble_scalar(dolfinx.fem.form(self.control_integral_form))

    def compute_control_derivative(self):
        if self.is_active:
            """Method 1 and Method 2 are the same"""
            # shape_form = self.weight * self.volume_dif * ufl.div(self.test_v) * self.dx  # method_1

            grad_form = ufl.derivative(
                self.weight * self.value_dif * self.control_integral_form,
                self.control, self.test_v
            )  # method_2
            return grad_form
        else:
            return 0.0

    def compute_objective(self):
        value = 0.0
        if self.is_active:
            value_integral = self._compute_integral_value()
            if self.method == 'absolute_sub':
                value += 0.5 * self.weight * np.power(value_integral - self.target_value, 2)
            elif self.method == 'percentage_div':
                value += self.weight * np.sqrt(np.power(value_integral / self.target_value - 1.0, 2))
            else:
                raise NotImplementedError
        return value

    def update(self):
        value_integral = self._compute_integral_value()
        if self.method == 'absolute_sub':
            value_dif = value_integral - self.target_value
        elif self.method == 'percentage_div':
            value_dif = value_integral / self.target_value - 1.0
        else:
            raise NotImplementedError
        self.value_dif.value = value_dif

    def update_weight(self, weight):
        self.weight = weight


class ControlRegularizationPipe(object):
    def __init__(self, regularization_list: List[ControlIntegralRegularization]):
        self.regularization_list = regularization_list

    def compute_objective(self) -> float:
        value = 0.0
        for reg in self.regularization_list:
            value += reg.compute_objective()
        return value

    def compute_control_derivative(self) -> Union[ufl.Form, float]:
        value = 0.0
        for reg in self.regularization_list:
            value += reg.compute_control_derivative()
        return value

    def update(self):
        for reg in self.regularization_list:
            reg.update()

    def get_cost_info(self) -> Dict:
        cost_dict = {}
        for term in self.regularization_list:
            cost_dict[term.name] = term.compute_objective()
        return cost_dict
