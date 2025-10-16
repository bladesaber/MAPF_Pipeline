from typing import Literal, Union

import dolfinx
import numpy as np
import ufl

from scripts_py.dolfinx_utils import AssembleUtils
from scripts_py.mesh_utils import MeshUtils


class ControlRegularization(object):
    def __init__(self, name: str):
        self.name = name

    def derivative(self):
        raise NotImplementedError
    
    def evaluate(self):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError

    def update_weight(self, weight):
        raise NotImplementedError


class ControlIntegralRegularization(ControlRegularization):
    '''regularization formula = integral(control) * weight'''

    def __init__(
        self,
        name: str,
        domain: dolfinx.mesh.Mesh,
        integral_form: Union[ufl.Form, dolfinx.fem.Function],
        control: dolfinx.fem.Function,
        weight: float,
        target_value: float,
        **kwargs,
    ):
        super().__init__(name)
        self.weight: float = weight
        self.target_value = target_value
        self.domain = domain
        self.control = control

        self.dx = MeshUtils.define_dx(self.domain)
        self.integral_form = integral_form
        self.control_integral_form: ufl.Form = integral_form * self.dx

        self.test_v = ufl.TestFunction(self.control.function_space)
        self._residual_form = dolfinx.fem.Constant(self.domain, 0.0)  # type: ignore

    def _compute_integral_value(self) -> float:
        return AssembleUtils.assemble_scalar(dolfinx.fem.form(self.control_integral_form))  # type: ignore

    def derivative(self) -> ufl.Form:
        # the same as shape_form = self.weight * self.volume_dif * ufl.div(self.test_v) * self.dx
        grad_form = ufl.derivative(
            self.weight * self._residual_form * self.control_integral_form, self.control, self.test_v  # type: ignore
        )
        return grad_form

    def evaluate(self):
        value_integral = self._compute_integral_value()
        return 0.5 * self.weight * np.power(value_integral - self.target_value, 2)

    def update(self):
        value_integral = self._compute_integral_value()
        self._residual_form.value = value_integral - self.target_value

    def update_weight(self, weight):
        self.weight = weight


class ControlRegularizationPipe(object):
    def __init__(self, regularization_list: list[ControlRegularization]):
        self.regularization_list = regularization_list

    def evaluate(self) -> tuple[float, dict]:
        value = 0.0
        reg_dict = {}
        for reg in self.regularization_list:
            reg_value = reg.evaluate()
            value += reg_value
            reg_dict[reg.name] = reg_value
        return value, reg_dict

    def derivative(self) -> Union[ufl.Form, float]:
        value = 0.0
        for reg in self.regularization_list:
            value += reg.derivative()
        return value

    def update(self):
        for reg in self.regularization_list:
            reg.update()

