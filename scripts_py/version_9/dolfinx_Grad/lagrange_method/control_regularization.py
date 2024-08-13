import dolfinx
import numpy as np
import ufl
from typing import Union, List, Literal

from ..dolfinx_utils import AssembleUtils, MeshUtils
from .type_database import ControlDataBase, GovDataBase


class ControlIntegralRegularization(object):
    """
    Need more Academic Basement
    """

    def __init__(
            self,
            domain: dolfinx.mesh.Mesh,
            integral_form: ufl.Form,
            control: dolfinx.fem.Function,
            mu: float,
            target_control_volume: float,
            method: Literal['absolute_sub', 'percentage_div'] = 'absolute_sub',
            name='ControlIntegralRegularization'
    ):
        self.name = name
        self.mu = mu
        self.target_control_volume = target_control_volume
        self.domain = domain
        self.integral_form = integral_form
        self.control = control

        self.is_active = False
        if self.mu > 0.0:
            self.is_active = True

        self.constant = dolfinx.fem.Constant(self.domain, 1.0)
        self.dx = MeshUtils.define_dx(self.domain)
        self.control_volume_form: ufl.Form = self.constant * integral_form * self.dx

        self.test_v = ufl.TestFunction(self.control.function_space)
        self.volume_dif = dolfinx.fem.Constant(self.domain, 0.0)
        self.method = method

    def _compute_volume(self) -> float:
        volume: float = AssembleUtils.assemble_scalar(dolfinx.fem.form(self.control_volume_form))
        return volume

    def compute_control_derivative(self):
        if self.is_active:
            """
            Method 1 and Method 2 are the same
            """
            # method1
            # shape_form = self.mu * self.volume_dif * ufl.div(self.test_v) * self.dx

            # method2
            grad_form = ufl.derivative(self.mu * self.volume_dif * self.control_volume_form, self.control, self.test_v)

            return grad_form
        else:
            return 0.0

    def compute_objective(self):
        value = 0.0
        if self.is_active:
            volume = self._compute_volume()
            if self.method == 'absolute_sub':
                value += 0.5 * self.mu * pow(volume - self.target_control_volume, 2)
            elif self.method == 'percentage_div':
                value += self.mu * np.sqrt(np.power(volume / self.target_control_volume - 1.0, 2))
            else:
                raise NotImplementedError
        return value

    def update(self):
        current_volume = self._compute_volume()
        if self.method == 'absolute_sub':
            volume_dif = current_volume - self.target_control_volume
        elif self.method == 'percentage_div':
            volume_dif = current_volume / self.target_control_volume - 1.0
        else:
            raise NotImplementedError
        self.volume_dif.value = volume_dif

    def update_weight(self, mu):
        self.mu = mu


class ControlRegularization(object):
    def __init__(self, regularization_list: List[Union[ControlIntegralRegularization]]):
        self.regularization_list = regularization_list

    def compute_objective(self):
        value = 0.0
        for term in self.regularization_list:
            value += term.compute_objective()
        return value

    def compute_control_derivative(self):
        value = 0.0
        for term in self.regularization_list:
            value += term.compute_control_derivative()
        return value

    def update(self):
        for term in self.regularization_list:
            term.update()

    def get_cost_info(self) -> List:
        cost_info = []
        for term in self.regularization_list:
            cost_info.append((term.name, term.compute_objective()))
        return cost_info
