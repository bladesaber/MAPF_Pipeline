from typing import Union, Literal

import dolfinx
import ufl

import numpy as np

from scripts_py.mesh_utils import MeshUtils
from scripts_py.dolfinx_utils import AssembleUtils
from scripts_py.dolfinx_adjoint.base_type import ShapeProblemDataBase
from scripts_py.eqn_solver import LinearProblemSolver


def t_grad(u: Union[ufl.Argument, dolfinx.fem.Function], n: ufl.FacetNormal):
    """Computes the tangential gradient of u.
    Args:
        u: The argument, whose tangential gradient is to be computed.
        n: The unit outer normal vector.
    Returns:
        The tangential gradient of u.
    ufl.inner为Frobenius内积
    """
    return ufl.grad(u) - ufl.outer(ufl.grad(u) * n, n)  # type: ignore


def t_div(u: Union[ufl.Argument, dolfinx.fem.Function], n: ufl.FacetNormal):
    """Computes the tangential divergence of u.

    Args:
        u: The argument, whose tangential divergence is to be computed.
        n: The unit outer normal vector.

    Returns:
        The tangential divergence of u.

    """
    return ufl.div(u) - ufl.inner(ufl.grad(u) * n, n)  # type: ignore


class ShapeRegularization(object):
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


class VolumeRegularization(ShapeRegularization):
    '''regularization formula = (volume - target_volume) * weight'''
    def __init__(
            self,
            name: str,
            shape_database: ShapeProblemDataBase,
            weight: float,
            target_volume_rho: float,
            **kwargs,
    ):
        self.name = name
        self.weight = weight
        self.target_volume_rho = target_volume_rho
        self.shape_database = shape_database

        self.constant = dolfinx.fem.Constant(self.shape_database.domain, 1.0)  # type: ignore
        self.dx = MeshUtils.define_dx(self.shape_database.domain)
        self.volume_form: ufl.Form = self.constant * self.dx
        self.volume_form_dolfin = dolfinx.fem.form(self.volume_form)

        self.target_volume = self._compute_volume() * self.target_volume_rho
        self.test_v = ufl.TestFunction(self.shape_database.deformation_space)

        self.volume_residual = dolfinx.fem.Constant(self.shape_database.domain, 0.0)  # type: ignore
        self.coord = MeshUtils.define_coordinate(self.shape_database.domain)

    def _compute_volume(self) -> float:
        return AssembleUtils.assemble_scalar(self.volume_form_dolfin) # type: ignore

    def derivative(self):
        # same as shape_form = self.mu * self.volume_dif * ufl.div(self.test_v) * self.dx
        shape_form = ufl.derivative(self.weight * self.volume_residual * self.volume_form, self.coord, self.test_v) # type: ignore
        return shape_form

    def evaluate(self):
        volume = self._compute_volume()
        return 0.5 * self.weight * np.power(volume - self.target_volume, 2)

    def update(self):
        current_volume = self._compute_volume()
        self.volume_residual.value = current_volume - self.target_volume

    def update_weight(self, weight):
        self.weight = weight


class SurfaceAreaRegularization(ShapeRegularization):
    '''regularization formula = (surface_area - target_surface_area) * weight'''
    def __init__(
            self,
            name: str,
            shape_database: ShapeProblemDataBase,
            weight: float,
            target_surface_rho: float
    ):
        self.name = name
        self.weight = weight
        self.target_surface_rho = target_surface_rho
        self.shape_database = shape_database

        self.constant = dolfinx.fem.Constant(self.shape_database.domain, 1.0)  # type: ignore
        self.ds = MeshUtils.define_ds(self.shape_database.domain)
        self.norm_vec = MeshUtils.define_facet_norm(self.shape_database.domain)
        self.surface_form: ufl.Form = self.constant * self.ds
        self.surface_form_dolfin = dolfinx.fem.form(self.surface_form)

        self.target_surface = self._compute_surface() * self.target_surface_rho
        self.test_v = ufl.TestFunction(self.shape_database.deformation_space)

        self.surface_residual = dolfinx.fem.Constant(self.shape_database.domain, 0.0)  # type: ignore
        self.coord = MeshUtils.define_coordinate(self.shape_database.domain)

    def _compute_surface(self) -> float:
        return AssembleUtils.assemble_scalar(self.surface_form_dolfin) # type: ignore

    def evaluate(self) -> float:
        surface_area = self._compute_surface()
        return 0.5 * self.weight * np.power(surface_area - self.target_surface, 2)

    def derivative(self):
        # same as shape_form = self.mu * self.surface_dif * t_div(self.test_v, self.norm_vec) * self.ds
        shape_form = ufl.derivative(self.weight * self.surface_residual * self.surface_form, self.coord, self.test_v) # type: ignore
        return shape_form

    def update(self):
        current_surface = self._compute_surface()
        self.surface_residual.value = current_surface - self.target_surface

    def update_weight(self, weight):
        self.weight = weight


class BarycenterRegularization(ShapeRegularization):
    '''regularization formula = (barycenter - target_barycenter) * weight'''
    def __init__(
            self,
            name: str,
            shape_database: ShapeProblemDataBase,
            weight: float,
            target_barycenter: np.ndarray
    ):
        assert target_barycenter.shape[0] in [2, 3], "target_barycenter must be a 2D or 3D array"
        
        self.name = name
        self.shape_database = shape_database
        self.weight = weight

        self.coord = MeshUtils.define_coordinate(self.shape_database.domain)
        self.dx = MeshUtils.define_dx(self.shape_database.domain)
        self.constant = dolfinx.fem.Constant(self.shape_database.domain, 1.0)  # type: ignore
        self.tdim = self.shape_database.domain.topology.dim

        self.volume_form: ufl.Form = self.constant * self.dx
        self.volume_form_dolfin = dolfinx.fem.form(self.volume_form)
        self.test_v = ufl.TestFunction(self.shape_database.deformation_space)
        self.cur_volume = dolfinx.fem.Constant(self.shape_database.domain, self._compute_volume())  # type: ignore

        # ------ Define x, y, z form ------
        self.x_form = self.coord[0] * self.dx  # type: ignore
        self.y_form = self.coord[1] * self.dx  # type: ignore
        if self.tdim == 3:
            self.z_form = self.coord[2] * self.dx  # type: ignore

        # ------ Define target barycenter ------
        self.target_barycenter = target_barycenter
        self.target_barycenter_x = dolfinx.fem.Constant(self.shape_database.domain, self.target_barycenter[0])
        self.target_barycenter_y = dolfinx.fem.Constant(self.shape_database.domain, self.target_barycenter[1])
        self.target_barycenter_z = dolfinx.fem.Constant(self.shape_database.domain, self.target_barycenter[2])

        # ------ Define current barycenter ------
        barycenter_np = self._compute_barycenter()
        self.cur_barycenter_x = dolfinx.fem.Constant(self.shape_database.domain, barycenter_np[0])
        self.cur_barycenter_y = dolfinx.fem.Constant(self.shape_database.domain, barycenter_np[1])
        if self.tdim == 3:
            self.cur_barycenter_z = dolfinx.fem.Constant(self.shape_database.domain, barycenter_np[2])

    def _compute_volume(self) -> float:
        return AssembleUtils.assemble_scalar(self.volume_form_dolfin) # type: ignore

    def _compute_barycenter(self, volume: float | None = None) -> np.ndarray:
        volume = self._compute_volume() if volume is None else volume
        barycenter_np = np.zeros(self.tdim)
        barycenter_np[0] = AssembleUtils.assemble_scalar(dolfinx.fem.form(self.x_form)) / volume  # type: ignore
        barycenter_np[1] = AssembleUtils.assemble_scalar(dolfinx.fem.form(self.y_form)) / volume  # type: ignore
        if self.tdim == 3:
            barycenter_np[2] = AssembleUtils.assemble_scalar(dolfinx.fem.form(self.z_form)) / volume  # type: ignore
        return barycenter_np

    def derivative(self):
        # the same as
        # shape_form = self.mu * (self.cur_barycenter_x - self.target_barycenter_x) * (
        #         self.cur_barycenter_x / self.cur_volume * ufl.div(self.test_v) +
        #         1. / self.cur_volume * (self.test_v[0] + self.coodr[0] * ufl.div(self.test_v))
        # ) * self.dx
        #
        # shape_form += self.mu * (self.cur_barycenter_y - self.target_barycenter_y) * (
        #         self.cur_barycenter_y / self.cur_volume * ufl.div(self.test_v) +
        #         1. / self.cur_volume * (self.test_v[1] + self.coodr[1] * ufl.div(self.test_v))
        # ) * self.dx
        # if self.shape_problem.domain.geometry.dim == 3:
        #     shape_form += self.mu * (self.cur_barycenter_z - self.target_barycenter_z) * (
        #             self.cur_barycenter_z / self.cur_volume * ufl.div(self.test_v) +
        #             1. / self.cur_volume * (self.test_v[2] + self.coodr[2] * ufl.div(self.test_v))
        #     ) * self.dx

        # todo need to check
        shape_form = ufl.derivative(
            self.weight * (self.cur_barycenter_x - self.target_barycenter_x) * self.x_form, self.coord[0], self.test_v  # type: ignore
        )
        shape_form += ufl.derivative(
            self.weight * (self.cur_barycenter_y - self.target_barycenter_y) * self.y_form, self.coord[1], self.test_v  # type: ignore
        )
        if self.tdim == 3:
            shape_form += ufl.derivative(
                self.weight * (self.cur_barycenter_z - self.target_barycenter_z) * self.z_form, self.coord[2], self.test_v  # type: ignore
            )
        return shape_form

    def evaluate(self):
        return 0.5 * self.weight * np.sum(np.abs(self._compute_barycenter() - self.target_barycenter))

    def update(self):
        cur_volume = self._compute_volume()
        cur_barycenter_np = self._compute_barycenter(cur_volume)

        self.cur_volume.value = cur_volume
        self.cur_barycenter_x.value = cur_barycenter_np[0]
        self.cur_barycenter_y.value = cur_barycenter_np[1]
        if self.tdim == 3:
            self.cur_barycenter_z.value = cur_barycenter_np[3]

    def update_weight(self, weight):
        self.weight = weight


class CurvatureRegularization(ShapeRegularization):
    pass


class ShapeRegularizationPipe(object):
    def __init__(self, regularization_list: list[ShapeRegularization]):
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

