import dolfinx
import numpy as np
import ufl
from typing import Union, List, Literal

from ..dolfinx_utils import AssembleUtils, MeshUtils
from .type_database import ShapeDataBase
from ..equation_solver import LinearProblemSolver

from ..vis_mesh_utils import VisUtils
import pyvista


def t_grad(u: Union[ufl.Argument, dolfinx.fem.Function], n: ufl.FacetNormal):
    """Computes the tangential gradient of u.

    Args:
        u: The argument, whose tangential gradient is to be computed.
        n: The unit outer normal vector.

    Returns:
        The tangential gradient of u.

    """
    return ufl.grad(u) - ufl.outer(ufl.grad(u) * n, n)


def t_div(u: Union[ufl.Argument, dolfinx.fem.Function], n: ufl.FacetNormal):
    """Computes the tangential divergence of u.

    Args:
        u: The argument, whose tangential divergence is to be computed.
        n: The unit outer normal vector.

    Returns:
        The tangential divergence of u.

    """
    return ufl.div(u) - ufl.inner(ufl.grad(u) * n, n)


class VolumeRegularization(object):
    """
    Need more Academic Basement
    """

    def __init__(
            self,
            shape_problem: ShapeDataBase,
            mu: float,
            target_volume_rho: float,
            method: Literal['absolute_sub', 'percentage_div'] = 'absolute_sub',
            name='VolumeRegularization'
    ):
        self.name = name
        self.mu = mu
        self.target_volume_rho = target_volume_rho
        self.shape_problem = shape_problem

        self.is_active = False
        if self.mu > 0.0:
            self.is_active = True

        self.constant = dolfinx.fem.Constant(self.shape_problem.domain, 1.0)
        self.dx = MeshUtils.define_dx(self.shape_problem.domain)
        self.volume_form: ufl.Form = self.constant * self.dx
        self.target_volume = self._compute_volume() * self.target_volume_rho
        self.test_v = ufl.TestFunction(self.shape_problem.deformation_space)

        self.volume_dif = dolfinx.fem.Constant(self.shape_problem.domain, 0.0)
        self.coodr = MeshUtils.define_coordinate(self.shape_problem.domain)
        self.method = method

    def _compute_volume(self) -> float:
        volume: float = AssembleUtils.assemble_scalar(dolfinx.fem.form(self.volume_form))
        return volume

    def compute_shape_derivative(self):
        if self.is_active:
            """
            Method 1 and Method 2 are the same
            """
            # method1
            # shape_form = self.mu * self.volume_dif * ufl.div(self.test_v) * self.dx

            # method2
            shape_form = ufl.derivative(self.mu * self.volume_dif * self.volume_form, self.coodr, self.test_v)

            return shape_form
        else:
            return 0.0

    def compute_objective(self):
        value = 0.0
        if self.is_active:
            volume = self._compute_volume()
            if self.method == 'absolute_sub':
                value += 0.5 * self.mu * pow(volume - self.target_volume, 2)
            elif self.method == 'percentage_div':
                value += self.mu * np.sqrt(np.power(volume / self.target_volume - 1.0, 2))
            else:
                raise NotImplementedError
        return value

    def update(self):
        current_volume = self._compute_volume()
        if self.method == 'absolute_sub':
            volume_dif = current_volume - self.target_volume
        elif self.method == 'percentage_div':
            volume_dif = current_volume / self.target_volume - 1.0
        else:
            raise NotImplementedError
        self.volume_dif.value = volume_dif

    def update_weight(self, mu):
        self.mu = mu


class SurfaceRegularization(object):
    """
    Need more Academic Basement
    """

    def __init__(
            self,
            shape_problem: ShapeDataBase,
            mu: float,
            target_surface_rho: float,
            name='SurfaceRegularization'
    ):
        self.name = name
        self.mu = mu
        self.target_surface_rho = target_surface_rho
        self.shape_problem = shape_problem

        self.is_active = False
        if self.mu > 0.0:
            self.is_active = True

        self.constant = dolfinx.fem.Constant(self.shape_problem.domain, 1.0)
        self.ds = MeshUtils.define_ds(self.shape_problem.domain)
        self.n = MeshUtils.define_facet_norm(self.shape_problem.domain)
        self.surface_form: ufl.Form = self.constant * self.ds
        self.target_surface = self._compute_surface() * self.target_surface_rho
        self.test_v = ufl.TestFunction(self.shape_problem.deformation_space)

        self.surface_dif = dolfinx.fem.Constant(self.shape_problem.domain, 0.0)
        self.coodr = MeshUtils.define_coordinate(self.shape_problem.domain)

    def compute_objective(self) -> float:
        value = 0.0
        if self.is_active:
            surface = self._compute_surface()
            value = 0.5 * self.mu * pow(surface - self.target_surface, 2)
        return value

    def compute_shape_derivative(self):
        if self.is_active:
            # shape_form = self.mu * self.surface_dif * t_div(self.test_v, self.n) * self.ds
            # TODO I am not sure
            shape_form = ufl.derivative(self.mu * self.surface_dif * self.surface_form, self.coodr, self.test_v)
            return shape_form
        else:
            return 0.0

    def _compute_surface(self) -> float:
        surface: float = AssembleUtils.assemble_scalar(dolfinx.fem.form(self.surface_form))
        return surface

    def update(self):
        current_surface = self._compute_surface()
        surface_dif = current_surface - self.target_surface
        self.surface_dif.value = surface_dif

    def update_weight(self, mu):
        self.mu = mu


class BarycenterRegularization(object):
    def __init__(
            self,
            shape_problem: ShapeDataBase,
            mu: float,
            target_barycenter: np.ndarray,
            name='BarycenterRegularization'
    ):
        self.name = name
        self.shape_problem = shape_problem
        self.mu = mu
        self.target_barycenter = target_barycenter

        self.is_active = False
        if self.mu > 0.0:
            self.is_active = True

        self.coodr = MeshUtils.define_coordinate(self.shape_problem.domain)
        self.dx = MeshUtils.define_dx(self.shape_problem.domain)
        self.constant = dolfinx.fem.Constant(self.shape_problem.domain, 1.0)

        self.tdim = self.shape_problem.domain.topology.dim
        self.x_form = self.coodr[0] * self.dx
        self.y_form = self.coodr[1] * self.dx
        if self.tdim == 3:
            self.z_form = self.coodr[2] * self.dx

        self.volume_form: ufl.Form = self.constant * self.dx
        self.test_v = ufl.TestFunction(self.shape_problem.deformation_space)

        self.target_barycenter_x = dolfinx.fem.Constant(self.shape_problem.domain, self.target_barycenter[0])
        self.target_barycenter_y = dolfinx.fem.Constant(self.shape_problem.domain, self.target_barycenter[1])
        self.target_barycenter_z = dolfinx.fem.Constant(self.shape_problem.domain, self.target_barycenter[2])

        barycenter_np = self._compute_barycenter()
        self.cur_barycenter_x = dolfinx.fem.Constant(self.shape_problem.domain, barycenter_np[0])
        self.cur_barycenter_y = dolfinx.fem.Constant(self.shape_problem.domain, barycenter_np[1])
        if self.tdim == 3:
            self.cur_barycenter_z = dolfinx.fem.Constant(self.shape_problem.domain, barycenter_np[2])

        volume = self._compute_volume()
        self.cur_volume = dolfinx.fem.Constant(self.shape_problem.domain, volume)

    def compute_shape_derivative(self):
        if self.is_active:
            # shape_form = self.mu * (self.cur_barycenter_x - self.target_barycenter_x) * (
            #         self.cur_barycenter_x / self.cur_volume * ufl.div(self.test_v) +
            #         1. / self.cur_volume * (self.test_v[0] + self.coodr[0] * ufl.div(self.test_v))
            # ) * self.dx
            #
            # shape_form += self.mu * (self.cur_barycenter_y - self.target_barycenter_y) * (
            #         self.cur_barycenter_y / self.cur_volume * ufl.div(self.test_v) +
            #         1. / self.cur_volume * (self.test_v[1] + self.coodr[1] * ufl.div(self.test_v))
            # ) * self.dx
            #
            # if self.shape_problem.domain.geometry.dim == 3:
            #     shape_form += self.mu * (self.cur_barycenter_z - self.target_barycenter_z) * (
            #             self.cur_barycenter_z / self.cur_volume * ufl.div(self.test_v) +
            #             1. / self.cur_volume * (self.test_v[2] + self.coodr[2] * ufl.div(self.test_v))
            #     ) * self.dx

            # TODO I am not sure
            shape_form = ufl.derivative(
                self.mu * (self.cur_barycenter_x - self.target_barycenter_x) * self.x_form,
                self.coodr[0], self.test_v
            )
            shape_form += ufl.derivative(
                self.mu * (self.cur_barycenter_y - self.target_barycenter_y) * self.y_form,
                self.coodr[1], self.test_v
            )
            if self.shape_problem.domain.geometry.dim == 3:
                shape_form += ufl.derivative(
                    self.mu * (self.cur_barycenter_z - self.target_barycenter_z) * self.z_form,
                    self.coodr[2], self.test_v
                )

            return shape_form
        else:
            return 0.0

    def compute_objective(self):
        value = 0.0
        if self.is_active:
            barycenter_np = self._compute_barycenter()

            value = np.pow(barycenter_np[0] - self.target_barycenter[0], 2) + \
                    np.pow(barycenter_np[1] - self.target_barycenter[1], 2)

            if self.tdim == 3:
                value += np.pow(barycenter_np[2] - self.target_barycenter[2], 2)

            value = 0.5 * self.mu * value

        return value

    def _compute_barycenter(self, volume=None):
        if volume is None:
            volume = self._compute_volume()

        barycenter_np = np.zeros(self.tdim)
        barycenter_np[0] = AssembleUtils.assemble_scalar(dolfinx.fem.form(self.x_form)) / volume
        barycenter_np[1] = AssembleUtils.assemble_scalar(dolfinx.fem.form(self.y_form)) / volume
        if self.tdim == 3:
            barycenter_np[2] = AssembleUtils.assemble_scalar(dolfinx.fem.form(self.z_form)) / volume

        return barycenter_np

    def _compute_volume(self) -> float:
        volume: float = AssembleUtils.assemble_scalar(dolfinx.fem.form(self.volume_form))
        return volume

    def update(self):
        cur_volume = self._compute_volume()
        cur_barycenter_np = self._compute_barycenter(cur_volume)

        self.cur_volume.value = cur_volume
        self.cur_barycenter_x.value = cur_barycenter_np[0]
        self.cur_barycenter_y = cur_barycenter_np[1]
        if self.tdim == 3:
            self.cur_barycenter_z.value = cur_barycenter_np[3]

    def update_weight(self, mu):
        self.mu = mu


class CurvatureRegularization(object):
    """
    Need more Academic Basement
    """

    def __init__(
            self,
            shape_problem: ShapeDataBase,
            mu: float,
            ksp_option,
            name='CurvatureRegularization'
    ):
        self.name = name
        self.shape_problem = shape_problem
        self.mu = mu
        self.ksp_option = ksp_option

        self.is_active = False
        if self.mu > 0.0:
            self.is_active = True

        self.kappa_curvature = dolfinx.fem.Function(self.shape_problem.deformation_space)
        self.n = MeshUtils.define_facet_norm(self.shape_problem.domain)
        self.coodr = MeshUtils.define_coordinate(self.shape_problem.domain)
        self.ds = MeshUtils.define_ds(self.shape_problem.domain)

        self.a_curvature_form = ufl.inner(
            ufl.TrialFunction(self.shape_problem.deformation_space),
            ufl.TestFunction(self.shape_problem.deformation_space)
        ) * self.ds

        self.l_curvature_form = ufl.inner(
            t_grad(self.coodr, self.n),
            t_grad(ufl.TestFunction(self.shape_problem.deformation_space), self.n),
        ) * self.ds

        self.test_v = ufl.TestFunction(self.shape_problem.deformation_space)

    def compute_shape_derivative(self):
        if self.is_active:
            identity = ufl.Identity(self.shape_problem.domain.geometry.dim)

            shape_form = self.mu * ufl.inner(
                (identity - (t_grad(self.coodr, self.n) + (t_grad(self.coodr, self.n)).T)) * t_grad(self.test_v,
                                                                                                    self.n),
                t_grad(self.kappa_curvature, self.n)
            ) * self.ds + 0.5 * t_div(self.test_v, self.n) * t_div(self.kappa_curvature, self.n) * self.ds

            return shape_form
        else:
            return 0.0

    def compute_objective(self):
        value = 0.0

        if self.is_active:
            self._compute_curvature(self.shape_problem.domain.comm)

            curvature_val = AssembleUtils.assemble_scalar(dolfinx.fem.form(
                ufl.inner(self.kappa_curvature, self.kappa_curvature) * self.ds
            ))
            value += 0.5 * self.mu * curvature_val

        return value

    def _compute_curvature(self):
        res_dict = LinearProblemSolver.solve_by_petsc_form(
            comm=self.shape_problem.domain.comm,
            uh=self.kappa_curvature,
            a_form=self.a_curvature_form,
            L_form=self.l_curvature_form,
            bcs=[],
            ksp_option=self.ksp_option
        )

    def update(self):
        self._compute_curvature()

    def update_weight(self, mu):
        self.mu = mu


type_regulariztions = Union[
    VolumeRegularization, SurfaceRegularization, BarycenterRegularization, CurvatureRegularization
]


class ShapeRegularization(object):
    def __init__(self, regularization_list: List[type_regulariztions]):
        self.regularization_list = regularization_list

    def compute_objective(self):
        value = 0.0
        for term in self.regularization_list:
            value += term.compute_objective()
        return value

    def compute_shape_derivative(self):
        value = 0.0
        for term in self.regularization_list:
            value += term.compute_shape_derivative()
        return value

    def update(self):
        for term in self.regularization_list:
            term.update()

    def get_cost_info(self) -> List:
        cost_info = []
        for term in self.regularization_list:
            cost_info.append((term.name, term.compute_objective()))
        return cost_info
