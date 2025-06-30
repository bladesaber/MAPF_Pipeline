import dolfinx
import numpy as np
import ufl
from typing import Union, List, Literal, Dict

from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils
from scripts_py.version_9.dolfinx_grad.dolfinx_utils import AssembleUtils
from .type_database import ShapeDataBase
from scripts_py.version_9.dolfinx_grad.equation_solver import LinearProblemSolver


def t_grad(u: Union[ufl.Argument, dolfinx.fem.Function], n: ufl.FacetNormal):
    """Computes the tangential gradient of u.
    Args:
        u: The argument, whose tangential gradient is to be computed.
        n: The unit outer normal vector.
    Returns:
        The tangential gradient of u.
    ufl.inner为Frobenius内积
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
    def __init__(
            self,
            name: str,
            shape_problem: ShapeDataBase,
            weight: float,
            target_volume_rho: float,
            method: Literal['absolute_sub', 'percentage_div'] = 'absolute_sub',
    ):
        self.name = name
        self.weight = weight
        self.target_volume_rho = target_volume_rho
        self.shape_problem = shape_problem
        self.is_active = True if self.weight > 0.0 else False

        self.constant = dolfinx.fem.Constant(self.shape_problem.domain, 1.0)
        self.dx = MeshUtils.define_dx(self.shape_problem.domain)
        self.volume_form: ufl.Form = self.constant * self.dx

        self.target_volume = self._compute_volume() * self.target_volume_rho
        self.test_v = ufl.TestFunction(self.shape_problem.deformation_space)

        self.volume_dif = dolfinx.fem.Constant(self.shape_problem.domain, 0.0)
        self.coord = MeshUtils.define_coordinate(self.shape_problem.domain)
        self.method = method

    def _compute_volume(self) -> float:
        return AssembleUtils.assemble_scalar(dolfinx.fem.form(self.volume_form))

    def compute_shape_derivative(self):
        if self.is_active:
            # shape_form = self.mu * self.volume_dif * ufl.div(self.test_v) * self.dx  # be replaced
            shape_form = ufl.derivative(self.weight * self.volume_dif * self.volume_form, self.coord, self.test_v)
            return shape_form
        else:
            return 0.0

    def compute_objective(self):
        value = 0.0
        if self.is_active:
            volume = self._compute_volume()
            if self.method == 'absolute_sub':
                value += 0.5 * self.weight * np.power(volume - self.target_volume, 2)
            elif self.method == 'percentage_div':
                value += self.weight * np.sqrt(np.power(volume / self.target_volume - 1.0, 2))
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

    def update_weight(self, weight):
        self.weight = weight


class SurfaceAreaRegularization(object):
    def __init__(
            self,
            name: str,
            shape_problem: ShapeDataBase,
            weight: float,
            target_surface_rho: float
    ):
        self.name = name
        self.weight = weight
        self.target_surface_rho = target_surface_rho
        self.shape_problem = shape_problem
        self.is_active = True if self.weight > 0.0 else False

        self.constant = dolfinx.fem.Constant(self.shape_problem.domain, 1.0)
        self.ds = MeshUtils.define_ds(self.shape_problem.domain)
        self.norm_vec = MeshUtils.define_facet_norm(self.shape_problem.domain)
        self.surface_form: ufl.Form = self.constant * self.ds

        self.target_surface = self._compute_surface() * self.target_surface_rho
        self.test_v = ufl.TestFunction(self.shape_problem.deformation_space)

        self.surface_dif = dolfinx.fem.Constant(self.shape_problem.domain, 0.0)
        self.coord = MeshUtils.define_coordinate(self.shape_problem.domain)

    def compute_objective(self) -> float:
        value = 0.0
        if self.is_active:
            surface = self._compute_surface()
            value = 0.5 * self.weight * np.power(surface - self.target_surface, 2)
        return value

    def compute_shape_derivative(self):
        if self.is_active:
            # shape_form = self.mu * self.surface_dif * t_div(self.test_v, self.norm_vec) * self.ds
            shape_form = ufl.derivative(self.weight * self.surface_dif * self.surface_form, self.coord, self.test_v)
            return shape_form
        else:
            return 0.0

    def _compute_surface(self) -> float:
        return AssembleUtils.assemble_scalar(dolfinx.fem.form(self.surface_form))

    def update(self):
        current_surface = self._compute_surface()
        surface_dif = current_surface - self.target_surface
        self.surface_dif.value = surface_dif

    def update_weight(self, weight):
        self.weight = weight


class BarycenterRegularization(object):
    def __init__(
            self,
            name: str,
            shape_problem: ShapeDataBase,
            weight: float,
            target_barycenter: np.ndarray
    ):
        self.name = name
        self.shape_problem = shape_problem
        self.weight = weight
        self.target_barycenter = target_barycenter
        self.is_active = True if self.weight > 0.0 else False

        self.coord = MeshUtils.define_coordinate(self.shape_problem.domain)
        self.dx = MeshUtils.define_dx(self.shape_problem.domain)
        self.constant = dolfinx.fem.Constant(self.shape_problem.domain, 1.0)
        self.tdim = self.shape_problem.domain.topology.dim

        self.x_form = self.coord[0] * self.dx
        self.y_form = self.coord[1] * self.dx
        if self.tdim == 3:
            self.z_form = self.coord[2] * self.dx

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

            shape_form = ufl.derivative(
                self.weight * (self.cur_barycenter_x - self.target_barycenter_x) * self.x_form,
                self.coord[0], self.test_v
            )
            shape_form += ufl.derivative(
                self.weight * (self.cur_barycenter_y - self.target_barycenter_y) * self.y_form,
                self.coord[1], self.test_v
            )
            if self.shape_problem.domain.geometry.dim == 3:
                shape_form += ufl.derivative(
                    self.weight * (self.cur_barycenter_z - self.target_barycenter_z) * self.z_form,
                    self.coord[2], self.test_v
                )
            return shape_form
        else:
            return 0.0

    def compute_objective(self):
        value = 0.0
        if self.is_active:
            barycenter_np = self._compute_barycenter()
            value = (
                    np.power(barycenter_np[0] - self.target_barycenter[0], 2)
                    + np.power(barycenter_np[1] - self.target_barycenter[1], 2)
            )
            if self.tdim == 3:
                value += np.power(barycenter_np[2] - self.target_barycenter[2], 2)
            value = 0.5 * self.weight * value
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
        return AssembleUtils.assemble_scalar(dolfinx.fem.form(self.volume_form))

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


class CurvatureRegularization(object):
    """Todo: Fail method"""
    ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }

    def __init__(
            self,
            name: str,
            shape_problem: ShapeDataBase,
            weight: float,
            ksp_option: Dict = None
    ):
        self.name = name
        self.shape_problem = shape_problem
        self.weight = weight
        self.ksp_option = ksp_option if ksp_option is not None else self.ksp_option
        self.is_active = True if self.weight > 0.0 else False

        self.kappa_curvature = dolfinx.fem.Function(self.shape_problem.deformation_space)
        self.n = MeshUtils.define_facet_norm(self.shape_problem.domain)
        self.coord = MeshUtils.define_coordinate(self.shape_problem.domain)
        self.ds = MeshUtils.define_ds(self.shape_problem.domain)
        self.test_v = ufl.TestFunction(self.shape_problem.deformation_space)

        self.a_form = ufl.inner(ufl.TrialFunction(self.shape_problem.deformation_space), self.test_v) * self.ds
        self.l_form = ufl.inner(t_grad(self.coord, self.n), t_grad(self.test_v, self.n)) * self.ds
        self.loss_form = ufl.inner(self.kappa_curvature, self.kappa_curvature) * self.ds
        self.is_solved = False

    def compute_shape_derivative(self):
        if self.is_active:
            # 这是抄cashocs的表达式,似乎并不可行
            identity = ufl.Identity(self.shape_problem.domain.geometry.dim)
            coord_tang = t_grad(self.coord, self.n)
            shape_form = (
                    self.weight *
                    ufl.inner(
                        (identity - (coord_tang + coord_tang.T)) * t_grad(self.test_v, self.n),
                        t_grad(self.kappa_curvature, self.n)
                    ) * self.ds
                    + 0.5 * t_div(self.test_v, self.n) * t_div(self.kappa_curvature, self.n) * self.ds
            )
            # shape_form = ufl.derivative(self.weight * self.loss_form, self.coord, self.test_v)
            return shape_form
        else:
            return 0.0

    def compute_objective(self, force_update=False):
        value = 0.0
        if self.is_active:
            self._compute_curvature(force_update=force_update)
            curvature_val = AssembleUtils.assemble_scalar(dolfinx.fem.form(self.loss_form))
            value += 0.5 * self.weight * curvature_val
        return value

    def _compute_curvature(self, force_update=False, with_debug=False):
        if (not self.is_solved) or force_update:
            res_dict = LinearProblemSolver.solve_fea_petsc(
                comm=self.shape_problem.domain.comm,
                uh=self.kappa_curvature,
                a_form=dolfinx.fem.form(self.a_form),
                l_form=dolfinx.fem.form(self.l_form),
                bcs=[],
                mat_assemble_method='lift',
                ksp_option=self.ksp_option,
                with_debug=with_debug
            )
            self.is_solved = True
            return res_dict['is_solved']
        return True

    def update(self):
        self.is_solved = False
        self._compute_curvature()

    def update_weight(self, weight):
        raise NotImplementedError


class ShapeRegularizationPipe(object):
    def __init__(
            self,
            regularization_list: List[Union[VolumeRegularization, SurfaceAreaRegularization, BarycenterRegularization]]
    ):
        self.regularization_list = regularization_list

    def compute_objective(self) -> float:
        value = 0.0
        for reg in self.regularization_list:
            value += reg.compute_objective()
        return value

    def compute_shape_derivative(self) -> ufl.Form:
        value = 0.0
        for reg in self.regularization_list:
            value += reg.compute_shape_derivative()
        return value

    def update(self):
        for reg in self.regularization_list:
            reg.update()

    def get_cost_info(self) -> Dict:
        cost_dict = {}
        for reg in self.regularization_list:
            cost_dict[reg.name] = reg.compute_objective()
        return cost_dict
