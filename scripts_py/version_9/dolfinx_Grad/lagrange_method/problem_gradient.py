import dolfinx.mesh
import numpy as np
import ufl
from typing import Callable, Union
from ufl import inner, div, grad
from functools import partial
from ufl.core import expr

from .cost_functions import LagrangianFunction
from .type_database import ControlDataBase, ShapeDataBase
from ..equation_solver import LinearProblemSolver
from ..dolfinx_utils import MeshUtils, AssembleUtils, UFLUtils
from .shape_regularization import ShapeRegularization


class ControlGradientProblem(object):
    def __init__(
            self,
            control_problems: ControlDataBase,
            lagrangian_function: LagrangianFunction,
            scalar_product: Callable = None
    ):
        self.has_solution = False

        self.control_problems = control_problems
        self.lagrangian_function = lagrangian_function

        # Used to express lhs(Left Hand Side)
        if scalar_product is None:
            self.scalar_product = self._scalar_product
        else:
            self.scalar_product = scalar_product

        self._compute_gradient_equations()

    @staticmethod
    def _scalar_product(control: dolfinx.fem.Function):
        lhs_form = ufl.inner(
            ufl.TrialFunction(control.function_space),
            ufl.TestFunction(control.function_space)
        ) * ufl.dx
        return lhs_form

    def _compute_gradient_equations(self):
        for control in self.control_problems.controls:
            grad_forms_rhs = self.lagrangian_function.derivative(
                control, ufl.TestFunction(control.function_space)
            )
            grad_forms_lhs = self.scalar_product(control)

            self.control_problems.set_gradient_eq_form(control.name, grad_forms_lhs, grad_forms_rhs)

    def solve(self, comm, **kwargs):
        if not self.has_solution:
            for control in self.control_problems.controls:
                control_name = control.name

                eq_forms = self.control_problems.control_eq_forms[control_name]
                lhs_form = eq_forms['gradient_eq_dolfinx_form_lhs']
                rhs_form = eq_forms['gradient_eq_dolfinx_form_rhs']
                bcs = self.control_problems.bcs.get(control_name, [])
                ksp_option = self.control_problems.gradient_ksp_options[control_name]
                grad_fun = self.control_problems.control_grads[control_name]

                res_dict = LinearProblemSolver.solve_by_petsc_form(
                    comm=comm,
                    uh=grad_fun,
                    a_form=lhs_form,
                    L_form=rhs_form,
                    bcs=bcs,
                    ksp_option=ksp_option,
                    **kwargs
                )

            self.has_solution = True
        return self.has_solution


class ShapeGradientProblem(object):
    def __init__(
            self,
            shape_problem: ShapeDataBase,
            lagrangian_function: LagrangianFunction,
            shape_regulariztions: ShapeRegularization,
            scalar_product: Callable = None,
            scalar_product_method: str = 'default'
    ):
        self.has_solution = False

        self.shape_problem = shape_problem
        self.lagrangian_function = lagrangian_function
        self.shape_regulariztions = shape_regulariztions

        if scalar_product is None:
            self.scalar_product = partial(self._scalar_product, method=scalar_product_method)
        else:
            self.scalar_product = scalar_product

        self.trial_u = ufl.TrialFunction(self.shape_problem.deformation_space)
        self.test_v = ufl.TestFunction(self.shape_problem.deformation_space)
        self.coodr = MeshUtils.define_coordinate(self.shape_problem.domain)

        self._compute_gradient_equations()

    def _compute_gradient_equations(self):
        grad_forms_rhs = self.lagrangian_function.derivative(self.coodr, self.test_v)
        grad_forms_rhs += self.shape_regulariztions.compute_shape_derivative()
        grad_forms_lhs = self.scalar_product(self.trial_u, self.test_v)
        self.shape_problem.set_gradient_eq_form(grad_forms_lhs, grad_forms_rhs)

    def _scalar_product(self, trial_u: ufl.Argument, test_v: ufl.Argument, method: str):
        if method == 'default':
            lhs_form = inner((grad(trial_u)), (grad(test_v))) * ufl.dx + inner(trial_u, test_v) * ufl.dx

        elif method == 'surface_metrics':
            # TODO Fail, it is wrong Now
            """
            Based on `Schulz and Siebenborn, Computational Comparison of Surface Metrics for
            PDE Constrained Shape Optimization <https://doi.org/10.1515/cmam-2016-0009>`_.
            """

            mu_lame = dolfinx.fem.Constant(self.shape_problem.domain, 1.0)
            inhomogeneous_exponent = dolfinx.fem.Constant(self.shape_problem.domain, 1.0)
            constant = dolfinx.fem.Constant(self.shape_problem.domain, 2.0)

            DG = dolfinx.fem.FunctionSpace(self.shape_problem.domain, ('DG', 0))
            cell_volume = dolfinx.fem.Function(DG, name='cell_volume')
            cell_volume.interpolate(UFLUtils.create_expression(ufl.CellVolume(self.shape_problem.domain), DG))
            vol_max_idx, vol_max = cell_volume.vector.max()
            cell_volume.vector.scale(1.0 / vol_max)

            def eps(u: Union[dolfinx.fem.Function, ufl.TrialFunction, ufl.TestFunction]) -> expr.Expr:
                """Computes the symmetric gradient of a vector field ``u``.
                Args:
                    u: A vector field
                Returns:
                    The symmetric gradient of ``u``

                """
                return 0.5 * (grad(u) + grad(u).T)

            lhs_form = constant * mu_lame / np.power(cell_volume, inhomogeneous_exponent) * \
                       inner(eps(trial_u), eps(test_v)) * ufl.dx

        else:
            raise NotImplementedError

        return lhs_form

    def solve(self, comm, **kwargs):
        if not self.has_solution:
            self.shape_regulariztions.update()

            res_dict = LinearProblemSolver.solve_by_petsc_form(
                comm=comm,
                uh=self.shape_problem.shape_grad,
                a_form=self.shape_problem.gradient_eq_form_lhs,
                L_form=self.shape_problem.gradient_eq_form_rhs,
                bcs=self.shape_problem.bcs,
                ksp_option=self.shape_problem.gradient_ksp_option,
                with_debug=False,
                **kwargs
            )

            # # ------
            # """
            # Ref: Automated shape differentation in the Unified Form Language
            # """
            # dJdx = dolfinx.fem.Function(self.shape_problem.deformation_space)
            # dJdx.vector.aypx(0.0, AssembleUtils.assemble_vec(self.shape_problem.gradient_eq_dolfinx_form_rhs))
            # lhs_form = inner((grad(self.trail_u)), (grad(self.test_v))) * ufl.dx
            # rhs_form = ufl.inner(dJdx, self.test_v) * ufl.dx
            # res_dict = LinearProblemSolver.solve_by_petsc_form(
            #     comm=comm,
            #     uh=self.shape_problem.shape_grad,
            #     a_form=lhs_form,
            #     L_form=rhs_form,
            #     bcs=self.shape_problem.bcs,
            #     ksp_option=self.shape_problem.gradient_ksp_option,
            #     with_debug=True,
            #     **kwargs
            # )
            # # ------

            self.has_solution = True
        return self.has_solution
