import dolfinx.mesh
import numpy as np
import ufl
from typing import Callable, Union, Dict, List, Literal
from ufl import inner, div, grad
from functools import partial
from ufl.core import expr
from basix.ufl import element

from scripts_py.version_9.dolfinx_grad.lagrange_method.cost_functions import LagrangianFunction
from scripts_py.version_9.dolfinx_grad.lagrange_method.type_database import ControlDataBase, ShapeDataBase
from scripts_py.version_9.dolfinx_grad.equation_solver import LinearProblemSolver
from scripts_py.version_9.dolfinx_grad.lagrange_method.shape_regularization import ShapeRegularizationPipe
from scripts_py.version_9.dolfinx_grad.lagrange_method.control_regularization import ControlRegularizationPipe

from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils
from scripts_py.version_9.dolfinx_grad.dolfinx_utils import DolfinUtils, AssembleUtils


class ControlGradientProblem(object):
    def __init__(
            self,
            control_problems: ControlDataBase,
            lagrangian_function: LagrangianFunction,
            control_regularization_list: ControlRegularizationPipe,
            scalar_product: Callable = None
    ):
        self.control_problems = control_problems
        self.lagrangian_function = lagrangian_function

        self.has_solution = False
        self.control_reg_pipe = control_regularization_list

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
            # 这里不能使用assemble_vec(lagrangian_function.derivative(control, v_TestFunction))
            # lagrangian_function.derivative(control, v_TestFunction)是weak form，它求取的不是control值，
            # 求取的ufl.TrialFunction(control.function_space)是指control的梯度
            grad_forms_rhs = self.lagrangian_function.derivative(
                control, ufl.TestFunction(control.function_space)
            )
            grad_forms_rhs += self.control_reg_pipe.compute_control_derivative()
            grad_forms_lhs = self.scalar_product(control)
            self.control_problems.set_gradient_eq_form(control.name, grad_forms_lhs, grad_forms_rhs)

    def solve(
            self,
            comm,
            mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ):
        if not self.has_solution:
            self.control_reg_pipe.update()

            for control in self.control_problems.controls:
                control_name = control.name

                eq_forms_dict = self.control_problems.control_eq_forms[control_name]
                lhs_form = eq_forms_dict['gradient_lhs_dolfin']
                rhs_form = eq_forms_dict['gradient_rhs_dolfin']
                bcs = self.control_problems.bcs.get(control_name, [])
                ksp_option = self.control_problems.gradient_ksp_options[control_name]
                grad_fun = self.control_problems.control_grads[control_name]

                res_dict = LinearProblemSolver.solve_fea_petsc(
                    comm=comm,
                    uh=grad_fun,
                    a_form=lhs_form,
                    l_form=rhs_form,
                    bcs=bcs,
                    mat_assemble_method=mat_assemble_method,
                    ksp_option=ksp_option,
                    with_debug=with_debug
                )
                if not res_dict['is_solved']:
                    self.has_solution = False
                    print(f'[Info]: {control.name} Gradient fail converge with {res_dict["converged_reason"]}')
                    return self.has_solution

            self.has_solution = True
        return self.has_solution


class ShapeStiffness(object):
    def __init__(
            self,
            mu_lame: dolfinx.fem.Function,
            domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            bry_fixed_markers: List[int],
            bry_free_markers: List[int],
            mu_free: float = 1.0,
            mu_fix: float = 0.1,
    ):
        self.mu_lame = mu_lame
        self.domain = domain
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.bry_fixed_markers = bry_fixed_markers
        self.bry_free_markers = bry_free_markers
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1

        self.ds = MeshUtils.define_ds(self.domain, self.facet_tags)
        self.cg_function_space = self.mu_lame.function_space

        self.mu_free = mu_free
        self.mu_fix = mu_fix
        self.inhomogeneous_mu = False
        self._setup_form()

        self.ksp_option = {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_mat_solver_type": "boomeramg",
            # "ksp_rtol": 1e-16,
            # "ksp_atol": 1e-50,
            # "ksp_max_it": 100,
        }

    def _setup_form(self):
        if np.abs(self.mu_free - self.mu_fix) / self.mu_fix > 1e-2:
            self.inhomogeneous_mu = True

            phi = ufl.TrialFunction(self.cg_function_space)
            psi = ufl.TestFunction(self.cg_function_space)

            self.A_mu = dolfinx.fem.form(inner(grad(phi), grad(psi)) * ufl.dx)
            self.l_mu = dolfinx.fem.form(inner(dolfinx.fem.Constant(self.domain, 0.0), psi) * ufl.dx)
            self.bcs = []

            for marker in self.bry_fixed_markers:
                bc_dofs = MeshUtils.extract_entity_dofs(
                    V=self.cg_function_space, dim=self.fdim,
                    entity_idxs=MeshUtils.extract_entity_indices(
                        self.domain, style='facet', marker=marker, mesh_tags=self.facet_tags
                    )
                )
                bc = dolfinx.fem.dirichletbc(
                    dolfinx.fem.Constant(self.domain, self.mu_fix), bc_dofs, self.cg_function_space
                )
                self.bcs.append(bc)

            for marker in self.bry_free_markers:
                bc_dofs = MeshUtils.extract_entity_dofs(
                    V=self.cg_function_space, dim=self.fdim,
                    entity_idxs=MeshUtils.extract_entity_indices(
                        self.domain, style='facet', marker=marker, mesh_tags=self.facet_tags
                    )
                )
                bc = dolfinx.fem.dirichletbc(
                    dolfinx.fem.Constant(self.domain, self.mu_free), bc_dofs, self.cg_function_space
                )
                self.bcs.append(bc)

    def compute(self, with_debug=False):
        if self.inhomogeneous_mu:
            res_dict = LinearProblemSolver.solve_fea_petsc(
                comm=self.domain.comm,
                uh=self.mu_lame,
                a_form=self.A_mu,
                l_form=self.l_mu,
                bcs=self.bcs,
                mat_assemble_method='lift',
                ksp_option=self.ksp_option,
                with_debug=with_debug
            )
            if not res_dict['is_solved']:
                raise ValueError('[Error]: Fail to solve shape stiffness deformation problem.')
        else:
            self.mu_lame.vector.set(self.mu_fix)


class ShapeGradientProblem(object):
    def __init__(
            self,
            shape_problem: ShapeDataBase,
            lagrangian_function: LagrangianFunction,
            shape_regularization_list: ShapeRegularizationPipe,
            scalar_product: Callable = None,
            scalar_product_method: Dict = {'method': "default"}
    ):
        self.has_solution = False

        self.shape_problem = shape_problem
        self.lagrangian_function = lagrangian_function
        self.shape_reg_pipe = shape_regularization_list

        self.scalar_product_method = scalar_product_method
        if scalar_product is None:
            self.scalar_product = partial(self._scalar_product, method_info=scalar_product_method)
        else:
            self.scalar_product = scalar_product

        self.trial_u = ufl.TrialFunction(self.shape_problem.deformation_space)
        self.test_v = ufl.TestFunction(self.shape_problem.deformation_space)
        self.coord = MeshUtils.define_coordinate(self.shape_problem.domain)
        self.scalar_point_element = element("CG", self.shape_problem.domain.basix_cell(), degree=1)
        self.scalar_cell_element = element("DG", self.shape_problem.domain.basix_cell(), degree=0)
        self.dg_functon_space = dolfinx.fem.functionspace(self.shape_problem.domain, self.scalar_cell_element)
        self.cg_functon_space = dolfinx.fem.functionspace(self.shape_problem.domain, self.scalar_point_element)
        self.cell_volume_expr = DolfinUtils.create_expression(
            ufl.CellVolume(self.shape_problem.domain), self.dg_functon_space
        )

        self._compute_gradient_equations()

    def _compute_gradient_equations(self):
        grad_forms_rhs = self.lagrangian_function.derivative(self.coord, self.test_v)
        grad_forms_rhs += self.shape_reg_pipe.compute_shape_derivative()
        grad_forms_lhs = self.scalar_product(self.trial_u, self.test_v)
        self.shape_problem.set_gradient_eq_form(grad_forms_lhs, grad_forms_rhs)

    def update_gradient_equations(self):
        self._compute_gradient_equations()

    def _scalar_product(self, trial_u: ufl.Argument, test_v: ufl.Argument, method_info: Dict):
        if method_info['method'] == "default":
            lhs_form = inner((grad(trial_u)), (grad(test_v))) * ufl.dx + inner(trial_u, test_v) * ufl.dx

        elif method_info['method'] == "Poincare-Steklov operator":
            """
            Based on `Schulz and Siebenborn, Computational Comparison of Surface Metrics for
            PDE Constrained Shape Optimization <https://doi.org/10.1515/cmam-2016-0009>`_.
            """

            self.mu_lame = dolfinx.fem.Function(self.cg_functon_space, name='mu_lame')
            self.mu_lame.vector.set(1.0)

            lambda_lame = method_info.get("lambda_lame", 1.0)
            damping_factor = method_info.get("damping_factor", 0.2)

            self.shape_stiffness = ShapeStiffness(
                mu_lame=self.mu_lame,
                domain=self.shape_problem.domain,
                cell_tags=method_info['cell_tags'],
                facet_tags=method_info['facet_tags'],
                bry_free_markers=method_info['bry_free_markers'],
                bry_fixed_markers=method_info['bry_fixed_markers'],
                mu_free=method_info.get('mu_free', 1.0),
                mu_fix=method_info.get('mu_fix', 1.0)
            )
            self.update_inhomogeneous = method_info['update_inhomogeneous']

            self.cell_volumes = dolfinx.fem.Function(self.dg_functon_space, name='cell_volume')
            if method_info["use_inhomogeneous"]:
                self.cell_volumes.interpolate(self.cell_volume_expr)
                vol_max_idx, vol_max = self.cell_volumes.vector.max()
                self.cell_volumes.vector.scale(1.0 / vol_max)

                self.inhomogeneous_exponent = dolfinx.fem.Constant(
                    self.shape_problem.domain, method_info["inhomogeneous_exponent"]
                )
            else:
                self.cell_volumes.vector.set(1.0)
                self.inhomogeneous_exponent = dolfinx.fem.Constant(self.shape_problem.domain, 1.0)

            def eps(u: Union[dolfinx.fem.Function, ufl.TrialFunction, ufl.TestFunction]) -> expr.Expr:
                """Computes the symmetric gradient of a vector field ``u``.
                Args:
                    u: A vector field
                Returns:
                    The symmetric gradient of ``u``
                """
                return dolfinx.fem.Constant(self.shape_problem.domain, 0.5) * (grad(u) + grad(u).T)

            constant = dolfinx.fem.Constant(self.shape_problem.domain, 2.0)
            lambda_constant = dolfinx.fem.Constant(self.shape_problem.domain, lambda_lame)
            damping_constant = dolfinx.fem.Constant(self.shape_problem.domain, damping_factor)

            lhs_form = (
                    constant * self.mu_lame / ufl.elem_pow(self.cell_volumes, self.inhomogeneous_exponent)
                    * inner(eps(trial_u), eps(test_v)) * ufl.dx

                    + lambda_constant / ufl.elem_pow(self.cell_volumes, self.inhomogeneous_exponent)
                    * div(trial_u) * div(test_v) * ufl.dx

                    + damping_constant / ufl.elem_pow(self.cell_volumes, self.inhomogeneous_exponent)
                    * inner(trial_u, test_v) * ufl.dx
            )
            self.shape_stiffness.compute(with_debug=True)

        else:
            raise NotImplementedError

        return lhs_form

    def solve(
            self,
            comm,
            mat_assemble_method: Literal['Identity_row', 'lift'],
            with_debug=False
    ):
        if not self.has_solution:
            self.shape_reg_pipe.update()

            res_dict = LinearProblemSolver.solve_fea_petsc(
                comm=comm,
                uh=self.shape_problem.shape_grad,
                a_form=self.shape_problem.gradient_lhs_dolfin,
                l_form=self.shape_problem.gradient_rhs_dolfin,
                bcs=self.shape_problem.bcs,
                mat_assemble_method=mat_assemble_method,
                ksp_option=self.shape_problem.gradient_ksp_option,
                with_debug=with_debug
            )

            # # ------
            # """
            # Ref: Automated shape differentiation in the Unified Form Language
            # """
            # dJdx = dolfinx.fem.Function(self.shape_problem.deformation_space)
            # dJdx.vector.aypx(0.0, AssembleUtils.assemble_vec(self.shape_problem.gradient_rhs_dolfin))
            # lhs_form = inner((grad(self.trial_u)), (grad(self.test_v))) * ufl.dx
            # rhs_form = ufl.inner(dJdx, self.test_v) * ufl.dx
            # res_dict = LinearProblemSolver.solve_fea_petsc(
            #     comm=comm,
            #     uh=self.shape_problem.shape_grad,
            #     a_form=dolfinx.fem.form(lhs_form),
            #     l_form=dolfinx.fem.form(rhs_form),
            #     bcs=self.shape_problem.bcs,
            #     mat_assemble_method=mat_assemble_method,
            #     ksp_option=self.shape_problem.gradient_ksp_option,
            #     with_debug=True
            # )
            # # ------

            if with_debug:
                print(f'[Info] Gradient maxError:{res_dict["max_error"]}')

            if not res_dict['is_solved']:
                self.has_solution = False
                print(f'[Info]: Shape gradient fail converge with {res_dict["converged_reason"]}')
                return self.has_solution
            self.has_solution = True
        return self.has_solution

    def update_scalar_product(self, with_debug):
        if self.scalar_product_method['method'] == 'Poincare-Steklov operator':
            self.shape_stiffness.compute(with_debug=with_debug)

            if self.update_inhomogeneous:
                self.cell_volumes.interpolate(self.cell_volume_expr)
                vol_max_idx, vol_max = self.cell_volumes.vector.max()
                self.cell_volumes.vector.scale(1.0 / vol_max)

