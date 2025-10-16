from typing import Callable, Union, Literal

import dolfinx.mesh
import ufl
from ufl import inner, div, grad
from functools import partial
from ufl.core import expr
from basix.ufl import element

import numpy as np

from scripts_py.dolfinx_adjoint.base_type import (
    ControlProblemDataBase,
    ShapeProblemDataBase,
)
from scripts_py.eqn_solver import LinearProblemSolver
from scripts_py.mesh_utils import MeshUtils
from scripts_py.dolfinx_utils import DolfinUtils

from scripts_py.dolfinx_adjoint.cost_functions import LagrangianFunction
from scripts_py.dolfinx_adjoint.regularization_shape import ShapeRegularizationPipe
from scripts_py.dolfinx_adjoint.regularization_control import ControlRegularizationPipe


class ControlGradientProblem(object):
    def __init__(
        self,
        control_database: ControlProblemDataBase,
        lagrangian_function: LagrangianFunction,
        control_regularization_list: ControlRegularizationPipe,
        scalar_product_kwargs: dict[str, Union[str, float, Callable]] = {}
    ):
        self.has_solution = False
        self.control_database = control_database
        self.lagrangian_function = lagrangian_function
        self.control_reg_pipe = control_regularization_list

        if scalar_product_kwargs.get('scalar_product', None) is None:
            self.scalar_product = self._scalar_product
        else:
            assert isinstance(scalar_product_kwargs['scalar_product'], Callable), "scalar_product must be a callable function"
            self.scalar_product = scalar_product_kwargs['scalar_product']
        self._compute_gradient_equations()

    @staticmethod
    def _scalar_product(control: dolfinx.fem.Function):
        lhs_form = (
            ufl.inner(
                ufl.TrialFunction(control.function_space),
                ufl.TestFunction(control.function_space),
            ) * ufl.dx
        )
        return lhs_form

    def _compute_gradient_equations(self):
        # todo i don't fully understand the meaning of form lhs and rhs
        for control in self.control_database.controls:
            # 这里不能使用assemble_vec(lagrangian_function.derivative(control, v_TestFunction))
            # lagrangian_function.derivative(control, v_TestFunction)是weak form，它求取的不是control值，
            # 求取的ufl.TrialFunction(control.function_space)是指control的梯度
            grad_forms_rhs = self.lagrangian_function.derivative(
                control, ufl.TestFunction(control.function_space)  # type: ignore
            )
            grad_forms_rhs += self.control_reg_pipe.derivative()
            grad_forms_lhs = self.scalar_product(control)
            self.control_database.set_gradient_eq_form(
                control.name, grad_forms_lhs, grad_forms_rhs
            )

    def solve(
        self, comm, mat_assemble_method: Literal["identity_row", "lift"] = "lift", with_residual=False,
    ):
        if not self.has_solution:
            self.control_reg_pipe.update()

            for control in self.control_database.controls:
                control_par = self.control_database.control_pars[control.name]
                
                bcs = []  # todo should i add bcs here?
                res_dict = LinearProblemSolver.solve_fea_petsc(
                    comm=comm,
                    uh=control_par.grad,
                    a_form=control_par.gradient_lhs_dolfin,
                    l_form=control_par.gradient_rhs_dolfin,
                    bcs=bcs,
                    mat_assemble_method=mat_assemble_method,
                    ksp_option=control_par.ksp_option,
                    with_residual=with_residual,
                )
                if with_residual:
                    print(f'[Log]: {control.name} Gradient maxResidual:{res_dict["max_residual"]}')
                    
                if not res_dict["is_solved"]:
                    self.has_solution = False
                    print(f'[Error]: {control.name} Gradient fail converge with {res_dict["converged_reason"]}')
                    return self.has_solution

            self.has_solution = True
        return self.has_solution


class ShapeStiffness(object):
    '''
    In shape optimization, the Poincaré-Steklov operator maps boundary displacements to boundary tractions. 
    This code implements a simplified approach where:
    - The shape stiffness is modeled via a variable Lamé parameter μ
    - Different regions have different stiffness values (mu_free vs mu_fix)
    - This creates a stiffness distribution that guides shape deformation
    '''
    def __init__(
        self,
        domain: dolfinx.mesh.Mesh,
        cell_tags: dolfinx.mesh.MeshTags,
        facet_tags: dolfinx.mesh.MeshTags,
        bry_fixed_markers: list[int],
        bry_free_markers: list[int],
        mu_lame: dolfinx.fem.Function,
        mu_free: float = 1.0,
        mu_fix: float = 0.1,
    ):
        self.mu_lame = mu_lame
        self.mu_free = mu_free
        self.mu_fix = mu_fix
        
        self.domain = domain
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.bry_fixed_markers = bry_fixed_markers
        self.bry_free_markers = bry_free_markers
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1

        self.ds = MeshUtils.define_ds(self.domain, self.facet_tags)
        self.cg_function_space = self.mu_lame.function_space

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
        
        self.A_mu: dolfinx.fem.Form
        self.l_mu: dolfinx.fem.Form
        self.bcs: list[dolfinx.fem.DirichletBC] = []

    def _setup_form(self):
        if np.abs(self.mu_free - self.mu_fix) / self.mu_fix > 1e-2:
            self.inhomogeneous_mu = True

            phi = ufl.TrialFunction(self.cg_function_space)
            psi = ufl.TestFunction(self.cg_function_space)

            # weak form of laplace equation: Δμ = f, here f = 0
            self.A_mu = dolfinx.fem.form(inner(grad(phi), grad(psi)) * ufl.dx)  # type: ignore
            self.l_mu = dolfinx.fem.form(inner(dolfinx.fem.Constant(self.domain, 0.0), psi) * ufl.dx)  # type: ignore
            
            self.bcs = []
            for marker in self.bry_fixed_markers:
                bc_dofs = MeshUtils.extract_entity_dofs(
                    V=self.cg_function_space,
                    dim=self.fdim,
                    entity_idxs=MeshUtils.extract_entity_indices(
                        self.domain, style="facet", marker=marker, mesh_tags=self.facet_tags,
                    ),
                )
                self.bcs.append(
                    dolfinx.fem.dirichletbc(
                        dolfinx.fem.Constant(self.domain, self.mu_fix), bc_dofs, self.cg_function_space,  # type: ignore
                    )
                )

            for marker in self.bry_free_markers:
                bc_dofs = MeshUtils.extract_entity_dofs(
                    V=self.cg_function_space,
                    dim=self.fdim,
                    entity_idxs=MeshUtils.extract_entity_indices(
                        self.domain,
                        style="facet",
                        marker=marker,
                        mesh_tags=self.facet_tags,
                    )
                )
                self.bcs.append(
                    dolfinx.fem.dirichletbc(
                        dolfinx.fem.Constant(self.domain, self.mu_free), bc_dofs, self.cg_function_space,  # type: ignore
                    )
                )

    def compute(self, with_residual=False):
        if self.inhomogeneous_mu:
            res_dict = LinearProblemSolver.solve_fea_petsc(
                comm=self.domain.comm,
                uh=self.mu_lame,
                a_form=self.A_mu,
                l_form=self.l_mu,
                bcs=self.bcs,
                mat_assemble_method="lift",
                ksp_option=self.ksp_option,
                with_residual=with_residual,
            )
            if not res_dict["is_solved"]:
                raise ValueError("[Error]: Fail to solve shape stiffness deformation problem.")
        else:
            self.mu_lame.x.array[:] = self.mu_fix


class ShapeGradientProblem(object):
    def __init__(
        self,
        shape_database: ShapeProblemDataBase,
        lagrangian_function: LagrangianFunction,
        shape_regularization_list: ShapeRegularizationPipe,
        scalar_product_kwargs: dict[str, Union[str, float, Callable]] = {"method": "default"}
    ):
        self.has_solution = False

        self.shape_database = shape_database
        self.lagrangian_function = lagrangian_function
        self.shape_reg_pipe = shape_regularization_list
        
        self.scalar_product_kwargs = scalar_product_kwargs
        if scalar_product_kwargs.get('scalar_product', None) is None:
            self.scalar_product = partial(self._scalar_product, method_info=scalar_product_kwargs)
        else:
            assert isinstance(scalar_product_kwargs['scalar_product'], Callable), "scalar_product must be a callable function"
            self.scalar_product = scalar_product_kwargs['scalar_product']
            scalar_product_kwargs.setdefault('method', 'custom')

        self.trial_u = ufl.TrialFunction(self.shape_database.deformation_space)
        self.test_v = ufl.TestFunction(self.shape_database.deformation_space)
        self.coord = MeshUtils.define_coordinate(self.shape_database.domain)
        
        self.scalar_point_element = element("CG", self.shape_database.domain.basix_cell(), degree=1)  # type: ignore
        self.scalar_cell_element = element("DG", self.shape_database.domain.basix_cell(), degree=0)  # type: ignore
        self.dg_functon_space = dolfinx.fem.functionspace(self.shape_database.domain, self.scalar_cell_element)
        self.cg_functon_space = dolfinx.fem.functionspace(self.shape_database.domain, self.scalar_point_element)
        
        self.cell_volume_expr = DolfinUtils.create_expression(
            ufl.CellVolume(self.shape_database.domain), self.dg_functon_space  # type: ignore
        )

        self._compute_gradient_equations()

    def _compute_gradient_equations(self):
        grad_forms_rhs = self.lagrangian_function.derivative(self.coord, self.test_v)  # type: ignore
        grad_forms_rhs += self.shape_reg_pipe.derivative()
        grad_forms_lhs = self.scalar_product(self.trial_u, self.test_v)  # type: ignore
        self.shape_database.set_gradient_eq_form(grad_forms_lhs, grad_forms_rhs)

    # def update_gradient_equations(self):
    #     self._compute_gradient_equations()

    def _scalar_product(self, trial_u: ufl.Argument, test_v: ufl.Argument, method_info: dict):
        if method_info["method"] == "default":
            lhs_form = (
                inner((grad(trial_u)), (grad(test_v))) * ufl.dx
                + inner(trial_u, test_v) * ufl.dx
            )

        elif method_info["method"] == "Poincare-Steklov operator":
            """
            Based on `Schulz and Siebenborn, Computational Comparison of Surface Metrics for
            PDE Constrained Shape Optimization <https://doi.org/10.1515/cmam-2016-0009>`.
            """

            self.mu_lame = dolfinx.fem.Function(self.cg_functon_space, name="mu_lame")
            self.mu_lame.x.array[:] = 1.0  # type: ignore
            self.shape_stiffness = ShapeStiffness(
                mu_lame=self.mu_lame,  # type: ignore
                domain=self.shape_database.domain,
                cell_tags=method_info["cell_tags"],
                facet_tags=method_info["facet_tags"],
                bry_free_markers=method_info["bry_free_markers"],
                bry_fixed_markers=method_info["bry_fixed_markers"],
                mu_free=method_info.get("mu_free", 1.0),
                mu_fix=method_info.get("mu_fix", 0.1),
            )
            
            lambda_lame = method_info.get("lambda_lame", 1.0)
            damping_factor = method_info.get("damping_factor", 0.2)
            
            self.update_inhomogeneous = method_info["update_inhomogeneous"]
            self.cell_volumes = dolfinx.fem.Function(self.dg_functon_space, name="cell_volume")
            
            if self.update_inhomogeneous:
                self.cell_volumes.interpolate(self.cell_volume_expr)  # type: ignore
                vol_max = np.max(self.cell_volumes.x.array)  # type: ignore
                self.cell_volumes.x.array[:] = self.cell_volumes.x.array / vol_max  # type: ignore

                self.inhomogeneous_exponent = dolfinx.fem.Constant(
                    self.shape_database.domain, method_info["inhomogeneous_exponent"]
                )
            else:
                self.cell_volumes.x.array[:] = 1.0  # type: ignore
                self.inhomogeneous_exponent = dolfinx.fem.Constant(self.shape_database.domain, 1.0)  # type: ignore

            def eps(u: Union[dolfinx.fem.Function, ufl.TrialFunction, ufl.TestFunction]) -> expr.Expr:  # type: ignore
                """Computes the symmetric gradient of a vector field ``u``.
                Args:
                    u: A vector field
                Returns:
                    The symmetric gradient of ``u``
                """
                return dolfinx.fem.Constant(self.shape_database.domain, 0.5) * (grad(u) + grad(u).T)  # type: ignore

            constant = dolfinx.fem.Constant(self.shape_database.domain, 2.0)  # type: ignore
            lambda_constant = dolfinx.fem.Constant(self.shape_database.domain, lambda_lame)
            damping_constant = dolfinx.fem.Constant(self.shape_database.domain, damping_factor)

            lhs_form = (
                constant * self.mu_lame / ufl.elem_pow(self.cell_volumes, self.inhomogeneous_exponent)  # type: ignore
                * inner(eps(trial_u), eps(test_v)) * ufl.dx

                + lambda_constant / ufl.elem_pow(self.cell_volumes, self.inhomogeneous_exponent)  # type: ignore
                * div(trial_u) * div(test_v) * ufl.dx

                + damping_constant / ufl.elem_pow(self.cell_volumes, self.inhomogeneous_exponent)  # type: ignore
                * inner(trial_u, test_v) * ufl.dx
            )
            self.shape_stiffness.compute(with_residual=True)

        else:
            raise NotImplementedError

        return lhs_form

    def solve(self, comm, mat_assemble_method: Literal["identity_row", "lift"], with_residual=False):
        assert mat_assemble_method == "identity_row", "Shape gradient problem only support identity_row assemble method." \
            "lift will cause boundary shifting."
        
        if not self.has_solution:
            # --- each time update the shape regularization pipe at first
            self.shape_reg_pipe.update()

            res_dict = LinearProblemSolver.solve_fea_petsc(
                comm=comm,
                uh=self.shape_database.shape_grad,
                a_form=self.shape_database.gradient_lhs_dolfin,
                l_form=self.shape_database.gradient_rhs_dolfin,
                bcs=[be_cell.bc for be_cell in self.shape_database.bc_cells.values()],
                mat_assemble_method=mat_assemble_method,
                ksp_option=self.shape_database.gradient_ksp_option,
                with_residual=with_residual,
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

            if with_residual:
                print(f'[Log] Gradient maxResidual:{res_dict["max_residual"]}')

            if not res_dict["is_solved"]:
                self.has_solution = False
                print(f'[Error]: Shape gradient fail converge with {res_dict["converged_reason"]}')
                return self.has_solution
        
        self.has_solution = True
        return self.has_solution

    def update_scalar_product(self, with_residual=False):
        if self.scalar_product_kwargs['method'] == "Poincare-Steklov operator":
            self.shape_stiffness.compute(with_residual=with_residual)
            if self.update_inhomogeneous:
                self.cell_volumes.interpolate(self.cell_volume_expr)  # type: ignore
                vol_max = np.max(self.cell_volumes.x.array)  # type: ignore
                self.cell_volumes.x.array[:] = self.cell_volumes.x.array / vol_max  # type: ignore
