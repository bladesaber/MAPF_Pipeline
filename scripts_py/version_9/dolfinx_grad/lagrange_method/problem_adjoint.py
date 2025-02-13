import ufl
from typing import List, Literal

from scripts_py.version_9.dolfinx_grad.lagrange_method.type_database import GovDataBase
from scripts_py.version_9.dolfinx_grad.lagrange_method.cost_functions import LagrangianFunction
from scripts_py.version_9.dolfinx_grad.equation_solver import LinearProblemSolver


class AdjointProblem(object):
    def __init__(
            self,
            state_problems: List[GovDataBase],
            lagrangian_function: LagrangianFunction,
    ):
        self.state_problems = state_problems
        self.has_solution = False

        self._compute_adjoint_boundary_conditions()

        self.lagrangian_function = lagrangian_function
        self._compute_adjoint_equations(lagrangian_function)

    def _compute_adjoint_boundary_conditions(self):
        for problem in self.state_problems:
            problem.compute_adjoint_bc()

    def _compute_adjoint_equations(self, lagrangian_function: LagrangianFunction):
        """
        Adjoint System is always Linear
        Lagrangian_function: F = J + integral(adjoint_var * constraint_function(state_var, control_var))
        Total Variation: dF = Variation(F, state_var) + Variation(F, control_var)
        To solve Adjoint System, we need Variation(F, state_var) = 0
        """
        for problem in self.state_problems:
            # adjoint_eq_form is Variation(F, state_var) = 0
            adjoint_eq_form = lagrangian_function.derivative(
                problem.state, ufl.TestFunction(problem.adjoint.function_space)
            )  # 先对state变量求取residual equation
            adjoint_eq_form = ufl.replace(
                adjoint_eq_form, {problem.adjoint: ufl.TrialFunction(problem.adjoint.function_space)}
            )  # 再将residual equation的adjoint变量替换成ufl.TrialFunction以求取其值
            adjoint_eq_form_lhs = ufl.lhs(adjoint_eq_form)
            adjoint_eq_form_rhs = ufl.rhs(adjoint_eq_form)
            problem.set_adjoint_eq_form(adjoint_eq_form, adjoint_eq_form_lhs, adjoint_eq_form_rhs)

    def solve(
            self,
            comm,
            mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ):
        if not self.has_solution:
            for problem in self.state_problems:
                res_dict = LinearProblemSolver.solve_fea_petsc(
                    comm=comm,
                    uh=problem.adjoint,
                    a_form=problem.adjoint_lhs_dolfin,
                    l_form=problem.adjoint_rhs_dolfin,
                    bcs=problem.homogenize_bcs,
                    mat_assemble_method=mat_assemble_method,
                    ksp_option=problem.adjoint_ksp_option,
                    with_debug=with_debug
                )
                if with_debug:
                    print(f'[Info] {problem.name} Adjoint maxError:{res_dict["max_error"]}')

                if not res_dict['is_solved']:
                    self.has_solution = False
                    print(f'[Info]: {problem.name} Adjoint problem fail converge with {res_dict["converged_reason"]}')
                    return self.has_solution

            self.has_solution = True
        return self.has_solution
