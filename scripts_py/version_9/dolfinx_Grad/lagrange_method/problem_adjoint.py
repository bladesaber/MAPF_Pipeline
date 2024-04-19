import ufl
from typing import List

from .type_database import GovDataBase
from .cost_functions import LagrangianFunction
from ..equation_solver import LinearProblemSolver


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
            )
            adjoint_eq_form = ufl.replace(
                adjoint_eq_form, {problem.adjoint: ufl.TrialFunction(problem.adjoint.function_space)}
            )
            adjoint_eq_form_lhs = ufl.lhs(adjoint_eq_form)
            adjoint_eq_form_rhs = ufl.rhs(adjoint_eq_form)
            problem.set_adjoint_eq_form(adjoint_eq_form, adjoint_eq_form_lhs, adjoint_eq_form_rhs)

    def solve(self, comm, check_converged=True, **kwargs):
        if not self.has_solution:
            for problem in self.state_problems:
                res_dict = LinearProblemSolver.solve_by_petsc_form(
                    comm=comm,
                    uh=problem.adjoint,
                    a_form=problem.adjoint_eq_form_lhs,
                    L_form=problem.adjoint_eq_form_rhs,
                    bcs=problem.homogenize_bcs,
                    ksp_option=problem.adjoint_ksp_option,
                    **kwargs
                )

                if check_converged:
                    if not LinearProblemSolver.is_converged(res_dict['converged_reason']):
                        raise ValueError(f"[ERROR] Adjoint Problem KSP Fail Converge {res_dict['converged_reason']}")

                if kwargs.get('with_debug', False):
                    print(f"[DEBUG AdjointSystem {problem.name}]: max_error:{res_dict['max_error']:.6f} "
                          f"cost_time:{res_dict['cost_time']:.2f}")
                    assert res_dict['max_error'] < 1e-6

            self.has_solution = True
        return self.has_solution
