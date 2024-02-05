import dolfinx
import ufl
from typing import List

from .type_database import GovDataBase
from ..equation_solver import LinearProblemSolver, NonLinearProblemSolver


class StateProblem(object):
    def __init__(self, state_problems: List[GovDataBase]):
        self.state_problems = state_problems
        self.has_solution = False
        self._compute_state_equations()

    def _compute_state_equations(self):
        for problem in self.state_problems:
            F_form = problem.F_form

            replace_map = {}
            if problem.is_linear:
                replace_map.update({problem.state: ufl.TrialFunction(problem.state.function_space)})
            replace_map.update({problem.adjoint: ufl.TestFunction(problem.adjoint.function_space)})

            state_eq_form = ufl.replace(F_form, replace_map)

            if problem.is_linear:
                state_eq_form_lhs = ufl.lhs(state_eq_form)
                state_eq_form_rhs = ufl.rhs(state_eq_form)

            else:
                state_eq_form_lhs = state_eq_form
                state_eq_form_rhs = 0.0

            problem.set_state_eq_form(eqs_form=state_eq_form, lhs=state_eq_form_lhs, rhs=state_eq_form_rhs)

    def solve(self, comm, **kwargs):
        if not self.has_solution:
            for problem in self.state_problems:
                if problem.is_linear:
                    res_dict = LinearProblemSolver.solve_by_petsc_form(
                        comm=comm,
                        uh=problem.state,
                        a_form=problem.state_eq_dolfinx_form_lhs,
                        L_form=problem.state_eq_dolfinx_form_rhs,
                        bcs=problem.bcs,
                        ksp_option=problem.state_ksp_option,
                        **kwargs
                    )

                    if kwargs.get('with_debug', False):
                        print(f"[DEBUG]: max_error:{res_dict['max_error']:.6f} cost_time:{res_dict['cost_time']:.2f}")

                else:
                    jacobi_form = kwargs.get('jacobi_form', None)
                    if jacobi_form is None:
                        function_space = kwargs.get('function_space', None)
                        if function_space is None:
                            function_space = problem.state.function_space

                        jacobi_form = ufl.derivative(
                            problem.state_eq_form_lhs, problem.state, ufl.TrialFunction(function_space)
                        )

                    res_dict = NonLinearProblemSolver.solve_by_petsc(
                        F_form=problem.state_eq_form_lhs,
                        uh=problem.state,
                        jacobi_form=jacobi_form,
                        bcs=problem.bcs,
                        comm=comm,
                        ksp_option=problem.state_ksp_option,
                        **kwargs
                    )

                    if not res_dict['is_converge']:
                        raise ValueError("[DEBUG###] Nonlinear Solver Fail")

            self.has_solution = True

        return self.has_solution
