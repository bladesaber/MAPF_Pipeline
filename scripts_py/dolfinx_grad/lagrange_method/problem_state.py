import dolfinx
import ufl
from typing import List, Literal

from scripts_py.version_9.dolfinx_grad.lagrange_method.type_database import GovDataBase
from scripts_py.version_9.dolfinx_grad.equation_solver import LinearProblemSolver, NonLinearProblemSolver


class StateProblem(object):
    def __init__(self, state_problems: List[GovDataBase]):
        self.state_problems = state_problems
        self.has_solution = False
        self._compute_state_equations()

    def _compute_state_equations(self):
        for problem in self.state_problems:
            f_form = problem.F_form

            replace_map = {}
            if problem.is_linear:
                replace_map.update({problem.state: ufl.TrialFunction(problem.state.function_space)})
            replace_map.update({problem.adjoint: ufl.TestFunction(problem.adjoint.function_space)})

            state_eq_form = ufl.replace(f_form, replace_map)
            if problem.is_linear:
                state_lhs_form = ufl.lhs(state_eq_form)
                state_rhs_form = ufl.rhs(state_eq_form)
            else:
                state_lhs_form = state_eq_form
                state_rhs_form = 0.0

            problem.set_state_eq_form(eqs_form=state_eq_form, lhs=state_lhs_form, rhs=state_rhs_form)

    def solve(
            self,
            comm,
            jacobi_form: ufl.Form = None,
            state_func_space: dolfinx.fem.FunctionSpace = None,
            mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ):
        if not self.has_solution:
            for problem in self.state_problems:
                if problem.is_linear:
                    res_dict = LinearProblemSolver.solve_fea_petsc(
                        comm=comm,
                        uh=problem.state,
                        a_form=problem.state_lhs_dolfin,
                        l_form=problem.state_rhs_dolfin,
                        bcs=problem.bcs,
                        mat_assemble_method=mat_assemble_method,
                        ksp_option=problem.state_ksp_option,
                        with_debug=with_debug
                    )
                    if with_debug:
                        print(f'[Info] {problem.name} State maxError:{res_dict["max_error"]}')

                    if not res_dict['is_solved']:
                        self.has_solution = False
                        print(f'[Info]: {problem.name} State fail converge with {res_dict["converged_reason"]}')
                        return self.has_solution

                else:
                    if jacobi_form is None:
                        if state_func_space is None:
                            state_func_space = problem.state.function_space
                        jacobi_form = ufl.derivative(
                            problem.state_lhs_form, problem.state, ufl.TrialFunction(state_func_space)
                        )
                    jacobi_dolfin = dolfinx.fem.form(jacobi_form)

                    res_dict = NonLinearProblemSolver.solve_fea_petsc(
                        comm=comm,
                        f_form=problem.state_lhs_dolfin,
                        uh=problem.state,
                        jacobi_form=jacobi_dolfin,
                        bcs=problem.bcs,
                        mat_assemble_method=mat_assemble_method,
                        snes_setting=problem.snes_option,
                        ksp_setting=problem.state_ksp_option,
                        criterion={},
                        with_debug=with_debug,
                        with_monitor=False
                    )
                    if with_debug:
                        print(f'[Info] {problem.name} State maxError:{res_dict["max_error"]}')

                    if not res_dict['is_solved']:
                        self.has_solution = False
                        print(f'[Info]: {problem.name} State fail converge with {res_dict["converged_reason"]}')
                        return self.has_solution

            self.has_solution = True
        return self.has_solution
