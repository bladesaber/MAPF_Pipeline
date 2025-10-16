import dolfinx
import ufl
from typing import Literal, Any

from scripts_py.dolfinx_adjoint.base_type import StateProblemDataBase
from scripts_py.eqn_solver import LinearProblemSolver, NonLinearProblemSolver


class StateProblem(object):
    def __init__(self, state_databases: list[StateProblemDataBase]):
        self.state_databases: list[StateProblemDataBase] = state_databases
        self.has_solution = False
        self._compute_state_equations()

    def _compute_state_equations(self):
        for database in self.state_databases:
            f_form = database.F_form

            replace_map: dict[Any, Any] = {}
            if database.is_linear:
                replace_map.update({
                    database.state: ufl.TrialFunction(database.state.function_space)  # type: ignore
                })
            replace_map.update({
                database.adjoint: ufl.TestFunction(database.adjoint.function_space)  # type: ignore
            })

            state_eq_form = ufl.replace(f_form, replace_map)
            if database.is_linear:
                state_lhs_form = ufl.lhs(state_eq_form)
                state_rhs_form = ufl.rhs(state_eq_form)
            else:
                state_lhs_form = state_eq_form
                state_rhs_form = 0.0
            
            database.set_state_eq_form(eqs_form=state_eq_form, lhs=state_lhs_form, rhs=state_rhs_form)  # type: ignore

    def solve(
        self,
        comm,
        jacobi_form: ufl.Form | None = None,
        state_func_space: dolfinx.fem.FunctionSpace | None = None,
        mat_assemble_method: Literal["identity_row", "lift"] = "lift",
        with_residual: bool = False,
    ):
        if not self.has_solution:
            for database in self.state_databases:
                if database.is_linear:
                    res_dict = LinearProblemSolver.solve_fea_petsc(
                        comm=comm,
                        uh=database.state,
                        a_form=database.state_lhs_dolfin,
                        l_form=database.state_rhs_dolfin,
                        bcs=[be_cell.bc for be_cell in database.bc_cells.values()],
                        mat_assemble_method=mat_assemble_method,
                        ksp_option=database.state_ksp_option,
                        with_residual=with_residual,
                    )
                    if with_residual:
                        print(f'[Log]: {database.name} maxResidual:{res_dict["max_residual"]}')

                    if not res_dict["is_solved"]:
                        self.has_solution = False
                        print(f'[Error]: {database.name} State fail converge with {res_dict["converged_reason"]}')
                        return self.has_solution

                else:
                    if jacobi_form is None:
                        if state_func_space is None:
                            state_func_space = database.state.function_space
                        jacobi_form = ufl.derivative(
                            database.state_lhs_form,
                            database.state,
                            ufl.TrialFunction(state_func_space),
                        )
                    jacobi_form_dolfin = dolfinx.fem.form(jacobi_form)  # type: ignore

                    res_dict = NonLinearProblemSolver.solve_fea_petsc(
                        comm=comm,
                        f_form=database.state_lhs_dolfin,
                        uh=database.state,
                        jacobi_form=jacobi_form_dolfin,  # type: ignore
                        bcs=[be_cell.bc for be_cell in database.bc_cells.values()],
                        mat_assemble_method=mat_assemble_method,
                        snes_setting=database.snes_option,
                        ksp_setting=database.state_ksp_option,
                        criterion={},
                        with_residual=with_residual,
                        with_monitor=False,
                    )
                    if with_residual:
                        print(f'[Log]: {database.name} maxResidual:{res_dict["max_residual"]}')

                    if not res_dict["is_solved"]:
                        self.has_solution = False
                        print(f'[Error]: {database.name} State fail converge with {res_dict["converged_reason"]}')
                        return self.has_solution

        self.has_solution = True
        return self.has_solution
