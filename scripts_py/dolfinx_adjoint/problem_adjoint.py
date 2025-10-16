import ufl
from typing import List, Literal

from scripts_py.dolfinx_adjoint.base_type import StateProblemDataBase
from scripts_py.dolfinx_adjoint.cost_functions import LagrangianFunction
from scripts_py.eqn_solver import LinearProblemSolver


class AdjointProblem(object):
    def __init__(
        self,
        state_databases: list[StateProblemDataBase],
        lagrangian_function: LagrangianFunction,
    ):
        self.state_databases = state_databases
        self.has_solution = False

        self._compute_adjoint_boundary_conditions()

        self.lagrangian_function = lagrangian_function
        self._compute_adjoint_equations(lagrangian_function)

    def _compute_adjoint_boundary_conditions(self):
        for database in self.state_databases:
            database.compute_adjoint_bc()

    def _compute_adjoint_equations(self, lagrangian_function: LagrangianFunction):
        """
        Adjoint System is always Linear
        Lagrangian_function: F = J + integral(adjoint_var * constraint_function(state_var, control_var))
        Total Variation: dF = Variation(F, state_var) + Variation(F, adjoint_var) + Variation(F, control_var)
        To solve Adjoint System, we need Variation(F, state_var) + Variation(F, adjoint_var) = 0
        Since Variation(F, adjoint_var) always equals 0 because it come from equation constraint, we have Variation(F, state_var) = 0
        """
        for database in self.state_databases:
            adjoint_eq_form = lagrangian_function.derivative(
                database.state, ufl.TestFunction(database.adjoint.function_space)  # type: ignore
            )  # 先对state变量求取residual equation
            
            adjoint_eq_form = ufl.replace(
                adjoint_eq_form,
                {
                    database.adjoint: ufl.TrialFunction(database.adjoint.function_space)  # type: ignore
                },
            )  # 再将residual equation的adjoint变量替换成ufl.TrialFunction以求取其值
            
            adjoint_eq_form_lhs = ufl.lhs(adjoint_eq_form)
            adjoint_eq_form_rhs = ufl.rhs(adjoint_eq_form)
            database.set_adjoint_eq_form(adjoint_eq_form, adjoint_eq_form_lhs, adjoint_eq_form_rhs)  # type: ignore

    def solve(
        self,
        comm,
        mat_assemble_method: Literal["identity_row", "lift"] = "lift",
        with_residual: bool = False,
    ):
        if not self.has_solution:
            for database in self.state_databases:
                res_dict = LinearProblemSolver.solve_fea_petsc(
                    comm=comm,
                    uh=database.adjoint,
                    a_form=database.adjoint_lhs_dolfin,
                    l_form=database.adjoint_rhs_dolfin,
                    bcs=[be_cell.homogenize_bc for be_cell in database.bc_cells.values()],
                    mat_assemble_method=mat_assemble_method,
                    ksp_option=database.adjoint_ksp_option,
                    with_residual=with_residual,
                )
                if with_residual:
                    print(f'[Log]: {database.name} Adjoint maxResidual:{res_dict["max_residual"]}')

                if not res_dict["is_solved"]:
                    self.has_solution = False
                    print(f'[Error]: {database.name} Adjoint problem fail converge with {res_dict["converged_reason"]}')
                    return self.has_solution

        self.has_solution = True
        return self.has_solution
