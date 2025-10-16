from typing import Callable, Union, Literal

import dolfinx
import ufl

import numpy as np

from scripts_py.dolfinx_adjoint.cost_functions import (
    CostFunctional,
    IntegralFunctional,
    ScalarTrackingFunctional,
    MinMaxFunctional,
    LagrangianFunction,
)
from scripts_py.dolfinx_adjoint.base_type import (
    StateProblemDataBase,
    ControlProblemDataBase,
    ShapeProblemDataBase,
)
from scripts_py.dolfinx_adjoint.problem_state import StateProblem
from scripts_py.dolfinx_adjoint.problem_adjoint import AdjointProblem
from scripts_py.dolfinx_adjoint.problem_gradient import (
    ControlGradientProblem,
    ShapeGradientProblem,
)
from scripts_py.dolfinx_adjoint.regularization_shape import ShapeRegularizationPipe
from scripts_py.dolfinx_adjoint.regularization_control import ControlRegularizationPipe


class ControlOptimizer(object):
    def __init__(
        self,
        state_problem: StateProblem,
        control_database: ControlProblemDataBase,
        cost_functional_list: list[CostFunctional],
        control_regularization_pipe: ControlRegularizationPipe,
        scalar_product_kwargs: dict[str, Union[str, float, Callable]] = {},
    ):
        self.state_problem = state_problem
        self.control_database = control_database
        self.cost_functional_list = cost_functional_list
        self.reg_pipe = control_regularization_pipe

        f_forms = [
            state_database.F_form
            for state_database in self.state_problem.state_databases
        ]
        self.state_list = [
            database.state for database in self.state_problem.state_databases
        ]
        self.adjoint_list = [
            database.adjoint for database in self.state_problem.state_databases
        ]

        self.lagrangian_function = LagrangianFunction(
            self.cost_functional_list, state_weak_forms=f_forms
        )

        self.adjoint_problem = AdjointProblem(
            state_databases=self.state_problem.state_databases,
            lagrangian_function=self.lagrangian_function,
        )

        self.gradient_problem = ControlGradientProblem(
            control_database=self.control_database,
            lagrangian_function=self.lagrangian_function,
            control_regularization_list=self.reg_pipe,
            scalar_product_kwargs=scalar_product_kwargs,
        )

    @staticmethod
    def compute_gradient_norm(u: dolfinx.fem.Function) -> float:
        grad_np = u.x.array
        return np.sqrt(grad_np.dot(grad_np))

    def evaluate_cost_functional(
        self,
        comm,
        update_state: bool,
        mat_assemble_method: Literal["identity_row", "lift"] = "lift",
        with_residual=False,
    ):
        if update_state:
            self.state_problem.has_solution = False
            self.state_problem.solve(
                comm,
                jacobi_form=None,
                state_func_space=None,
                mat_assemble_method=mat_assemble_method,
                with_residual=with_residual,
            )

        val = 0.0
        for func in self.cost_functional_list:
            val += func.evaluate()
        reg_val, reg_dict = self.reg_pipe.evaluate()
        return val + reg_val

    def compute_state(
        self,
        comm,
        mat_assemble_method: Literal["identity_row", "lift"] = "lift",
        with_residual=False,
    ):
        self.state_problem.has_solution = False
        has_solution = self.state_problem.solve(
            comm, mat_assemble_method=mat_assemble_method, with_residual=with_residual
        )
        return has_solution

    def compute_adjoint(
        self,
        comm,
        mat_assemble_method: Literal["identity_row", "lift"] = "lift",
        with_residual=False,
    ):
        self.adjoint_problem.has_solution = False
        has_solution = self.adjoint_problem.solve(
            comm, mat_assemble_method=mat_assemble_method, with_residual=with_residual
        )
        return has_solution

    def compute_gradient(
        self,
        comm,
        state_kwargs: dict,
        adjoint_kwargs: dict,
        gradient_kwargs: dict,
        update_state=True,
        update_adjoint=True,
    ) -> dict:
        # ------ step1: solve state problem
        self.state_problem.has_solution = not update_state
        has_solution = self.state_problem.solve(
            comm,
            mat_assemble_method=state_kwargs["mat_assemble_method"],
            with_residual=state_kwargs["with_residual"],
        )
        if not has_solution:
            return {"is_success": False, "log": "State problem solving fail."}

        # ------ step2: update cost function state
        self.update_cost_funcs()

        # ------ step3: solve adjoint problem
        self.adjoint_problem.has_solution = not update_adjoint
        has_solution = self.adjoint_problem.solve(
            comm,
            mat_assemble_method=adjoint_kwargs["mat_assemble_method"],
            with_residual=adjoint_kwargs["with_residual"],
        )
        if not has_solution:
            return {"is_success": False, "log": "Adjoint problem solving fail."}

        # ------ step4: update scalar product
        self.update_update_scalar_product()

        # ------ step5: solve gradient problem
        self.gradient_problem.has_solution = False
        has_solution = self.gradient_problem.solve(
            comm,
            mat_assemble_method=gradient_kwargs["mat_assemble_method"],
            with_residual=gradient_kwargs["with_residual"],
        )
        if not has_solution:
            return {"is_success": False, "log": "Gradient problem solving fail."}

        return {
            "is_success": True,
            "gradients": {
                name: control.grad
                for name, control in self.gradient_problem.control_database.control_pars.items()
            },
        }

    def update_update_scalar_product(self):
        pass

    def update_cost_funcs(self):
        for cost_func in self.cost_functional_list:
            cost_func.update()

    def get_cost_info(
        self,
        update_state: bool,
        comm,
        mat_assemble_method: Literal["identity_row", "lift"] = "lift",
        with_residual=False,
    ) -> dict:
        if update_state:
            self.compute_state(
                comm,
                mat_assemble_method=mat_assemble_method,
                with_residual=with_residual,
            )
        
        cost_dict = {}
        for func in self.cost_functional_list:
            cost_dict[func.name] = func.evaluate()
        _, reg_dict = self.reg_pipe.evaluate()
        cost_dict.update(reg_dict)
        return cost_dict


class ShapeOptimizer(ControlOptimizer):
    def __init__(
        self,
        state_problem: StateProblem,
        shape_database: ShapeProblemDataBase,
        cost_functional_list: list[CostFunctional],
        shape_regularization_list: ShapeRegularizationPipe,
        scalar_product_kwargs: dict[str, Union[str, float, Callable]] = {"method": "default"},
    ):
        self.state_problem = state_problem
        self.shape_database = shape_database
        self.cost_functional_list = cost_functional_list
        self.reg_pipe = shape_regularization_list

        f_forms = [database.F_form for database in self.state_problem.state_databases]
        self.state_list = [
            database.state for database in self.state_problem.state_databases
        ]
        self.adjoint_list = [
            database.adjoint for database in self.state_problem.state_databases
        ]
        
        self.lagrangian_function = LagrangianFunction(self.cost_functional_list, f_forms)

        self.adjoint_problem = AdjointProblem(
            state_databases=self.state_problem.state_databases,
            lagrangian_function=self.lagrangian_function,
        )
        self.gradient_problem = ShapeGradientProblem(
            shape_database=self.shape_database,
            lagrangian_function=self.lagrangian_function,
            shape_regularization_list=self.reg_pipe,
            scalar_product_kwargs=scalar_product_kwargs,
        )

    def compute_gradient(
        self,
        comm,
        state_kwargs: dict,
        adjoint_kwargs: dict,
        gradient_kwargs: dict,
        scalar_product_kwargs: dict,
        update_state=True,
        update_adjoint=True,
    ) -> dict:
        assert gradient_kwargs["mat_assemble_method"] == "identity_row", "Shape gradient problem only support identity_row assemble method." \
            "lift will cause boundary shifting."
        
        # ------ step1: solve state problem
        self.state_problem.has_solution = not update_state
        has_solution = self.state_problem.solve(
            comm,
            mat_assemble_method=state_kwargs["mat_assemble_method"],
            with_residual=state_kwargs["with_residual"],
        )
        if not has_solution:
            return {"is_success": False, "log": "State problem solving fail."}
        
        # ------ step2: update cost function state
        self.update_cost_funcs()

        # ------ step3: solve adjoint problem
        self.adjoint_problem.has_solution = not update_adjoint
        has_solution = self.adjoint_problem.solve(
            comm,
            mat_assemble_method=adjoint_kwargs["mat_assemble_method"],
            with_residual=adjoint_kwargs["with_residual"],
        )
        if not has_solution:
            return {"is_success": False, "log": "Adjoint problem solving fail."}

        # ------ step4: solve gradient problem
        self.update_update_scalar_product(with_residual=scalar_product_kwargs.get('with_residual', False))

        # ------ step5: solve gradient problem
        self.gradient_problem.has_solution = False
        has_solution = self.gradient_problem.solve(
            comm,
            mat_assemble_method=gradient_kwargs["mat_assemble_method"],
            with_residual=gradient_kwargs["with_residual"],
        )
        if not has_solution:
            return {"is_success": False, "log": "Gradient problem solving fail."}

        return {
            "is_success": True,
            "shape_grad": self.gradient_problem.shape_database.shape_grad,
        }

    def update_update_scalar_product(self, with_residual):
        self.gradient_problem.update_scalar_product(with_residual=with_residual)
