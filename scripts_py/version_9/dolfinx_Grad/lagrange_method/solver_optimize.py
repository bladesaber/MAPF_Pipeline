import dolfinx
import ufl
from typing import List, Callable
import numpy as np

from .cost_functions import CostFunctional_types, LagrangianFunction
from .type_database import GovDataBase, ControlDataBase, ShapeDataBase
from .problem_state import StateProblem
from .problem_adjoint import AdjointProblem
from .problem_gradient import ControlGradientProblem, ShapeGradientProblem
from .shape_regularization import ShapeRegularization


class OptimalControlProblem(object):
    def __init__(
            self,
            state_problems: List[GovDataBase],
            control_problem: ControlDataBase,
            cost_functional_list: List[CostFunctional_types],
            scalar_product: Callable = None,
            **kwargs
    ):
        self.state_problems = state_problems
        self.control_problem = control_problem
        self.cost_functional_list = cost_functional_list

        F_forms = [state_problem.F_form for state_problem in self.state_problems]
        self.lagrangian_function = LagrangianFunction(
            self.cost_functional_list, F_forms
        )

        self.states = [problem.state for problem in self.state_problems]
        self.adjoints = [problem.adjoint for problem in self.state_problems]

        self.state_system = StateProblem(state_problems=self.state_problems)
        self.adjoint_system = AdjointProblem(
            state_problems=self.state_problems,
            lagrangian_function=self.lagrangian_function
        )

        self.gradient_system = ControlGradientProblem(
            control_problems=self.control_problem,
            lagrangian_function=self.lagrangian_function,
            scalar_product=scalar_product
        )

    @staticmethod
    def compute_gradient_norm(u: dolfinx.fem.Function) -> float:
        return np.sqrt(u.vector.dot(u.vector))

    def evaluate_cost_functional(self, comm, update_state: bool, **kwargs):
        if update_state:
            self.state_system.has_solution = False
            has_solution = self.state_system.solve(comm, **kwargs)

        val = 0.0
        for func in self.cost_functional_list:
            val += func.evaluate()
        return val

    def compute_state(self, comm, **kwargs):
        self.state_system.has_solution = False
        has_solution = self.state_system.solve(comm, **kwargs)

    def compute_adjoint(self, comm, **kwargs):
        self.adjoint_system.has_solution = False
        has_solution = self.adjoint_system.solve(comm, **kwargs)

    def compute_gradient(self, comm, **kwargs):
        self.state_system.has_solution = False
        has_solution = self.state_system.solve(comm, **kwargs)

        self.adjoint_system.has_solution = False
        has_solution = self.adjoint_system.solve(comm, **kwargs)

        self.gradient_system.has_solution = False
        has_solution = self.gradient_system.solve(comm, **kwargs)

        return self.control_problem.control_grads


class OptimalShapeProblem(OptimalControlProblem):
    def __init__(
            self,
            state_problems: List[GovDataBase],
            shape_problem: ShapeDataBase,
            cost_functional_list: List[CostFunctional_types],
            shape_regulariztions: ShapeRegularization,
            scalar_product: Callable = None,
            **kwargs
    ):
        self.state_problems = state_problems
        self.shape_problem = shape_problem
        self.cost_functional_list = cost_functional_list
        self.shape_regulariztions = shape_regulariztions

        F_forms = [state_problem.F_form for state_problem in self.state_problems]
        self.lagrangian_function = LagrangianFunction(
            self.cost_functional_list, F_forms
        )

        self.states = [problem.state for problem in self.state_problems]
        self.adjoints = [problem.adjoint for problem in self.state_problems]

        self.state_system = StateProblem(state_problems=self.state_problems)
        self.adjoint_system = AdjointProblem(
            state_problems=self.state_problems,
            lagrangian_function=self.lagrangian_function
        )

        self.gradient_system = ShapeGradientProblem(
            shape_problem=shape_problem,
            lagrangian_function=self.lagrangian_function,
            scalar_product=scalar_product,
            shape_regulariztions=self.shape_regulariztions
        )

    def evaluate_cost_functional(self, comm, update_state: bool, **kwargs):
        if update_state:
            self.state_system.has_solution = False
            has_solution = self.state_system.solve(comm, **kwargs)

        val = 0.0
        for func in self.cost_functional_list:
            val += func.evaluate()

        val += self.shape_regulariztions.compute_objective()
        return val

    def compute_gradient(self, comm, **kwargs):
        self.state_system.has_solution = False
        has_solution = self.state_system.solve(comm, **kwargs)

        self.adjoint_system.has_solution = False
        has_solution = self.adjoint_system.solve(comm, **kwargs)

        self.gradient_system.has_solution = False
        has_solution = self.gradient_system.solve(comm, **kwargs)

        return self.shape_problem.shape_grad
