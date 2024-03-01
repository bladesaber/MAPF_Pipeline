import dolfinx
import ufl
from typing import List, Callable, Dict
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
            state_system: StateProblem,
            control_problem: ControlDataBase,
            cost_functional_list: List[CostFunctional_types],
            scalar_product: Callable = None,
            **kwargs
    ):
        self.state_system = state_system
        self.control_problem = control_problem
        self.cost_functional_list = cost_functional_list

        F_forms = [state_problem.F_form for state_problem in self.state_system.state_problems]
        self.lagrangian_function = LagrangianFunction(
            self.cost_functional_list, F_forms
        )

        self.states = [problem.state for problem in self.state_system.state_problems]
        self.adjoints = [problem.adjoint for problem in self.state_system.state_problems]

        self.adjoint_system = AdjointProblem(
            state_problems=self.state_system.state_problems,
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

        for cost_func in self.cost_functional_list:
            cost_func.update()

    def compute_adjoint(self, comm, **kwargs):
        self.adjoint_system.has_solution = False
        has_solution = self.adjoint_system.solve(comm, **kwargs)

    def compute_gradient(self, comm, state_kwargs: Dict = {}, adjoint_kwargs: Dict = {}, gradient_kwargs: Dict = {}):
        self.state_system.has_solution = False
        has_solution = self.state_system.solve(comm, **state_kwargs)

        for cost_func in self.cost_functional_list:
            cost_func.update()

        self.adjoint_system.has_solution = False
        has_solution = self.adjoint_system.solve(comm, **adjoint_kwargs)

        self.gradient_system.has_solution = False
        has_solution = self.gradient_system.solve(comm, **gradient_kwargs)

        return self.control_problem.control_grads

    def update_update_scalar_product(self):
        pass


class OptimalShapeProblem(OptimalControlProblem):
    def __init__(
            self,
            state_system: StateProblem,
            shape_problem: ShapeDataBase,
            cost_functional_list: List[CostFunctional_types],
            shape_regulariztions: ShapeRegularization,
            scalar_product: Callable = None,
            scalar_product_method: Dict = {'method': "default"},
            **kwargs
    ):
        self.state_system = state_system
        self.shape_problem = shape_problem
        self.cost_functional_list = cost_functional_list
        self.shape_regulariztions = shape_regulariztions

        F_forms = [state_problem.F_form for state_problem in self.state_system.state_problems]
        self.lagrangian_function = LagrangianFunction(
            self.cost_functional_list, F_forms
        )

        self.states = [problem.state for problem in self.state_system.state_problems]
        self.adjoints = [problem.adjoint for problem in self.state_system.state_problems]

        self.adjoint_system = AdjointProblem(
            state_problems=self.state_system.state_problems,
            lagrangian_function=self.lagrangian_function
        )

        self.gradient_system = ShapeGradientProblem(
            shape_problem=shape_problem,
            lagrangian_function=self.lagrangian_function,
            scalar_product=scalar_product,
            shape_regulariztions=self.shape_regulariztions,
            scalar_product_method=scalar_product_method,
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

    def compute_gradient(
            self,
            comm,
            state_kwargs: Dict = {},
            adjoint_kwargs: Dict = {},
            gradient_kwargs: Dict = {},
            scalar_product_kwrags: Dict = {}
    ):
        self.update_update_scalar_product(**scalar_product_kwrags)

        self.state_system.has_solution = False
        has_solution = self.state_system.solve(comm, **state_kwargs)

        self.update_cost_funcs()

        self.adjoint_system.has_solution = False
        has_solution = self.adjoint_system.solve(comm, **adjoint_kwargs)

        self.gradient_system.has_solution = False
        has_solution = self.gradient_system.solve(comm, **gradient_kwargs)

        return self.shape_problem.shape_grad

    def update_update_scalar_product(self, **kwargs):
        self.gradient_system.update_scalar_product(**kwargs)

    def update_cost_funcs(self):
        for cost_func in self.cost_functional_list:
            cost_func.update()
