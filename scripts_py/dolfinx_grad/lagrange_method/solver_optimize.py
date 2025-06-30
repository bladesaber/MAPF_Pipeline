import dolfinx
import ufl
from typing import List, Callable, Dict, Union, Literal, Tuple
import numpy as np

from scripts_py.version_9.dolfinx_grad.lagrange_method.cost_functions import IntegralFunctional, \
    ScalarTrackingFunctional, MinMaxFunctional, LagrangianFunction
from scripts_py.version_9.dolfinx_grad.lagrange_method.type_database import GovDataBase, ControlDataBase, ShapeDataBase
from scripts_py.version_9.dolfinx_grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.problem_adjoint import AdjointProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.problem_gradient import ControlGradientProblem, \
    ShapeGradientProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.shape_regularization import ShapeRegularizationPipe
from scripts_py.version_9.dolfinx_grad.lagrange_method.control_regularization import ControlRegularizationPipe


class OptimalControlProblem(object):
    def __init__(
            self,
            state_system: StateProblem,
            control_problem: ControlDataBase,
            cost_functional_list: List[Union[IntegralFunctional, ScalarTrackingFunctional, MinMaxFunctional]],
            control_regularization_pipe: ControlRegularizationPipe,
            scalar_product: Callable = None,
    ):
        self.state_system = state_system
        self.control_problem = control_problem
        self.cost_functional_list = cost_functional_list
        self.reg_pipe = control_regularization_pipe

        f_forms = [state_problem.F_form for state_problem in self.state_system.state_problems]
        self.lagrangian_function = LagrangianFunction(self.cost_functional_list, f_forms)

        self.state_list = [problem.state for problem in self.state_system.state_problems]
        self.adjoint_list = [problem.adjoint for problem in self.state_system.state_problems]

        self.adjoint_system = AdjointProblem(
            state_problems=self.state_system.state_problems,
            lagrangian_function=self.lagrangian_function
        )

        self.gradient_system = ControlGradientProblem(
            control_problems=self.control_problem,
            lagrangian_function=self.lagrangian_function,
            scalar_product=scalar_product,
            control_regularization_list=self.reg_pipe
        )

    @staticmethod
    def compute_gradient_norm(u: dolfinx.fem.Function) -> float:
        return np.sqrt(u.vector.dot(u.vector))

    def evaluate_cost_functional(
            self,
            comm, update_state: bool,
            mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ):
        if update_state:
            self.state_system.has_solution = False
            self.state_system.solve(
                comm, jacobi_form=None, state_func_space=None,
                mat_assemble_method=mat_assemble_method,
                with_debug=with_debug
            )

        val = 0.0
        for func in self.cost_functional_list:
            val += func.evaluate()
        val += self.reg_pipe.compute_objective()
        return val

    def compute_state(
            self, comm,
            mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ):
        self.state_system.has_solution = False
        has_solution = self.state_system.solve(comm, mat_assemble_method=mat_assemble_method, with_debug=with_debug)
        return has_solution

    def compute_adjoint(
            self, comm,
            mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ):
        self.adjoint_system.has_solution = False
        has_solution = self.adjoint_system.solve(comm, mat_assemble_method=mat_assemble_method, with_debug=with_debug)
        return has_solution

    def compute_gradient(
            self, comm,
            state_kwargs: Dict,
            adjoint_kwargs: Dict,
            gradient_kwargs: Dict,
            update_state=True,
            update_adjoint=True
    ) -> Tuple[bool, Dict[str, Union[dolfinx.fem.Function, str]]]:
        self.state_system.has_solution = not update_state
        has_solution = self.state_system.solve(
            comm,
            mat_assemble_method=state_kwargs['mat_assemble_method'],
            with_debug=state_kwargs['with_debug']
        )
        if not has_solution:
            return False, {'log': 'State problem solving fail.'}
        self.update_cost_func_state()

        self.adjoint_system.has_solution = not update_adjoint
        has_solution = self.adjoint_system.solve(
            comm,
            mat_assemble_method=adjoint_kwargs['mat_assemble_method'],
            with_debug=adjoint_kwargs['with_debug']
        )
        if not has_solution:
            return False, {'log': 'Adjoint problem solving fail.'}

        self.gradient_system.has_solution = False
        has_solution = self.gradient_system.solve(
            comm,
            mat_assemble_method=gradient_kwargs['mat_assemble_method'],
            with_debug=gradient_kwargs['with_debug']
        )
        if not has_solution:
            return False, {'log': 'Gradient problem solving fail.'}

        return True, self.control_problem.control_grads

    def update_update_scalar_product(self):
        pass

    def update_cost_func_state(self):
        for cost_func in self.cost_functional_list:
            cost_func.update()

    def get_cost_info(
            self, comm,
            mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ) -> Dict:
        cost_dict = {}
        for func in self.cost_functional_list:
            cost_dict[func.name] = func.evaluate()
        cost_dict.update(self.reg_pipe.get_cost_info())
        return cost_dict


class OptimalShapeProblem(OptimalControlProblem):
    def __init__(
            self,
            state_system: StateProblem,
            shape_problem: ShapeDataBase,
            cost_functional_list: List[Union[IntegralFunctional, ScalarTrackingFunctional, MinMaxFunctional]],
            shape_regularization_list: ShapeRegularizationPipe,
            scalar_product: Callable = None,
            scalar_product_method: Dict = {'method': "default"},
    ):
        self.state_system = state_system
        self.shape_problem = shape_problem
        self.cost_functional_list = cost_functional_list
        self.reg_pipe = shape_regularization_list

        f_forms = [state_problem.F_form for state_problem in self.state_system.state_problems]
        self.lagrangian_function = LagrangianFunction(self.cost_functional_list, f_forms)

        self.state_list = [problem.state for problem in self.state_system.state_problems]
        self.adjoint_list = [problem.adjoint for problem in self.state_system.state_problems]

        self.adjoint_system = AdjointProblem(
            state_problems=self.state_system.state_problems,
            lagrangian_function=self.lagrangian_function
        )

        self.gradient_system = ShapeGradientProblem(
            shape_problem=shape_problem,
            lagrangian_function=self.lagrangian_function,
            scalar_product=scalar_product,
            shape_regularization_list=self.reg_pipe,
            scalar_product_method=scalar_product_method,
        )

    def compute_gradient(
            self,
            comm,
            state_kwargs: Dict,
            adjoint_kwargs: Dict,
            gradient_kwargs: Dict,
            scalar_product_kwargs: Dict,
            update_state=True, update_adjoint=True
    ) -> Tuple[bool, Union[dolfinx.fem.Function, Dict[str, str]]]:
        self.update_update_scalar_product(with_debug=scalar_product_kwargs['with_debug'])

        self.state_system.has_solution = not update_state
        has_solution = self.state_system.solve(
            comm,
            mat_assemble_method=state_kwargs['mat_assemble_method'],
            with_debug=state_kwargs['with_debug']
        )
        if not has_solution:
            return False, {'log': 'State problem solving fail.'}
        self.update_cost_func_state()

        self.adjoint_system.has_solution = not update_adjoint
        has_solution = self.adjoint_system.solve(
            comm,
            mat_assemble_method=adjoint_kwargs['mat_assemble_method'],
            with_debug=adjoint_kwargs['with_debug']
        )
        if not has_solution:
            return False, {'log': 'Adjoint problem solving fail.'}

        self.gradient_system.has_solution = False
        has_solution = self.gradient_system.solve(
            comm,
            mat_assemble_method=gradient_kwargs['mat_assemble_method'],
            with_debug=gradient_kwargs['with_debug']
        )
        if not has_solution:
            return False, {'log': 'Gradient problem solving fail.'}

        return True, self.shape_problem.shape_grad

    def update_update_scalar_product(self, with_debug):
        self.gradient_system.update_scalar_product(with_debug=with_debug)
