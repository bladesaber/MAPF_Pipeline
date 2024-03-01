"""
Ref: An Introduction to the Space Mapping Technique. Authors: Mohamed Bakr, Kaj Madsen
Ref: Space Mapping for PDE Constrained Shape Optimization. Sebastian Blauth
"""

import numpy as np
import dolfinx
import ufl
from typing import Callable, List, Union, Dict

from .solver_optimize import OptimalControlProblem, OptimalShapeProblem
from ..equation_solver import LinearProblemSolver, NonLinearProblemSolver


class FineModel(object):
    def __init__(
            self,
            state_form: ufl.Form,
            uh_fine: dolfinx.fem.Function,
            control: Union[dolfinx.mesh.Mesh, dolfinx.fem.Function],
            bcs: List[dolfinx.fem.DirichletBC],
            ksp_option: Dict,
            is_linear: bool
    ):
        self.cost_functional_value = np.inf
        self.ksp_option = ksp_option

        self.state_form = state_form
        self.bcs = bcs

        self.uh_fine = uh_fine
        self.control = control  # Control Parameter of FineModel FunctionSpace

        self.is_linear = is_linear
        self.check_valid()
        if self.is_linear:
            self.lhs = ufl.lhs(self.state_form)
            self.rhs = ufl.rhs(self.state_form)
        else:
            self.lhs = self.state_form
            self.rhs = 0.0

    def check_valid(self):
        if self.is_linear:
            valid = len(self.state_form.arguments()) == 2
        else:
            valid = False
            for coef in self.state_form.coefficients():
                if coef == self.uh_fine:
                    valid = True
        return valid

    def solve_and_evaluate(self, comm, **kwargs):
        if self.is_linear:
            res_dict = LinearProblemSolver.solve_by_petsc_form(
                comm=comm,
                uh=self.uh_fine,
                a_form=self.lhs,
                L_form=self.rhs,
                bcs=self.bcs,
                ksp_option=self.ksp_option,
                **kwargs
            )

        else:
            jacobi_form = kwargs.get('jacobi_form', None)
            if jacobi_form is None:
                function_space = kwargs.get('function_space', None)
                if function_space is None:
                    function_space = self.uh_fine.function_space

                jacobi_form = ufl.derivative(self.lhs, self.uh_fine, ufl.TrialFunction(function_space))

            res_dict = NonLinearProblemSolver.solve_by_petsc(
                F_form=self.lhs,
                uh=self.uh_fine,
                jacobi_form=jacobi_form,
                bcs=self.bcs,
                comm=comm,
                ksp_option=self.ksp_option,
                **kwargs
            )

        if kwargs.get('with_debug', False):
            print(f"[DEBUG]: max_error:{res_dict['max_error']:.6f} cost_time:{res_dict['cost_time']:.2f}")

        self.evaluate_loss()

    def evaluate_loss(self):
        raise NotImplementedError


class CoarseModel(object):
    def __init__(
            self,
            opt_problem: Union[OptimalControlProblem, OptimalShapeProblem],
            control: Union[dolfinx.mesh.Mesh, dolfinx.fem.Function],
            extract_parameter_func: Callable,
            **kwargs
    ):
        self.opt_problem = opt_problem
        self.control = control
        self._extract_parameter = extract_parameter_func

        self.parameter_init: np.ndarray = None
        self.parameter_optimal: np.ndarray = None

    def _optimize(self, **kwargs):
        raise NotImplementedError

    def solve(self, **kwargs):
        self.parameter_init = self._extract_parameter(self.control)
        self._optimize(**kwargs)
        self.parameter_optimal = self._extract_parameter(self.control)


class ParameterExtraction(object):
    def __init__(
            self,
            opt_problem: Union[OptimalControlProblem, OptimalShapeProblem],
            control: Union[dolfinx.mesh.Mesh, dolfinx.fem.Function],
            extract_parameter_func: Callable,
    ):
        self.opt_problem = opt_problem
        self.control = control  # Control Parameter of CoarseModel FunctionSpace
        self._extract_parameter = extract_parameter_func
        self.parameter_extract: np.ndarray = None

    def _optimize(self, **kwargs):
        raise NotImplementedError

    def solve(self, **kwargs):
        self._optimize(**kwargs)
        self.parameter_extract = self._extract_parameter(self.control)


class SpaceMappingProblem(object):
    def __init__(
            self,
            coarse_model: CoarseModel,
            fine_model: FineModel,
            parameter_extraction: ParameterExtraction,
            tol: float, max_iter: int,
            is_coarse_fine_collinear: bool = False
    ):
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.parameter_extraction = parameter_extraction
        self.is_coarse_fine_collinear = is_coarse_fine_collinear
        self.invert_scale = 1.0 if self.is_coarse_fine_collinear else -1.0

        self.tol = tol
        self.max_iter = max_iter

    def solve(
            self,
            coarseModel_kwargs: Dict = {},
            fineModel_kwargs: Dict = {},
            paraExtract_kwargs: Dict = {},
            **kwargs
    ):
        self.coarse_model.solve(**coarseModel_kwargs)
        best_para_of_z_space = self.coarse_model.parameter_optimal

        step = 0
        converged = False
        while True:
            step += 1

            # ------ step 1: update uh_fine
            self.fine_model.solve_and_evaluate(**fineModel_kwargs)

            # ------ step 2: compute z_cur based on new uh_fine
            self.parameter_extraction.solve(**paraExtract_kwargs)

            # ------ step 2.5: log something
            self.pre_log(step)

            # ------ step 3: compute direction
            grad = self._compute_direction_step(
                para_s2z=self.parameter_extraction.parameter_extract,
                para_best_z=best_para_of_z_space,
                **kwargs
            )

            # ------ step 4: update model
            updated_info = self._update(grad, self.fine_model, **kwargs)

            # ------ step 5: compute eps
            eps = self._compute_eps(updated_info)

            # ------ step 5.5: update model
            self.post_log(step, eps)

            # ------ step 6: check convergence
            if eps <= self.tol:
                converged = True
                break

            if step >= self.max_iter:
                break

        return converged

    def _compute_eps(self, updated_info: Dict) -> float:
        raise NotImplementedError

    def _compute_direction_step(self, para_s2z: np.ndarray, para_best_z: np.ndarray, **kwargs) -> np.ndarray:
        """
        Direction is Calculated in Coarse FunctionSpace but Used in Fine FunctionSpace, so You Must confirm
        relationship between Coarse FunctionSpace and Fine FunctionSpace is similar and linear

        Param:
            para_s2z: Parameter of coarseModel functionSpace mapped by current fineModel functionSpace
            para_best_z: The best parameter of coarseModel functionSpace
        """
        raise NotImplementedError

    def _update(self, grad: np.ndarray, fine_model: FineModel, **kwargs) -> Dict:
        raise NotImplementedError

    def pre_log(self, step: int):
        pass

    def post_log(self, step: int, eps: float):
        pass
