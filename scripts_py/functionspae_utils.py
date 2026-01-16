from typing import Literal

import dolfinx
import ufl
from mpi4py import MPI

import numpy as np

from scripts_py.mesh_utils import MeshObj
from scripts_py.dolfinx_utils import DolfinUtils
from scripts_py.eqn_solver import LinearProblemSolver


class FunctionSpaceUtils(object):
    @staticmethod
    def convert_function_space(
        mesh: MeshObj, 
        original_func_space: dolfinx.fem.FunctionSpace,
        original_func_data: np.ndarray,
        target_func_space: dolfinx.fem.FunctionSpace,
        method: Literal['interpolate', 'L2', 'H1'],
        bcs: list[dolfinx.fem.DirichletBC] = [],
        with_dolfinx_function: bool = False,
        ksp_option = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        }
    ):
        """
        Be careful: the goal of function space transform is not to satisfy any PDE on the boundary.
        Specify function space may not be consistent with the boundary condition.
        if bcs are strict boundary conditions (which mean whole domain has boundary conditions), the result may be not correct.
        
        Based on Riesz representation theorem, the function space transform is equivalent to the projection of the original function to the target function space.
        """
        if method == 'interpolate':
            original_func = dolfinx.fem.Function(original_func_space)
            original_func.x.array[:] = original_func_data  # type: ignore
            target_func = dolfinx.fem.Function(target_func_space)
            target_func.interpolate(original_func)  # type: ignore

        elif method == 'L2':
            """inner(target_func_trial, target_func_test) = inner(original_func, target_func_test)"""
            target_func = dolfinx.fem.Function(target_func_space)
            original_func = dolfinx.fem.Function(original_func_space)
            original_func.x.array[:] = original_func_data  # type: ignore
            
            u = ufl.TrialFunction(target_func_space)
            v = ufl.TestFunction(target_func_space)
            a_form = ufl.inner(u, v) * ufl.dx
            L_form = ufl.inner(original_func, v) * ufl.dx
            
            res_dict = LinearProblemSolver.solve_fea_petsc(
                comm = mesh.domain.comm,
                uh = target_func,  # type: ignore
                a_form = a_form,
                l_form = L_form,
                bcs = bcs,
                ksp_option = ksp_option,
                with_residual = False,
            )
            if not res_dict['is_solved']:
                raise ValueError(f"Failed to solve L2 problem.")
                    
        elif method == 'H1':
            target_func = dolfinx.fem.Function(target_func_space)
            original_func = dolfinx.fem.Function(original_func_space)
            original_func.x.array[:] = original_func_data  # type: ignore
            
            u = ufl.TrialFunction(target_func_space)
            v = ufl.TestFunction(target_func_space)
            a_form = ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            L_form = ufl.inner(original_func, v) * ufl.dx
            
            res_dict = LinearProblemSolver.solve_fea_petsc(
                comm = mesh.domain.comm,
                uh = target_func,  # type: ignore
                a_form = a_form,
                l_form = L_form,
                bcs = bcs,
                ksp_option = ksp_option,
                with_residual = False,
            )
            if not res_dict['is_solved']:
                raise ValueError(f"Failed to solve H1 problem.")
        
        else:
            raise ValueError(f"Invalid method: {method}")
        
        if with_dolfinx_function:
            return target_func
        else:
            return target_func.x.array.copy()  # type: ignore
    
    @staticmethod
    def convert_dolfinx_function_space(
        mesh: MeshObj,
        original_func: dolfinx.fem.Function,
        target_func_space: dolfinx.fem.FunctionSpace,
        method: Literal['interpolate', 'L2', 'H1'],
        bcs: list[dolfinx.fem.DirichletBC] = [],
        with_dolfinx_function: bool = False,
        ksp_option = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        }
    ):
        return FunctionSpaceUtils.convert_function_space(
            mesh = mesh,
            original_func_space = original_func.function_space,
            original_func_data = original_func.x.array.copy(),
            target_func_space = target_func_space,
            method = method,
            bcs = bcs,
            with_dolfinx_function = with_dolfinx_function,
            ksp_option = ksp_option,
        )
    
    @staticmethod
    def convert_point_to_cell(
        mesh: MeshObj,
        func_data: np.ndarray|dolfinx.fem.Function,
        is_vector: bool,
        method: Literal['interpolate', 'L2'],
        bcs: list[dolfinx.fem.DirichletBC] = [],
        with_dolfinx_function: bool = False,
        ksp_option = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        }
    ):
        if isinstance(func_data, dolfinx.fem.Function):
            func_data_np = func_data.x.array.copy()
        else:
            func_data_np = func_data
        
        if is_vector:
            original_func_space = DolfinUtils.point_vector_function_space(mesh.domain)
            target_func_space = DolfinUtils.cell_vector_function_space(mesh.domain)
        else:
            original_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
            target_func_space = DolfinUtils.cell_scalar_function_space(mesh.domain)
        
        return FunctionSpaceUtils.convert_function_space(
            mesh = mesh,
            original_func_space = original_func_space,
            original_func_data = func_data_np,
            target_func_space = target_func_space,
            method = method,
            bcs = bcs,
            with_dolfinx_function = with_dolfinx_function,
            ksp_option=ksp_option
        )
    
    @staticmethod
    def convert_cell_to_point(
        mesh: MeshObj,
        func_data: np.ndarray|dolfinx.fem.Function,
        is_vector: bool,
        method: Literal['interpolate', 'L2'],
        bcs: list[dolfinx.fem.DirichletBC] = [],
        with_dolfinx_function: bool = False,
        ksp_option = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        }
    ):
        if isinstance(func_data, dolfinx.fem.Function):
            func_data_np = func_data.x.array.copy()
        else:
            func_data_np = func_data
        
        if is_vector:
            original_func_space = DolfinUtils.cell_vector_function_space(mesh.domain)
            target_func_space = DolfinUtils.point_vector_function_space(mesh.domain)
        else:
            original_func_space = DolfinUtils.cell_scalar_function_space(mesh.domain)
            target_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
        
        return FunctionSpaceUtils.convert_function_space(
            mesh = mesh,
            original_func_space = original_func_space,
            original_func_data = func_data_np,
            target_func_space = target_func_space,
            method = method,
            bcs = bcs,
            with_dolfinx_function = with_dolfinx_function,
            ksp_option=ksp_option
        )
