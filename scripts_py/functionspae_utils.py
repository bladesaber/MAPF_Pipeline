from typing import Literal

import dolfinx
import ufl
from mpi4py import MPI

import numpy as np

from scripts_py.mesh_utils import MeshObj
from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils
from scripts_py.eqn_solver import LinearProblemSolver
from scripts_py.sparse_utils import PetscUtils


class FunctionSpaceUtils(object):
    @staticmethod
    def convert_function_space(
        mesh: MeshObj, 
        original_func_space: dolfinx.fem.FunctionSpace,
        original_func_data: np.ndarray,
        target_func_space: dolfinx.fem.FunctionSpace,
        method: Literal['interpolate', 'L2', 'H1', 'L2+poisson'],
        bcs: list[dolfinx.fem.DirichletBC] = [],
        with_dolfinx_function: bool = False,
        ksp_option = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
        **kwargs
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
        
        elif method == 'L2+poisson':
            target_func = dolfinx.fem.Function(target_func_space)
            original_func = dolfinx.fem.Function(original_func_space)
            original_func.x.array[:] = original_func_data  # type: ignore
            
            u = ufl.TrialFunction(target_func_space)
            v = ufl.TestFunction(target_func_space)
            a_form = ufl.inner(u, v) * ufl.dx + kwargs.get('alpha', 0.03) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
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
        method: Literal['interpolate', 'L2', 'H1', 'L2+poisson'],
        bcs: list[dolfinx.fem.DirichletBC] = [],
        with_dolfinx_function: bool = False,
        ksp_option = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
        **kwargs
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
            **kwargs
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
        method: Literal['interpolate', 'L2', 'volume_weighted'],
        bcs: list[dolfinx.fem.DirichletBC] = [],
        with_dolfinx_function: bool = False,
        target_func_space: dolfinx.fem.FunctionSpace|None = None,
        ksp_option = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
    ):
        if isinstance(func_data, dolfinx.fem.Function):
            func_data_np = func_data.x.array.copy()
            original_func_space = func_data.function_space
        else:
            func_data_np = func_data
            original_func_space = None
        
        if original_func_space is None:
            if is_vector:
                original_func_space = DolfinUtils.cell_vector_function_space(mesh.domain)
            else:
                original_func_space = DolfinUtils.cell_scalar_function_space(mesh.domain)
        
        if target_func_space is None:
            if is_vector:
                target_func_space = DolfinUtils.point_vector_function_space(mesh.domain)
            else:
                target_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
        
        if method == 'volume_weighted':
            point2cells = mesh.get_point_to_cells()
            cell_volumes = mesh.get_cell_volumes()
            if is_vector:
                nodal_values = np.zeros((mesh.num_points, mesh.tdim))
                func_data_np = func_data_np.reshape((-1, mesh.tdim))
            else:
                nodal_values = np.zeros(mesh.num_points)
            
            for vert_i in range(mesh.num_points):
                adj_cells = point2cells[vert_i]
                if len(adj_cells) == 0:
                    raise ValueError(f"Point {vert_i} is isolated.")
                
                if is_vector:
                    cell_value_sum = np.sum(func_data_np[adj_cells, :] * cell_volumes[adj_cells].reshape((-1, 1)), axis=0)
                else:
                    cell_value_sum = np.sum(func_data_np[adj_cells] * cell_volumes[adj_cells])
                total_vol = np.sum(cell_volumes[adj_cells])
                if total_vol <= 1e-14:  # avoid division by zero
                    raise ValueError(f"Total volume is too small.")
                
                if is_vector:
                    nodal_values[vert_i, :] = cell_value_sum / total_vol
                else:
                    nodal_values[vert_i] = cell_value_sum / total_vol
            
            if is_vector:
                point_func_space = DolfinUtils.point_vector_function_space(mesh.domain)
            else:
                point_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
            
            interplot_func = dolfinx.fem.Function(point_func_space)
            interplot_func.x.array[:] = nodal_values.reshape((-1,))  # type: ignore
            
            target_func = dolfinx.fem.Function(target_func_space)
            target_func.interpolate(interplot_func)  # type: ignore
            if with_dolfinx_function:
                return target_func
            else:
                return target_func.x.array.copy()  # type: ignore
            
        else:
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
    def ad_convert_riesz(
        mesh: MeshObj,
        func_space: dolfinx.fem.FunctionSpace,
        func_data: np.ndarray,
        riesz_representation: Literal['L2', 'H1'],
        ksp_option = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
        comm = MPI.COMM_WORLD
    ):
        if riesz_representation == "L2":
            u = ufl.TrialFunction(func_space)
            v = ufl.TestFunction(func_space)

            a_form = ufl.inner(u, v) * ufl.dx
            a_mat = AssembleUtils.assemble_mat(a_form, [], method='lift')
            b_vec = PetscUtils.create_vec(size=func_data.shape[0])
            b_vec.array[:] = func_data
            
            solver = LinearProblemSolver.create_petsc_solver(
                comm = comm, 
                solver_setting = ksp_option, 
                mat = a_mat
            )
            res_dict = LinearProblemSolver.solve_by_petsc(
                b_vec=b_vec, 
                a_mat=a_mat, 
                solver=solver, 
                setOperators=False, 
                with_residual=False
            )
            if not res_dict['is_solved']:
                raise ValueError(f"Failed to solve Riesz problem.")
            riesz_np = res_dict['res'].array

        elif riesz_representation == "H1":
            u = ufl.TrialFunction(func_space)
            v = ufl.TestFunction(func_space)

            a_form = ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            a_mat = AssembleUtils.assemble_mat(a_form, [], method='lift')
            b_vec = PetscUtils.create_vec(size=func_data.shape[0])
            b_vec.array[:] = func_data
            
            solver = LinearProblemSolver.create_petsc_solver(
                comm = comm, 
                solver_setting = ksp_option, 
                mat = a_mat
            )
            res_dict = LinearProblemSolver.solve_by_petsc(
                b_vec=b_vec, 
                a_mat=a_mat, 
                solver=solver, 
                setOperators=False, 
                with_residual=False
            )
            if not res_dict['is_solved']:
                raise ValueError(f"Failed to solve Riesz problem.")
            riesz_np = res_dict['res'].array

        else:
            raise NotImplementedError(f"Unknown Riesz representation: {riesz_representation}")
        
        return riesz_np


