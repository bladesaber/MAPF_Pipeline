from typing import Union, Callable, Any, Literal
import os
import shutil

import dolfinx
import ufl
from ufl.algorithms.analysis import extract_arguments

import numpy as np
import pyvista
import subprocess
import json
from functools import partial
import copy

from thirdparty.pyadjoint.pyadjoint.block import Block
from thirdparty.pyadjoint.pyadjoint.overloaded_type import get_overloaded_class, create_overloaded_object
from thirdparty.pyadjoint.pyadjoint.tape import (
    Tape,
    annotate_tape, 
    get_working_tape, 
    no_annotations,
    stop_annotating
)

from scripts_py.dolfinx_utils import DolfinUtils, BoundaryUtils
from scripts_py.auto_dolfinx.blocks.block_solve import EquationSolveBlock
from scripts_py.auto_dolfinx.opt_solver.topo_fluid import AutoFluidBoundary
from scripts_py.auto_dolfinx.cbs_topo.seq_mesh_utils import SeqMeshObj
from scripts_py.mesh_utils import MeshObj, MeshUtils
from scripts_py.simulator.fipy_simulator import FipyStokeSimulator, FluidBoundary
from scripts_py.functionspae_utils import FunctionSpaceUtils

from basix.ufl import element, mixed_element
from scripts_py.eqn_solver import LinearProblemSolver, NonLinearProblemSolver


class CacheMemory(object):
    def __init__(self) -> None:
        self.cache_memory = {}
    
    def update_cache(self, key: str, value: Any) -> None:
        self.cache_memory[key] = value
    
    def get_cache(self, key: str) -> Any:
        return self.cache_memory[key] if key in self.cache_memory else None
    
    def clear_cache(self) -> None:
        self.cache_memory.clear()


def refit_to_dolfinx(
    mesh: MeshObj,
    F_form: ufl.Form,
    state: dolfinx.fem.Function,
    adjoint: dolfinx.fem.Function,
    bcs: list[dolfinx.fem.DirichletBC],
    forward_ksp_option: dict,
    forward_kwargs: dict,
    is_linear: bool,
):
    if is_linear:
        F_bilinear_form = ufl.replace(
            F_form, {
                state: ufl.TrialFunction(state.function_space),  # type: ignore
                adjoint: ufl.TestFunction(adjoint.function_space),  # type: ignore
            }
        )
        lhs: ufl.Form = ufl.lhs(F_bilinear_form)  # type: ignore
        rhs: ufl.Form = ufl.rhs(F_bilinear_form)
                    
        res_dict = LinearProblemSolver.solve_fea_petsc(
            comm=mesh.domain.comm, 
            uh=state, 
            a_form=lhs,
            l_form=rhs, 
            bcs=bcs, 
            ksp_option=forward_ksp_option['ksp_setting'],
            mat_assemble_method=forward_kwargs['mat_assemble_method'],
            with_residual=forward_kwargs['with_residual']
        )
        if not res_dict['is_solved']:
            raise ValueError(f"Failed to solve recompute problem.")
            
        output = res_dict['res']
            
    else:
        F_recompute_linear_form: ufl.Form = ufl.replace(
            F_form, {adjoint: ufl.TestFunction(adjoint.function_space)}  # type: ignore
        )
            
        jacobi_form = ufl.derivative(
            F_recompute_linear_form,
            state,
            ufl.TrialFunction(state.function_space),
        )
        jacobi_form_dolfin = DolfinUtils.ufl_to_dolfin(jacobi_form)
            
        res_dict = NonLinearProblemSolver.solve_fea_petsc(
            comm=mesh.domain.comm,
            f_form=F_recompute_linear_form,
            uh=state,
            jacobi_form=jacobi_form_dolfin,  # type: ignore
            bcs=bcs,
            mat_assemble_method=forward_kwargs['mat_assemble_method'],
            snes_setting=forward_ksp_option['snes_setting'],
            ksp_setting=forward_ksp_option['ksp_setting'],
            criterion={},
            with_residual=forward_kwargs['with_residual'],
        )
        if not res_dict['is_solved']:
            raise ValueError(f"Failed to solve recompute problem.")

        output = res_dict['res']
    
    return output


def convert_point_to_cell(
    point_field: np.ndarray, structure_mesh: MeshObj, is_vector: bool, 
    method: Literal['interpolate', 'L2'] = 'interpolate', bcs: list[dolfinx.fem.DirichletBC] = []
):
    cell_func = FunctionSpaceUtils.convert_point_to_cell(
        mesh = structure_mesh,
        func_data = point_field,
        is_vector = is_vector,
        method = method,
        bcs = bcs,
        with_dolfinx_function = True,
    )
    if is_vector:
        return cell_func.x.array.copy().reshape((-1, point_field.shape[1]))  # type: ignore
    else:
        return cell_func.x.array.copy()  # type: ignore


def convert_cell_to_point(
    cell_field: np.ndarray, structure_mesh: MeshObj, is_vector: bool,
    method: Literal['interpolate', 'L2'] = 'interpolate', bcs: list[dolfinx.fem.DirichletBC] = []
):
    point_func = FunctionSpaceUtils.convert_cell_to_point(
        mesh = structure_mesh,
        func_data = cell_field,
        is_vector = is_vector,
        method = method,
        bcs = bcs,
        with_dolfinx_function = True,
    )
    if is_vector:
        return point_func.x.array.copy().reshape((-1, cell_field.shape[1]))  # type: ignore
    else:
        return point_func.x.array.copy()  # type: ignore


def define_interpolation_bcs(
    mesh: MeshObj,
    boundaries: list[AutoFluidBoundary],
    V_func_space: dolfinx.fem.FunctionSpace,
    Q_func_space: dolfinx.fem.FunctionSpace,
) -> tuple[list[dolfinx.fem.DirichletBC], list[dolfinx.fem.DirichletBC]]:
    """function space transform is not consistent with strict boundary condition.
        so i just fix the wall and pressure boundary.
    """
    velocity_L2_bcs = []
    pressure_L2_bcs = []
    for bry_info in boundaries:
        if bry_info.boundary_type == 'velocity':
            # inflow_velocity_exp = partial(
            #     FluidBoundary.inflow_velocity_exp,
            #     velocity=np.array(bry_info.value),
            #     tdim=mesh.tdim,
            #     face_center=bry_info.face_center, # type: ignore
            #     face_norm_vector=bry_info.face_norm_vector, # type: ignore
            #     face_radius=bry_info.face_radius, # type: ignore
            #     method=bry_info.interpolate_method,
            # )
            # velocity_L2_bcs.append(
            #     BoundaryUtils.define_dirichlet_bc(
            #         bc_func_space=V_func_space, 
            #         value_func_space=V_func_space, 
            #         dofs=MeshUtils.extract_entity_dofs(
            #             V_func_space, mesh.fdim,
            #             MeshUtils.extract_entity_indices(
            #                 mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=bry_info.marker
            #             )
            #         ), 
            #         bc_value=inflow_velocity_exp, 
            #         is_mixed_element=False)
            # )
            pass
             
        elif bry_info.boundary_type == 'pressure':
            pressure_L2_bcs.append(
                BoundaryUtils.define_dirichlet_bc(
                    bc_func_space=Q_func_space, 
                    value_func_space=Q_func_space, 
                    dofs=MeshUtils.extract_entity_dofs(
                        Q_func_space, mesh.fdim,
                        MeshUtils.extract_entity_indices(
                            mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=bry_info.marker
                        )
                    ), 
                    bc_value=bry_info.value, 
                    is_mixed_element=False)
            )
        elif bry_info.boundary_type == 'wall':
            velocity_L2_bcs.append(
                BoundaryUtils.define_dirichlet_bc(
                    bc_func_space=V_func_space, 
                    value_func_space=V_func_space, 
                    dofs=MeshUtils.extract_entity_dofs(
                        V_func_space, mesh.fdim,
                        MeshUtils.extract_entity_indices(
                            mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=bry_info.marker
                        )
                    ), 
                    bc_value=dolfinx.fem.Function(V_func_space), 
                    is_mixed_element=False)
            )
    
    return velocity_L2_bcs, pressure_L2_bcs


def convert_fipy_result_to_dolfinx(
    mesh: MeshObj,
    cell_velocity: np.ndarray, 
    cell_pressure: np.ndarray,
    boundaries: list[AutoFluidBoundary],
    V_func_space: dolfinx.fem.FunctionSpace,
    Q_func_space: dolfinx.fem.FunctionSpace,
    uh: dolfinx.fem.Function, 
    V_to_W_dofs: np.ndarray,
    Q_to_W_dofs: np.ndarray,
    # interpolation_ksp_option = {
    #     'ksp_type': 'richardson',
    #     'ksp_max_it': 1, # only one iteration = equivalent to preonly
    #     'pc_type': 'lu',
    #     'pc_factor_mat_solver_type': 'mumps',
    #     'mat_mumps_icntl_24': 1,
    #     'mat_mumps_icntl_25': 0,
    # },
    interpolation_ksp_option = {  # NOTE: is it better than richardson? it seems better and faster than richardson. Why?
        "ksp_type": 'preonly',
        "pc_type": 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }
):
    velocity_L2_bcs, pressure_L2_bcs = define_interpolation_bcs(
        mesh=mesh,
        boundaries=boundaries,
        V_func_space=V_func_space,
        Q_func_space=Q_func_space,
    )
    
    cell_vector_func_space = DolfinUtils.cell_vector_function_space(mesh.domain)
    cell_velocity_fun = dolfinx.fem.Function(cell_vector_func_space)
    cell_velocity_fun.x.array[:] = cell_velocity.reshape((-1,))  # type: ignore
    # NOTE: 无论什么映射方法,MeshSize越大,精度越差
    velocity_interpolate_fun = FunctionSpaceUtils.convert_dolfinx_function_space(
        mesh=mesh,
        original_func=cell_velocity_fun,  # type: ignore
        target_func_space=V_func_space,
        method='L2',
        bcs=velocity_L2_bcs,
        with_dolfinx_function=True,
        ksp_option=interpolation_ksp_option,
    )
    # NOTE: interpolate is not good enough
    # velocity_interpolate_fun = FunctionSpaceUtils.convert_dolfinx_function_space(
    #     mesh=mesh,
    #     original_func=cell_velocity_fun,  # type: ignore
    #     target_func_space=V_func_space,
    #     method='interpolate',
    #     with_dolfinx_function=True,
    # )
    # NOTE: after mapping, the result is not consistent with the boundary condition.
    # And velocity and pressure mapping are not consistent by having shift. 关于FVM映射FEM的问题无法解决
    # higher_degree_velocity_interpolate_fun = FunctionSpaceUtils.convert_dolfinx_function_space(
    #     mesh=mesh,
    #     original_func=cell_velocity_fun,  # type: ignore
    #     target_func_space=DolfinUtils.point_vector_function_space(mesh.domain, degree=2),  # terrible
    #     method='L2',
    #     bcs=velocity_L2_bcs,
    #     with_dolfinx_function=True,
    #     ksp_option=interpolation_ksp_option,
    # )
    # velocity_interpolate_fun = dolfinx.fem.Function(V_func_space)
    # velocity_interpolate_fun.interpolate(higher_degree_velocity_interpolate_fun)  # type: ignore
    # # ----------------------------
    
    cell_scalar_func_space = DolfinUtils.cell_scalar_function_space(mesh.domain)
    cell_pressure_fun = dolfinx.fem.Function(cell_scalar_func_space)
    cell_pressure_fun.x.array[:] = cell_pressure.reshape((-1,))  # type: ignore
    pressure_interpolate_fun = FunctionSpaceUtils.convert_dolfinx_function_space(
        mesh=mesh,
        original_func=cell_pressure_fun,  # type: ignore
        target_func_space=Q_func_space,
        method='L2',
        bcs=pressure_L2_bcs,
        with_dolfinx_function=True,
        ksp_option=interpolation_ksp_option,
    )
    # pressure_interpolate_fun = FunctionSpaceUtils.convert_dolfinx_function_space(
    #     mesh=mesh,
    #     original_func=cell_pressure_fun,  # type: ignore
    #     target_func_space=Q_func_space,
    #     method='interpolate',
    #     with_dolfinx_function=True,
    # )
    # higher_degree_pressure_interpolate_fun = FunctionSpaceUtils.convert_dolfinx_function_space(
    #     mesh=mesh,
    #     original_func=cell_pressure_fun,  # type: ignore
    #     target_func_space=DolfinUtils.point_scalar_function_space(mesh.domain, degree=2),
    #     method='L2',
    #     bcs=pressure_L2_bcs,
    #     with_dolfinx_function=True,
    #     ksp_option=interpolation_ksp_option,
    # )
    # pressure_interpolate_fun = dolfinx.fem.Function(Q_func_space)
    # pressure_interpolate_fun.interpolate(higher_degree_pressure_interpolate_fun)  # type: ignore
    
    uh.x.array[V_to_W_dofs] = velocity_interpolate_fun.x.array.copy()  # type: ignore
    uh.x.array[Q_to_W_dofs] = pressure_interpolate_fun.x.array.copy()  # type: ignore
    
    return uh


def convert_fipy_result_to_mix_dolfinx(
    mesh: MeshObj,
    cell_velocity: np.ndarray, 
    cell_pressure: np.ndarray,
    uh: dolfinx.fem.Function, 
    interpolation_ksp_option = {
        "ksp_type": 'preonly',
        "pc_type": 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }
):
    """与convert_fipy_result_to_dolfinx效果一样"""
    velocity_element = element("DG", mesh.cell_types(), degree=0, shape=(mesh.tdim,))
    pressure_element = element("DG", mesh.cell_types(), degree=0)
    mix_element = mixed_element([velocity_element, pressure_element])
    W = dolfinx.fem.functionspace(mesh.domain, mix_element)
    W0, W1 = W.sub(0), W.sub(1)
    V, V_to_W_dofs = W0.collapse()
    Q, Q_to_W_dofs = W1.collapse()
    
    cell_uh = dolfinx.fem.Function(W, name=f"map_state")
    cell_uh.x.array[V_to_W_dofs] = cell_velocity.reshape((-1,))  # type: ignore
    cell_uh.x.array[Q_to_W_dofs] = cell_pressure.reshape((-1,))  # type: ignore
    
    interpolate_fun = FunctionSpaceUtils.convert_dolfinx_function_space(
        mesh=mesh,
        original_func=cell_uh,  # type: ignore
        target_func_space=uh.function_space,
        method='L2',
        bcs=[],
        with_dolfinx_function=True,
        ksp_option=interpolation_ksp_option,
    )
    uh.x.array[:] = interpolate_fun.x.array.copy()  # type: ignore
    return uh


def run_stoke_fluid_simulation(
    boundaries: list[AutoFluidBoundary],
    msh_file: str,
    velocityRelaxation: float,
    pressureRelaxation: float,
    log_step: int,
    hot_steps: int,
    simulation_iterations: int,
    convergence_threshold: float,
    save_dir: str,
    use_tqdm: bool,
    use_log: bool,
    parallel_num: int,
    
    structure_mesh: MeshObj,
    structure_viscosity: float|np.ndarray|list,
    structure_source_term: np.ndarray|None,
    structure_obs_weight: np.ndarray|None,
    
    force_msh_init: bool = False,
    init_vtu: pyvista.UnstructuredGrid|None = None,
):
    assert save_dir is not None, "save_dir is not initialized"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    source_msh_file = os.path.join(save_dir, 'source_msh.msh')
    shutil.copy(msh_file, source_msh_file)
    
    cfg_file = os.path.join(save_dir, 'fipy_simulation_cfg.json')
    result_dir = os.path.join(save_dir, 'cache')
    
    if init_vtu is None:
        init_vtu = structure_mesh.grid.copy()
        init_vtu.cell_data.clear()
        init_vtu.point_data.clear()
    
    if structure_source_term is not None:
        if structure_source_term.shape[0] == structure_mesh.num_points:
            structure_source_term = convert_point_to_cell(
                structure_source_term, structure_mesh, False, method='interpolate', bcs=[]
            )
        init_vtu.cell_data['source_term'] = structure_source_term
        
    if structure_obs_weight is not None:
        if structure_obs_weight.shape[0] == structure_mesh.num_points:
            structure_obs_weight = convert_point_to_cell(
                structure_obs_weight, structure_mesh, False, method='interpolate', bcs=[]
            )
        init_vtu.cell_data['obstacle'] = structure_obs_weight
        
    if isinstance(structure_viscosity, (np.ndarray, list)):
        if isinstance(structure_viscosity, list):
            structure_viscosity = np.array(structure_viscosity)
        if structure_viscosity.shape[0] == structure_mesh.num_points:
            structure_viscosity = convert_point_to_cell(
                structure_viscosity, structure_mesh, False, method='interpolate', bcs=[]
            )
        init_vtu.cell_data['viscosity'] = structure_viscosity
        viscosity = None
    elif isinstance(structure_viscosity, float):
        if 'viscosity' in init_vtu.cell_data:
            init_vtu.cell_data.pop('viscosity')  # delete the viscosity field
        viscosity = structure_viscosity
    else:
        raise ValueError(f"Invalid structure_viscosity type: {type(structure_viscosity)}")
    
    init_vtu_file = os.path.join(save_dir, 'init.vtu')
    init_vtu.save(init_vtu_file)
    
    FipyStokeSimulator.create_simulation_config(
        cfg_file=cfg_file,
        source_msh_file=source_msh_file,
        dim=structure_mesh.tdim,
        velocityRelaxation=velocityRelaxation,
        pressureRelaxation=pressureRelaxation,
        boundary_list=boundaries,  # type: ignore
        iterations=simulation_iterations,
        convergence_threshold=convergence_threshold,
        hot_steps=hot_steps,
        log_step=log_step,
        save_dir=result_dir,
        use_tqdm=use_tqdm,
        use_log=use_log,
        viscosity=viscosity,
        structure_init_vtu_file=init_vtu_file,
    )
    
    if parallel_num > 1:
        cmd_str = f'mpirun -n {parallel_num} python scripts_py/simulator/fipy_simulator.py --cfg_file {cfg_file} --force_msh_init {force_msh_init}'
    else:
        cmd_str = f'python scripts_py/simulator/fipy_simulator.py --cfg_file {cfg_file} --force_msh_init {force_msh_init}'
    
    subprocess.run(cmd_str, shell=True, cwd='/home/admin123456/Desktop/work/MAPF_Pipeline')
    
    result_file = os.path.join(result_dir, 'result.vtu')
    residual_json = os.path.join(result_dir, 'residual.json')
    assert os.path.exists(result_file), "[Error]: simulattion failed"
    assert os.path.exists(residual_json), "[Error]: simulattion failed"
    
    with open(residual_json, 'r') as f:
        res_report = json.load(f)
    
    if use_tqdm and use_log:
        log = f'[Log]: {parallel_num} Simulator: x_residual: {res_report['x_residual']:.8f} y_residual: {res_report['y_residual']:.8f} '
        if 'z_residual' in res_report:
            log += f'z_residual: {res_report['z_residual']:.8f} '
        log += f'p_residual: {res_report['p_residual']:.8f} '
        log += f'continuity_residual: {res_report['continuity_residual']:.8f}'
        print(log)
            
    res_vtu = pyvista.read(result_file)
    cell_velocity = res_vtu.cell_data['velocity'][:, :structure_mesh.tdim]
    cell_pressure = res_vtu.cell_data['pressure']
    
    # ------ store fipy result ------
    cache_memory = CacheMemory()
    cache_memory.update_cache('init_vtu', res_vtu)
    
    return cell_velocity, cell_pressure, cache_memory


def equation_cfd_topoWeight_solve(
    # ------ simulator related ------
    fluid_type: Literal['stoke', 'navier_stokes', 'rans'],
    boundaries: list[AutoFluidBoundary],
    msh_file: str,
    viscosity: float|np.ndarray|list,
    velocityRelaxation: float,
    pressureRelaxation: float,
    structure_mesh: SeqMeshObj,
    structure_obs_weight: np.ndarray|None,
    log_step: int,
    hot_steps: int,
    simulation_iterations: int,
    convergence_threshold: float,
    save_dir: str,
    parallel_num: int,
    use_tqdm: bool,
    use_log: bool,
    with_cache: bool,
    
    # ------ result convert related ------
    V_to_W_dofs: np.ndarray,
    Q_to_W_dofs: np.ndarray,
    V_func_space: dolfinx.fem.FunctionSpace,
    Q_func_space: dolfinx.fem.FunctionSpace,
    
    # ------ alpha related ------
    topo_weight_name: str,
    alpha_fun: Callable,
    alpha_min: float,
    alpha_max: float,
    alpha_p: float,
    
    # ------ autdolfinx related ------
    F_form: ufl.Form,
    state: dolfinx.fem.Function,
    adjoint: dolfinx.fem.Function,
    tlm_ksp_option={
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    },
    adj_ksp_option={
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    },
    tlm_kwargs: dict = {
        'mat_assemble_method': 'lift',
        'with_residual': False
    },
    adj_kwargs: dict = {
        'mat_assemble_method': 'lift',
        'with_residual': False
    },
    tape: Tape = None,  # type: ignore
    with_block: bool = False,
    
    forward_ksp_option: dict = {
        'ksp_setting': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
        'snes_setting': {}
    },
    forward_kwargs: dict = {
        'mat_assemble_method': 'lift',
        'with_residual': False
    },
    **kwargs
):
    if tape is None:
        with_annotate = annotate_tape()
    else:
        with_annotate = True
    
    with stop_annotating():
        find_topo_weight = False
        for coeff in F_form.coefficients():  # type: ignore
            if coeff.name == topo_weight_name:
                topo_weight = coeff
                find_topo_weight = True
                break
        if not find_topo_weight:
            raise ValueError(f"Topo weight {topo_weight_name} not found in F_form.")
        
        structure_source_term = alpha_fun(
            rho=topo_weight.x.array.copy(),
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            p=alpha_p,
            pow_operator=np.power
        )
        
        if fluid_type == 'stoke':
            cell_velocity, cell_pressure, cache_memory = run_stoke_fluid_simulation(
                boundaries=boundaries,
                msh_file=msh_file,
                velocityRelaxation=velocityRelaxation,
                pressureRelaxation=pressureRelaxation,
                log_step=log_step,
                hot_steps=hot_steps,
                simulation_iterations=simulation_iterations,
                convergence_threshold=convergence_threshold,
                save_dir=save_dir,
                use_tqdm=use_tqdm,
                use_log=use_log,
                parallel_num=parallel_num,
                structure_mesh=structure_mesh,
                structure_viscosity=viscosity,
                structure_source_term=structure_source_term,
                structure_obs_weight=structure_obs_weight,
                force_msh_init=True,
                init_vtu=None,
            )
        else:
            raise ValueError(f"Invalid fluid type: {fluid_type}")
                    
        state = convert_fipy_result_to_dolfinx(
            mesh=structure_mesh,
            cell_velocity=cell_velocity,
            cell_pressure=cell_pressure,
            boundaries=boundaries,
            V_func_space=V_func_space,
            Q_func_space=Q_func_space,
            uh=state,
            V_to_W_dofs=V_to_W_dofs,
            Q_to_W_dofs=Q_to_W_dofs,
        )
        
        # very terrible
        # # NOTE: try to make result to consistent with boundaries. 由于FVM结果与FEM不一致,导致了FixFlux Cost非常不稳定
        # if fluid_type == 'stoke':
        #     state = refit_to_dolfinx(
        #         mesh=structure_mesh,
        #         F_form=F_form,
        #         state=state,
        #         adjoint=adjoint,
        #         bcs=[bry_info.bry_cell for bry_info in boundaries], # type: ignore
        #         forward_ksp_option=forward_ksp_option,
        #         forward_kwargs=forward_kwargs,
        #         is_linear=True,
        #     )
        # else:
        #     raise ValueError(f"Invalid fluid type: {fluid_type}")
        
        output = state
        
        if not with_cache:
            cache_memory.clear_cache()
            cache_memory = None
    
    if with_annotate:
        block = CfdTopoWeightEqnSolveBlock(
            # ------ simulator related ------
            boundaries = boundaries,
            msh_file = msh_file,
            fluid_type = fluid_type,
            viscosity = viscosity,
            velocityRelaxation = velocityRelaxation,
            pressureRelaxation = pressureRelaxation,
            structure_mesh = structure_mesh,
            structure_obs_weight = structure_obs_weight,
            hot_steps = hot_steps,
            log_step = log_step,
            simulation_iterations = simulation_iterations,
            convergence_threshold = convergence_threshold,
            save_dir = save_dir,
            parallel_num = parallel_num,
            use_tqdm = use_tqdm,
            use_log = use_log,
            cache_memory = cache_memory,
            
            # ------ result convert related ------
            V_to_W_dofs = V_to_W_dofs,
            Q_to_W_dofs = Q_to_W_dofs,
            V_func_space = V_func_space,
            Q_func_space = Q_func_space,
            
            # ------ alpha related ------
            topo_weight_name = topo_weight_name,
            alpha_fun = alpha_fun,
            alpha_min = alpha_min,
            alpha_max = alpha_max,
            alpha_p = alpha_p,
            
            # ------ autdolfinx related ------
            F_form=F_form,
            state=state,
            adjoint=adjoint,
            tlm_ksp_option=tlm_ksp_option,
            adj_ksp_option=adj_ksp_option,
            tlm_kwargs=tlm_kwargs,
            adj_kwargs=adj_kwargs,
            
            forward_ksp_option=forward_ksp_option,
            forward_kwargs=forward_kwargs,
        )
        
        if tape is None:
            tape = get_working_tape()
        tape.add_block(block)
        block.add_output(output.create_block_variable())  # type: ignore
    
    if with_block and with_annotate:
        return output, block
    return output


class CfdTopoWeightEqnSolveBlock(EquationSolveBlock):
    def __init__(
        self,
        # ------ simulator related ------
        boundaries: list[AutoFluidBoundary],
        msh_file: str,
        fluid_type: Literal['stoke', 'navier_stokes'],
        viscosity: float|np.ndarray|list,
        velocityRelaxation: float,
        pressureRelaxation: float,
        structure_mesh: SeqMeshObj,
        structure_obs_weight: np.ndarray|None,
        hot_steps: int,
        log_step: int,
        simulation_iterations: int,
        convergence_threshold: float,
        save_dir: str,
        parallel_num: int,
        use_tqdm: bool,
        use_log: bool,
        cache_memory: CacheMemory|None,
        
        # ------ alpha related ------
        topo_weight_name: str,
        alpha_fun: Callable,
        alpha_min: float,
        alpha_max: float,
        alpha_p: float,
        
        # ------ result convert related ------
        V_to_W_dofs: np.ndarray,
        Q_to_W_dofs: np.ndarray,
        V_func_space: dolfinx.fem.FunctionSpace,
        Q_func_space: dolfinx.fem.FunctionSpace,
        
        # ------ autdolfinx related ------
        F_form: ufl.Form,
        state: dolfinx.fem.Function,
        adjoint: dolfinx.fem.Function,
        tlm_ksp_option: dict, 
        adj_ksp_option: dict, 
        tlm_kwargs: dict,
        adj_kwargs: dict,
        
        forward_ksp_option: dict,
        forward_kwargs: dict,
    ):
        super().__init__(
            domain=structure_mesh.domain,
            F_form=F_form,
            state=state,
            adjoint=adjoint,
            bcs=[bry_info.bry_cell for bry_info in boundaries], # type: ignore
            is_linear=True,
            tlm_ksp_option=tlm_ksp_option,
            adj_ksp_option=adj_ksp_option, 
            tlm_kwargs=tlm_kwargs,
            adj_kwargs=adj_kwargs,
            
            forward_ksp_option=forward_ksp_option,
            forward_kwargs=forward_kwargs,
        )
        
        # ------ simulator related ------
        self.boundaries = boundaries
        self.msh_file = msh_file
        self.fluid_type = fluid_type
        self.viscosity = viscosity
        self.velocityRelaxation = velocityRelaxation
        self.pressureRelaxation = pressureRelaxation
        self.structure_mesh = structure_mesh
        self.structure_obs_weight = structure_obs_weight
        self.log_step = log_step
        self.hot_steps = hot_steps
        self.simulation_iterations = simulation_iterations
        self.convergence_threshold = convergence_threshold
        self.save_dir = save_dir
        self.parallel_num = parallel_num
        self.use_tqdm = use_tqdm
        self.use_log = use_log
        self.cache_memory = cache_memory
        
        # ------ alpha related ------
        self.topo_weight_name = topo_weight_name
        self.alpha_fun = alpha_fun
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_p = alpha_p
        
        # ------ result convert related ------
        self.V_to_W_dofs = V_to_W_dofs
        self.Q_to_W_dofs = Q_to_W_dofs
        self.V_func_space = V_func_space
        self.Q_func_space = Q_func_space
        
    def recompute_component(self, inputs, block_variable, idx, prepared: dict) -> dolfinx.fem.Function:
        F_recompute_form: ufl.Form = prepared['F_recompute_form']
        uh: dolfinx.fem.Function = prepared['uh']

        find_topo_weight = False
        for coeff in F_recompute_form.coefficients():  # type: ignore
            if coeff.name == self.topo_weight_name:
                topo_weight = coeff
                find_topo_weight = True
                break
        if not find_topo_weight:
            raise ValueError(f"Topo weight {self.topo_weight_name} not found in F_form.")
            
        structure_source_term = self.alpha_fun(
            rho=topo_weight.x.array.copy(),
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            p=self.alpha_p,
            pow_operator=np.power
        )
        
        if self.cache_memory is not None:
            init_vtu = self.cache_memory.get_cache('init_vtu')
        else:
            init_vtu = None
        
        if self.fluid_type == 'stoke':
            cell_velocity, cell_pressure, cache_memory = run_stoke_fluid_simulation(
                boundaries=self.boundaries,
                msh_file=self.msh_file,
                velocityRelaxation=self.velocityRelaxation,
                pressureRelaxation=self.pressureRelaxation,
                log_step=self.log_step,
                hot_steps=self.hot_steps,
                simulation_iterations=self.simulation_iterations,
                convergence_threshold=self.convergence_threshold,
                save_dir=self.save_dir,
                use_tqdm=self.use_tqdm,
                use_log=self.use_log,
                parallel_num=self.parallel_num,
                structure_mesh=self.structure_mesh,
                structure_viscosity=self.viscosity,
                structure_source_term=structure_source_term,
                structure_obs_weight=self.structure_obs_weight,
                force_msh_init=True,
                init_vtu=init_vtu,
            )
        else:
            raise ValueError(f"Invalid fluid type: {self.fluid_type}")
        
        if self.cache_memory is not None:
            self.cache_memory = cache_memory
        
        uh = convert_fipy_result_to_dolfinx(
            mesh=self.structure_mesh,
            cell_velocity=cell_velocity,
            cell_pressure=cell_pressure,
            boundaries=self.boundaries,
            V_func_space=self.V_func_space,
            Q_func_space=self.Q_func_space,
            uh=uh,
            V_to_W_dofs=self.V_to_W_dofs,
            Q_to_W_dofs=self.Q_to_W_dofs,
        )
        
        # very terrible
        # if self.fluid_type == 'stoke':
        #     uh = refit_to_dolfinx(
        #         mesh=self.structure_mesh,
        #         F_form=self.F_form,
        #         state=self.state,
        #         adjoint=self.adjoint,
        #         bcs=[bry_info.bry_cell for bry_info in self.boundaries], # type: ignore
        #         forward_ksp_option=self.forward_ksp_option,
        #         forward_kwargs=self.forward_kwargs,
        #         is_linear=True,
        #     )
        # else:
        #     raise ValueError(f"Invalid fluid type: {self.fluid_type}")
        
        return uh
 