from typing import Union, Callable, Any, Literal
import os
import shutil

import dolfinx
import ufl

import numpy as np
import pyvista
import subprocess
import json

from thirdparty.pyadjoint.pyadjoint.block import Block
from thirdparty.pyadjoint.pyadjoint.overloaded_type import get_overloaded_class, create_overloaded_object
from thirdparty.pyadjoint.pyadjoint.tape import (
    Tape,
    annotate_tape, 
    get_working_tape, 
    no_annotations,
    stop_annotating
)

from scripts_py.dolfinx_utils import DolfinUtils
from scripts_py.auto_dolfinx.blocks.block_solve import EquationSolveBlock
from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import AutoFluidBoundary
from scripts_py.auto_dolfinx.cbs_topo.seq_mesh_utils import SeqMeshObj
from scripts_py.auto_dolfinx.cbs_topo.fipy_obs_simulator import FipyStokeObsTopoSimulator


class CacheMemory(object):
    def __init__(self) -> None:
        self.cache_memory = {}
    
    def update_cache(self, key: str, value: Any) -> None:
        self.cache_memory[key] = value
    
    def get_cache(self, key: str) -> Any:
        return self.cache_memory[key] if key in self.cache_memory else None
    
    def clear_cache(self) -> None:
        self.cache_memory.clear()


def convert_result_to_dolfinx(
    domain: dolfinx.mesh.Mesh, 
    uh: dolfinx.fem.Function, 
    velocity: np.ndarray, 
    pressure: np.ndarray,
    V_to_W_dofs: np.ndarray,
    Q_to_W_dofs: np.ndarray,
    V_func_space: dolfinx.fem.FunctionSpace,
    Q_func_space: dolfinx.fem.FunctionSpace,
):
    cell_vector_func_space = DolfinUtils.cell_vector_function_space(domain)
    velocity_fun = dolfinx.fem.Function(cell_vector_func_space)
    velocity_fun.x.array[:] = velocity.reshape((-1,))  # type: ignore
    
    cell_scalar_func_space = DolfinUtils.cell_scalar_function_space(domain)
    pressure_fun = dolfinx.fem.Function(cell_scalar_func_space)
    pressure_fun.x.array[:] = pressure.reshape((-1,))  # type: ignore
    
    # # ------ visualize result ------
    # plotter = pyvista.Plotter()
    # grid = MeshUtils.mesh_to_vtk(domain)
    # grid.cell_data['velocity'] = velocity_fun.x.array.copy().reshape((-1, 3))  # type: ignore
    # plotter.add_arrows(grid.cell_centers().points, grid.cell_data['velocity'], mag=0.3)
    # plotter.show()
    # # ------ visualize result ------
    
    velocity_interpolate_fun = dolfinx.fem.Function(V_func_space)
    velocity_interpolate_fun.interpolate(velocity_fun)  # type: ignore
        
    pressure_interpolate_fun = dolfinx.fem.Function(Q_func_space)
    pressure_interpolate_fun.interpolate(pressure_fun)  # type: ignore
                
    uh.x.array[V_to_W_dofs] = velocity_interpolate_fun.x.array.copy()  # type: ignore
    uh.x.array[Q_to_W_dofs] = pressure_interpolate_fun.x.array.copy()  # type: ignore

    return uh


def run_fipy_simulation(
    boundaries: list[AutoFluidBoundary],
    msh_file: str,
    viscosity: float,
    velocityRelaxation: float,
    pressureRelaxation: float,
    structure_mesh: SeqMeshObj,
    structure_obs_weight: np.ndarray|None,
    structure_alpha: np.ndarray|None,
    log_step: int,
    hot_steps: int,
    simulation_iterations: int,
    convergence_threshold: float,
    save_dir: str,
    use_tqdm: bool,
    use_log: bool,
    parallel_num: int,
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
    
    if structure_alpha is not None:
        init_vtu.cell_data['alpha'] = structure_alpha
    if structure_obs_weight is not None:
        init_vtu.cell_data['obstacle'] = structure_obs_weight
    
    init_vtu_file = os.path.join(save_dir, 'init.vtu')
    init_vtu.save(init_vtu_file)
    
    FipyStokeObsTopoSimulator.create_simulation_config(
        cfg_file=cfg_file,
        source_msh_file=source_msh_file,
        dim=structure_mesh.tdim,
        viscosity=viscosity,
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
        structure_init_vtu_file=init_vtu_file,
        vtu_velocity_field='velocity',
        vtu_pressure_field='pressure',
        vtu_alpha_field='alpha',
        vtu_obs_weight_field='obstacle',
    )
    
    if parallel_num > 1:
        cmd_str = f'mpirun -n {parallel_num} python scripts_py/auto_dolfinx/cbs_topo/fipy_obs_simulator.py --cfg_file {cfg_file} --force_msh_init {force_msh_init}'
    else:
        cmd_str = f'python scripts_py/auto_dolfinx/cbs_topo/fipy_obs_simulator.py --cfg_file {cfg_file} --force_msh_init {force_msh_init}'
    
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


def equation_stokeEqn_solve(
    # ------ simulator related ------
    boundaries: list[AutoFluidBoundary],
    msh_file: str,
    viscosity: float,
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
        
        structure_alpha = alpha_fun(
            rho=topo_weight.x.array.copy(),
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            p=alpha_p,
            pow_operator=np.power
        )
        
        cell_velocity, cell_pressure, cache_memory = run_fipy_simulation(
            boundaries=boundaries,
            msh_file=msh_file,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            structure_mesh=structure_mesh,
            structure_obs_weight=structure_obs_weight,
            structure_alpha=structure_alpha,
            log_step=log_step,
            hot_steps=hot_steps,
            simulation_iterations=simulation_iterations,
            convergence_threshold=convergence_threshold,
            save_dir=save_dir,
            use_tqdm=use_tqdm,
            use_log=use_log,
            parallel_num=parallel_num,
            force_msh_init=True,
            init_vtu=None,
        )
                
        state = convert_result_to_dolfinx(
            domain=structure_mesh.domain,
            uh=state,
            velocity=cell_velocity,
            pressure=cell_pressure,
            V_to_W_dofs=V_to_W_dofs,
            Q_to_W_dofs=Q_to_W_dofs,
            V_func_space=V_func_space,
            Q_func_space=Q_func_space,
        )
        output = state
        
        if not with_cache:
            cache_memory.clear_cache()
            cache_memory = None
    
    if with_annotate:
        block = StockEquationSolveBlock(
            # ------ simulator related ------
            boundaries = boundaries,
            msh_file = msh_file,
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
        )
        
        if tape is None:
            tape = get_working_tape()
        tape.add_block(block)
        block.add_output(output.create_block_variable())  # type: ignore
    
    if with_block and with_annotate:
        return output, block
    return output


class StockEquationSolveBlock(EquationSolveBlock):
    def __init__(
        self,
        # ------ simulator related ------
        boundaries: list[AutoFluidBoundary],
        msh_file: str,
        viscosity: float,
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
            forward_ksp_option={},
            tlm_kwargs=tlm_kwargs,
            adj_kwargs=adj_kwargs,
            forward_kwargs={},
        )
        
        # ------ simulator related ------
        self.boundaries = boundaries
        self.msh_file = msh_file
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
            
        structure_alpha = self.alpha_fun(
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
                
        velocity, pressure, cache_memory = run_fipy_simulation(
            boundaries=self.boundaries,
            msh_file=self.msh_file,
            viscosity=self.viscosity,
            velocityRelaxation=self.velocityRelaxation,
            pressureRelaxation=self.pressureRelaxation,
            structure_mesh=self.structure_mesh,
            structure_obs_weight=self.structure_obs_weight,
            structure_alpha=structure_alpha,
            log_step=self.log_step,
            hot_steps=self.hot_steps,
            simulation_iterations=self.simulation_iterations,
            convergence_threshold=self.convergence_threshold,
            save_dir=self.save_dir,
            use_tqdm=self.use_tqdm,
            use_log=self.use_log,
            parallel_num=self.parallel_num,
            force_msh_init=True,
            init_vtu=init_vtu,
        )
        if self.cache_memory is not None:
            self.cache_memory = cache_memory
        
        uh = convert_result_to_dolfinx(
            domain=self.domain,
            uh=uh,
            velocity=velocity,
            pressure=pressure,
            V_to_W_dofs=self.V_to_W_dofs,
            Q_to_W_dofs=self.Q_to_W_dofs,
            V_func_space=self.V_func_space,
            Q_func_space=self.Q_func_space,
        )
        return uh
    