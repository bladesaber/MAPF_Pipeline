from typing import Union, Callable, Any, Literal

import dolfinx
from matplotlib import use
import ufl
from ufl.algorithms import expand_derivatives
from ufl.algorithms.analysis import extract_arguments

import numpy as np
from petsc4py import PETSc
import pyvista

from thirdparty.pyadjoint.pyadjoint.block import Block
from thirdparty.pyadjoint.pyadjoint.overloaded_type import get_overloaded_class, create_overloaded_object
from thirdparty.pyadjoint.pyadjoint.tape import (
    Tape,
    annotate_tape, 
    get_working_tape, 
    no_annotations,
    stop_annotating
)

from scripts_py.dolfinx_utils import BoundaryUtils
from scripts_py.mesh_utils import MeshUtils
from scripts_py.eqn_solver import LinearProblemSolver, NonLinearProblemSolver
from scripts_py.dolfinx_utils import AssembleUtils
from scripts_py.sparse_utils import PetscUtils
from scripts_py.dolfinx_utils import DolfinUtils

from scripts_py.auto_dolfinx.types.type_dirichletBC import DolfinxDirichletBC
from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction

from scripts_py.auto_dolfinx.blocks.block_solve import EquationSolveBlock
from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import AutoFluidBoundary
from scripts_py.auto_dolfinx.cbs_topo.seq_mesh_utils import SeqMeshObj
from scripts_py.auto_dolfinx.cbs_topo.fipy_obstacle_simulator import FipyObstacleTopologySimulator


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
    simulation_iterations: int,
    convergence_threshold: float,
    save_dir: str,
    
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
    with_block: bool = False
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
        
        simulator = FipyObstacleTopologySimulator.from_msh(
            source_msh_file=msh_file,
            dim=structure_mesh.tdim,
            viscosity=viscosity,
            velocityRelaxation=velocityRelaxation,
            pressureRelaxation=pressureRelaxation,
            structure_mesh=structure_mesh,
            structure_alpha=structure_alpha,
            structure_obs_weight=structure_obs_weight,
        )
        for boundary in boundaries:
            simulator.add_boundary(boundary)
        simulator.define_boundary()
        res_report = simulator.run(
            iterations=simulation_iterations,
            convergence_threshold=convergence_threshold,
            log_step=log_step, 
            save_dir=save_dir,
            use_tqdm=False,
            use_log=True
        )
        if not res_report['is_solve']:
            print(f'[Error] x_residual: {res_report['x_residual']} \
                y_residual: {res_report['y_residual']} \
                z_residual: {res_report['z_residual']} \
                p_residual: {res_report['p_residual']} \
                continuity_residual: {res_report['continuity_residual']}')
            raise Exception('Stop')
        
        velocity = simulator.get_velocity(map_to_structure=True)
        pressure = simulator.get_pressure(map_to_structure=True)
        state = convert_result_to_dolfinx(
            domain=structure_mesh.domain,
            uh=state,
            velocity=velocity,
            pressure=pressure,
            V_to_W_dofs=V_to_W_dofs,
            Q_to_W_dofs=Q_to_W_dofs,
            V_func_space=V_func_space,
            Q_func_space=Q_func_space,
        )
        output = state
    
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
            log_step = log_step,
            simulation_iterations = simulation_iterations,
            convergence_threshold = convergence_threshold,
            save_dir = save_dir,
            
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
        log_step: int,
        simulation_iterations: int,
        convergence_threshold: float,
        save_dir: str,
        
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
        self.simulation_iterations = simulation_iterations
        self.convergence_threshold = convergence_threshold
        self.save_dir = save_dir
        
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
                    
        simulator = FipyObstacleTopologySimulator.from_msh(
            source_msh_file=self.msh_file,
            dim=self.structure_mesh.tdim,
            structure_alpha=structure_alpha,
            structure_obs_weight=self.structure_obs_weight,
            viscosity=self.viscosity,
            velocityRelaxation=self.velocityRelaxation,
            pressureRelaxation=self.pressureRelaxation,
            structure_mesh=self.structure_mesh
        )
        for boundary in self.boundaries:
            simulator.add_boundary(boundary)
        simulator.define_boundary()
        res_report = simulator.run(
            iterations=self.simulation_iterations, 
            convergence_threshold=self.convergence_threshold,
            log_step=self.log_step, 
            save_dir=self.save_dir,
            use_tqdm=True,
            use_log=True
        )
        if not res_report['is_solve']:
            print(f'[Error] x_residual: {res_report['x_residual']} \
                y_residual: {res_report['y_residual']} \
                z_residual: {res_report['z_residual']} \
                p_residual: {res_report['p_residual']} \
                continuity_residual: {res_report['continuity_residual']}')
            raise Exception('Stop')
            
        velocity = simulator.get_velocity(map_to_structure=True)
        pressure = simulator.get_pressure(map_to_structure=True)
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
    