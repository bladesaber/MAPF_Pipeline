import os
import numpy as np
import pyvista

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.auto_dolfinx.opt_solver.topo_fluid import TopoFluidOptimizer, AutoFluidBoundary
from scripts_py.vtu_recorder import TensorBoardRecorder
from thirdparty.pyadjoint.pyadjoint import Tape
from scripts_py.auto_dolfinx.opt_solver.topo_fluid_fipy import TopoFluidOptimizerFipy
from scripts_py.simulator.fipy_simulator import FipyStokeSimulator, FluidBoundary
from scripts_py.auto_dolfinx.opt_solver.topo_fluid_hierarchy import TopoFluidHierarchyOptimizer


base_dir = '/home/admin123456/Desktop/work/gmsh_test/t48'

boundaries = [
    AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[0.0, 0.0, 1.0],
        marker=1,
        face_center=np.array([0.5, 0.5, 0.0]),
        face_norm_vector=np.array([0.0, 0.0, 1.0]),
        face_radius=0.15
    ),
    AutoFluidBoundary(
        name='outlet1',
        boundary_type='pressure',
        value=0.0,
        marker=2,
        face_center=np.array([0.15, 0.15, 1.0]),
        face_norm_vector=np.array([0.0, 0.0, 1.0]),
        face_radius=0.05
    ),
    AutoFluidBoundary(
        name='outlet2',
        boundary_type='pressure',
        value=0.0,
        marker=3,
        face_center=np.array([0.85, 0.85, 1.0]),
        face_norm_vector=np.array([0.0, 0.0, 1.0]),
        face_radius=0.05
    ),
    AutoFluidBoundary(
        name='outlet3',
        boundary_type='pressure',
        value=0.0,
        marker=4,
        face_center=np.array([0.15, 0.85, 1.0]),
        face_norm_vector=np.array([0.0, 0.0, 1.0]),
        face_radius=0.05
    ),
    AutoFluidBoundary(
        name='outlet4',
        boundary_type='pressure',
        value=0.0,
        marker=5,
        face_center=np.array([0.85, 0.15, 1.0]),
        face_norm_vector=np.array([0.0, 0.0, 1.0]),
        face_radius=0.05
    ),
    
    AutoFluidBoundary(
        name='outlet5',
        boundary_type='pressure',
        value=0.0,
        marker=6,
        face_center=np.array([0.35, 0.5, 1.0]),
        face_norm_vector=np.array([0.0, 0.0, 1.0]),
        face_radius=0.05
    ),
    AutoFluidBoundary(
        name='outlet6',
        boundary_type='pressure',
        value=0.0,
        marker=7,
        face_center=np.array([0.65, 0.5, 1.0]),
        face_norm_vector=np.array([0.0, 0.0, 1.0]),
        face_radius=0.05
    ),
    AutoFluidBoundary(
        name='outlet7',
        boundary_type='pressure',
        value=0.0,
        marker=8,
        face_center=np.array([0.5, 0.35, 1.0]),
        face_norm_vector=np.array([0.0, 0.0, 1.0]),
        face_radius=0.05
    ),
    AutoFluidBoundary(
        name='outlet8',
        boundary_type='pressure',
        value=0.0,
        marker=9,
        face_center=np.array([0.5, 0.65, 1.0]),
        face_norm_vector=np.array([0.0, 0.0, 1.0]),
        face_radius=0.05
    ),
    
    AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=10
    )
]

optim = TopoFluidHierarchyOptimizer(
    name='t48',
    run_dir=base_dir,
    space_dim=3,
    space_left_bottom_corner=np.array([0.0, 0.0, 0.0]),
    space_right_top_corner=np.array([1.0, 1.0, 1.0]),
    space_resolutions=[0.15, 0.1, 0.05],
    mesh_min_sizes=[0.03, 0.02, 0.01],
    mesh_max_sizes=[0.08, 0.04, 0.02],
    boundary_list=boundaries,
    fluid_type='stoke',
    viscosity_init=1.0 / 400.,
    topo_weight_init=0.2,
    init_topo_weight_method='poisson',
    us_fipy=True,
    extract_threshold=0.35,
    diffusion_step=0,
)
optim.optimize(
    run_cfgs=[
        {
            'obj_cost_args_list':[
                {
                    'style': 'MinDissipation+BrinkmanPenalized',
                    'with_viscosity': True,
                    'dissipation_weight': 1.0,
                    'penalized_weight': 1.0,
                },
                {
                    'style': 'FixedFlux',
                    'inlet_markers': [1],
                    'outlet_markers': [2, 3, 4, 5, 6, 7, 8, 9],
                    'outlet_marker_ratios': [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    'outlet_marker_weights': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # TODO: shouldn't be main cost
                    'with_rescale': True
                }
            ],
            'reg_cost_args':{
                'style': 'Volume',
                'weight': 0.5,
                'target_volume': 0.2,
                'reg_rescale': True,
            },
            'obstacle_cost_args':None,
            'fipy_simulator_cfg': {
                'simulation_iterations': 200,
                'convergence_threshold': 1e-4,
                'velocityRelaxation': 0.6,
                'pressureRelaxation': 0.8,
                'hot_steps': 10,
                'log_step': 2,
                'parallel_num': 8,
                'use_tqdm': True,
                'use_log': True,
                'max_iteration': 300,
                'line_search_steps': [0.1, 0.05],
                'momentum_weight': 0.8,
                'full_run': False,
            },
        },
        {
            'reg_cost_args': {
                'style': 'Volume',
                'weight': 0.75,
                'target_volume': 0.1,
                'reg_rescale': True,
            },
        },
        {
            'reg_cost_args': {
                'style': 'Volume',
                'weight': 1.0,
                'target_volume': 0.06,
                'reg_rescale': True,
            },
            'fipy_simulator_cfg': {
                'simulation_iterations': 200,
                'convergence_threshold': 1e-4,
                'velocityRelaxation': 0.6,
                'pressureRelaxation': 0.8,
                'hot_steps': 10,
                'log_step': 2,
                'parallel_num': 8,
                'use_tqdm': True,
                'use_log': True,
                'max_iteration': 300,
                'line_search_steps': [0.05],
                'momentum_weight': 0.8,
                'full_run': False,
            },
        }
    ],
    residual_error_threshold=1e-2,
    inout_pipe_length=0.03
)

