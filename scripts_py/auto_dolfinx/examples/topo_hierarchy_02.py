import os
import numpy as np
import pyvista

from scripts_py.auto_dolfinx.opt_solver.topo_fluid_hierarchy import AutoFluidBoundary
from scripts_py.auto_dolfinx.opt_solver.topo_fluid_hierarchy import TopoFluidHierarchyOptimizer


base_dir = '/home/admin123456/Desktop/work/gmsh_test/t49'

boundaries = [
    AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[0.0, 1.0],
        marker=1,
        face_center=np.array([1.0, 0.0]),
        face_norm_vector=np.array([0.0, 1.0]),
        face_radius=0.1
    ),
    
    AutoFluidBoundary(
        name='outlet1',
        boundary_type='pressure',
        value=0.0,
        marker=2,
        face_center=np.array([0.2, 1.0]),
        face_norm_vector=np.array([0.0, 1.0]),
        face_radius=0.1
    ),
    AutoFluidBoundary(
        name='outlet2',
        boundary_type='pressure',
        value=0.0,
        marker=3,
        face_center=np.array([0.6, 1.0]),
        face_norm_vector=np.array([0.0, 1.0]),
        face_radius=0.1
    ),
    AutoFluidBoundary(
        name='outlet3',
        boundary_type='pressure',
        value=0.0,
        marker=4,
        face_center=np.array([1.0, 1.0]),
        face_norm_vector=np.array([0.0, 1.0]),
        face_radius=0.1
    ),
    AutoFluidBoundary(
        name='outlet4',
        boundary_type='pressure',
        value=0.0,
        marker=5,
        face_center=np.array([1.4, 1.0]),
        face_norm_vector=np.array([0.0, 1.0]),
        face_radius=0.1
    ),
    AutoFluidBoundary(
        name='outlet5',
        boundary_type='pressure',
        value=0.0,
        marker=6,
        face_center=np.array([1.8, 1.0]),
        face_norm_vector=np.array([0.0, 1.0]),
        face_radius=0.1
    ),
    
    AutoFluidBoundary(
        name='outlet6',
        boundary_type='pressure',
        value=0.0,
        marker=7,
        face_center=np.array([0.0, 0.4]),
        face_norm_vector=np.array([-1.0, 0.0]),
        face_radius=0.1
    ),
    AutoFluidBoundary(
        name='outlet7',
        boundary_type='pressure',
        value=0.0,
        marker=8,
        face_center=np.array([0.0, 0.85]),
        face_norm_vector=np.array([-1.0, 0.0]),
        face_radius=0.1
    ),
    
    AutoFluidBoundary(
        name='outlet8',
        boundary_type='pressure',
        value=0.0,
        marker=9,
        face_center=np.array([2.0, 0.4]),
        face_norm_vector=np.array([1.0, 0.0]),
        face_radius=0.1
    ),
    AutoFluidBoundary(
        name='outlet9',
        boundary_type='pressure',
        value=0.0,
        marker=10,
        face_center=np.array([2.0, 0.85]),
        face_norm_vector=np.array([1.0, 0.0]),
        face_radius=0.1
    ),
    
    AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=11
    )
]


outlet_markers = []
for bry in boundaries:
    if bry.boundary_type == 'pressure':
        outlet_markers.append(bry.marker)
outlet_marker_ratios = np.ones(len(outlet_markers)) / len(outlet_markers)
outlet_marker_weights = np.ones(len(outlet_markers)) * 0.02
print('outlet_markers:', outlet_markers)
print('outlet_marker_ratios:', outlet_marker_ratios)
print('outlet_marker_weights:', outlet_marker_weights)

optim = TopoFluidHierarchyOptimizer(
    name='t49',
    run_dir=base_dir,
    space_dim=2,
    space_left_bottom_corner=np.array([0.0, 0.0]),
    space_right_top_corner=np.array([2.0, 1.0]),
    space_resolutions=[0.1, 0.05, 0.03, 0.03],
    mesh_min_sizes=[0.03, 0.02, 0.01, 0.005],
    mesh_max_sizes=[0.06, 0.04, 0.02, 0.01],
    boundary_list=boundaries,
    fluid_type='stoke',
    viscosity_init=1.0 / 400.,
    topo_weight_init=0.1,
    us_fipy=False,
    diffusion_step=0,
    
    # --- setting 1
    # init_topo_weight_method='filter',
    # extract_threshold=0.3,
    
    # --- setting 2
    # init_topo_weight_method='poisson',
    # extract_threshold=0.4,
    
    # --- setting 3
    init_topo_weight_method='raw',
    extract_threshold=0.4,
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
                    'outlet_markers': outlet_markers,
                    'outlet_marker_ratios': outlet_marker_ratios.tolist(),
                    'outlet_marker_weights': outlet_marker_weights.tolist(),  # TODO: shouldn't be main cost
                    # 'outlet_marker_ratios': outlet_marker_ratios,  # TODO: why use np.ndarray will cause not converge?
                    # 'outlet_marker_weights': outlet_marker_weights,
                    'with_rescale': True
                }
            ],
            'reg_cost_args':{
                'style': 'Volume',
                'weight': 1.0,
                'target_volume': 0.6,
                'reg_rescale': True,
            },
            'obstacle_cost_args':None,
            'dolfinx_simulator_cfg': {
                'max_iteration': 200,
                'line_search_steps': [0.1, 0.05],
                'momentum_weight': 0.8,
                'full_run': False,
            }
        },
        {
            'reg_cost_args': {
                'style': 'Volume',
                'weight': 0.75,
                'target_volume': 0.5,
                'reg_rescale': True,
            },
        },
        {
            'reg_cost_args': {
                'style': 'Volume',
                'weight': 1.0,
                'target_volume': 0.3,
                'reg_rescale': True,
            },
            'dolfinx_simulator_cfg': {
                'max_iteration': 200,
                'line_search_steps': [0.05],
                'momentum_weight': 0.8,
                'full_run': False,
            }
        },
        {
            'reg_cost_args': {
                'style': 'Volume',
                'weight': 0.25,
                'target_volume': 0.2,
                'reg_rescale': True,
            },
            'dolfinx_simulator_cfg': {
                'max_iteration': 200,
                'line_search_steps': [0.05],
                'momentum_weight': 0.8,
                'full_run': True,
            }
        }
    ]
)