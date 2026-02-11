import os
import numpy as np
import pyvista
from mpi4py import MPI

from scripts_py.auto_dolfinx.opt_solver.topo_fluid_hierarchy import AutoFluidBoundary
from scripts_py.auto_dolfinx.opt_solver.topo_fluid_hierarchy import TopoFluidHierarchyOptimizer


base_dir = '/home/admin123456/Desktop/work/gmsh_test/t50_3'

right_boundary = 4.0
top_boundary = 4.0


def generate_top_pipes(num_pipes: int, start_marker: int, face_radius=0.1):
    face_norm_vector=np.array([0.0, 1.0])
    # face_centers = np.array([])
    # while len(face_centers) < num_pipes:
    #     random_center = np.array([0.0, top_boundary])
    #     random_center[0] = np.random.uniform(face_radius + 0.05, right_boundary - face_radius - 0.05, size=(1,))
    #     if face_centers.shape[0] == 0:
    #         face_centers = np.array([random_center])
    #     else:
    #         dist_to_prev = np.linalg.norm(face_centers - random_center, axis=1)
    #         if np.min(dist_to_prev) > 2 * face_radius + 0.15:
    #             face_centers = np.concatenate([face_centers, [random_center]], axis=0)
    
    if num_pipes == 1:
        face_centers = np.array([[right_boundary / 2.0, top_boundary]])
    else:
        face_centers = np.ones((num_pipes, 2)) * top_boundary
        face_centers[:, 0] = np.linspace(face_radius * 4.0, right_boundary - face_radius * 4.0, num_pipes)
    
    marker_series = np.arange(start_marker + 1, start_marker + num_pipes + 1)
    bry_list = []
    for face_center, marker in zip(face_centers, marker_series):
        bry_list.append(AutoFluidBoundary(
            name=f'outlet{marker}',
            boundary_type='pressure',
            value=0.0,
            marker=int(marker),
            face_center=face_center,
            face_norm_vector=face_norm_vector,
            face_radius=face_radius,
        ))
    return bry_list, int(marker_series[-1])


def generate_left_pipes(num_pipes: int, start_marker: int, face_radius=0.1):
    face_norm_vector=np.array([-1.0, 0.0])
    # face_centers = np.array([])
    # while len(face_centers) < num_pipes:
    #     random_center = np.array([0.0, 0.0])
    #     random_y = np.random.uniform(face_radius + 0.05, top_boundary - face_radius - 0.05, size=(1,))[0]
    #     # if random_y < 0.5:
    #     #     continue
    #     random_center[1] = random_y
    #     if face_centers.shape[0] == 0:
    #         face_centers = np.array([random_center])
    #     else:
    #         dist_to_prev = np.linalg.norm(face_centers - random_center, axis=1)
    #         if np.min(dist_to_prev) > 2 * face_radius + 0.15:
    #             face_centers = np.concatenate([face_centers, [random_center]], axis=0)
    
    if num_pipes == 1:
        face_centers = np.array([[0.0, top_boundary / 2.0]])
    else:
        face_centers = np.zeros((num_pipes, 2))
        face_centers[:, 1] = np.linspace(face_radius * 4.0, top_boundary - face_radius * 4.0, num_pipes)
    
    marker_series = np.arange(start_marker + 1, start_marker + num_pipes + 1)
    bry_list = []
    for face_center, marker in zip(face_centers, marker_series):
        bry_list.append(AutoFluidBoundary(
            name=f'outlet{marker}',
            boundary_type='pressure',
            value=0.0,
            marker=int(marker),
            face_center=face_center,
            face_norm_vector=face_norm_vector,
            face_radius=face_radius,
        ))
    return bry_list, int(marker_series[-1])


def generate_right_pipes(num_pipes: int, start_marker: int, face_radius=0.1):
    face_norm_vector=np.array([1.0, 0.0])
    # face_centers = np.array([])
    # while len(face_centers) < num_pipes:
    #     random_center = np.array([right_boundary, 0.0])
    #     random_y = np.random.uniform(face_radius + 0.05, top_boundary - face_radius - 0.05, size=(1,))[0]
    #     # if random_y < 0.5:
    #     #     continue
    #     random_center[1] = random_y
    #     if face_centers.shape[0] == 0:
    #         face_centers = np.array([random_center])
    #     else:
    #         dist_to_prev = np.linalg.norm(face_centers - random_center, axis=1)
    #         if np.min(dist_to_prev) > 2 * face_radius + 0.15:
    #             face_centers = np.concatenate([face_centers, [random_center]], axis=0)
    
    if num_pipes == 1:
        face_centers = np.array([[right_boundary, top_boundary / 2.0]])
    else:
        face_centers = np.ones((num_pipes, 2)) * right_boundary
        face_centers[:, 1] = np.linspace(face_radius * 4.0, top_boundary - face_radius * 4.0, num_pipes)
    
    marker_series = np.arange(start_marker + 1, start_marker + num_pipes + 1)
    bry_list = []
    for face_center, marker in zip(face_centers, marker_series):
        bry_list.append(AutoFluidBoundary(
            name=f'outlet{marker}',
            boundary_type='pressure',
            value=0.0,
            marker=int(marker),
            face_center=face_center,
            face_norm_vector=face_norm_vector,
            face_radius=face_radius,
        ))
    return bry_list, int(marker_series[-1])


boundaries = [
    AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[0.0, 1.0],
        marker=1,
        face_center=np.array([right_boundary / 2.0, 0.0]),
        face_norm_vector=np.array([0.0, 1.0]),
        face_radius=0.1
    )
]

start_marker = 1
top_pipes, start_marker = generate_top_pipes(9, start_marker)
left_pipes, start_marker = generate_left_pipes(9, start_marker)
right_pipes, start_marker = generate_right_pipes(9, start_marker)
boundaries.extend(top_pipes)
boundaries.extend(left_pipes)
boundaries.extend(right_pipes)

boundaries.append(
    AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=start_marker + 1
    )
)

outlet_markers = []
for bry in boundaries:
    if bry.boundary_type == 'pressure':
        outlet_markers.append(bry.marker)
outlet_marker_ratios = np.ones(len(outlet_markers)) / len(outlet_markers)
outlet_marker_weights = np.ones(len(outlet_markers))
print('outlet_markers:', outlet_markers)
print('outlet_marker_ratios:', outlet_marker_ratios)
print('outlet_marker_weights:', outlet_marker_weights)

target_ratio = 0.2
target_volume = (right_boundary * top_boundary) * target_ratio
optim = TopoFluidHierarchyOptimizer(
    name='t49',
    run_dir=base_dir,
    space_dim=2,
    space_left_bottom_corner=np.array([0.0, 0.0]),
    space_right_top_corner=np.array([right_boundary, top_boundary]),
    space_resolutions=[0.1],
    mesh_min_sizes=[0.03],
    mesh_max_sizes=[0.06],
    boundary_list=boundaries,
    fluid_type='stoke',
    viscosity_init=1.0 / 400.,
    # topo_weight_init=max(target_ratio, 0.3),
    topo_weight_init=0.8,
    us_fipy=True,
    diffusion_step=0,
    init_topo_weight_method='raw',
    extract_threshold=0.4,
    alpha_fun={
        'name': 'exponential',
        'args': {
            'alpha_min': 1e-12,
            'alpha_max': 300.0,
            'p': 3.0,
        }
    },
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
                    'outlet_marker_weights': (outlet_marker_weights * 1e-1).tolist(),
                    'with_rescale': True,
                    'with_max_dif_cost': False,
                    'max_dif_weight': 5e-2,
                },
            ],
            'reg_cost_args':{
                'style': 'Volume',
                'weight': 1e-2,
                'target_volume': target_volume,
                'reg_rescale': True,
            },
            'obstacle_cost_args':None,
            'dolfinx_simulator_cfg': {
                'max_iteration': 300,
                'line_search_steps': [0.1],
                'momentum_weight': 0.8,
                'full_run': False,
                'hot_steps': 80,
                'use_lr_scheduler': False,
                'min_lr': 1e-2,
                'max_lr': 1e-1,
                'loss_decay_weight': 0.0,
            },
            'fipy_simulator_cfg': {
                'use_lr_scheduler': True,
                'min_lr': 1e-2,
                'max_lr': 1e-1,
                'loss_decay_weight': 0.0,
                'simulation_iterations': 200,
                'convergence_threshold': 1e-4,
                'velocityRelaxation': 0.6,
                'pressureRelaxation': 0.8,
                'hot_steps': 50,
                'simulation_hot_steps': 4,
                'log_step': 2,
                'parallel_num': 8,
                'use_tqdm': True,
                'use_log': True,
                'max_iteration': 300,
                'line_search_steps': [0.1],
                'momentum_weight': 0.8,
                'full_run': True,
            },
        },
    ]
)
