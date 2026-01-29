import os
import numpy as np
import pyvista

from scripts_py.auto_dolfinx.opt_solver.topo_fluid import AutoFluidBoundary
from scripts_py.auto_dolfinx.opt_solver.topo_fluid_hierarchy import TopoFluidHierarchyOptimizer


"""仿真工具不足够鲁棒性"""

base_dir = '/home/admin123456/Desktop/work/gmsh_test/t50'


def generate_top_pipes(num_pipes: int, start_marker: int, face_radius=0.05):
    face_norm_vector=np.array([0.0, 0.0, 1.0])
    face_centers = np.array([])
    while len(face_centers) < num_pipes:
        random_center = np.ones(3)
        random_xy = np.random.uniform(face_radius + 0.05, 1.0 - face_radius - 0.05, size=(2,))
        random_center[:2] = random_xy
        if face_centers.shape[0] == 0:
            face_centers = np.array([random_center])
        else:
            dist_to_prev = np.linalg.norm(face_centers - random_center, axis=1)
            if np.min(dist_to_prev) > 2 * face_radius + face_radius:
                face_centers = np.concatenate([face_centers, [random_center]], axis=0)
    
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


def generate_front_pipes(num_pipes: int, start_marker: int, face_radius=0.05):
    face_norm_vector=np.array([0.0, -1.0, 0.0])
    face_centers = np.array([])
    while len(face_centers) < num_pipes:
        random_center = np.zeros(3)
        random_xz = np.random.uniform(face_radius + 0.05, 1.0 - face_radius - 0.05, size=(2,))
        if random_xz[1] < 0.4:
            continue
        random_center[0] = random_xz[0]
        random_center[2] = random_xz[1]

        if face_centers.shape[0] == 0:
            face_centers = np.array([random_center])
        else:
            dist_to_prev = np.linalg.norm(face_centers - random_center, axis=1)
            if np.min(dist_to_prev) > 2 * face_radius + face_radius:
                face_centers = np.concatenate([face_centers, [random_center]], axis=0)
    
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


def generate_back_pipes(num_pipes: int, start_marker: int, face_radius=0.05):
    face_norm_vector=np.array([0.0, 1.0, 0.0])
    face_centers = np.array([])
    while len(face_centers) < num_pipes:
        random_center = np.ones(3)
        random_xz = np.random.uniform(face_radius + 0.05, 1.0 - face_radius - 0.05, size=(2,))
        if random_xz[1] < 0.4:
            continue
        random_center[0] = random_xz[0]
        random_center[2] = random_xz[1]

        if face_centers.shape[0] == 0:
            face_centers = np.array([random_center])
        else:
            dist_to_prev = np.linalg.norm(face_centers - random_center, axis=1)
            if np.min(dist_to_prev) > 2 * face_radius + face_radius:
                face_centers = np.concatenate([face_centers, [random_center]], axis=0)
    
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


def generate_left_pipes(num_pipes: int, start_marker: int, face_radius=0.05):
    face_norm_vector=np.array([-1.0, 0.0, 0.0])
    face_centers = np.array([])
    while len(face_centers) < num_pipes:
        random_center = np.zeros(3)
        random_yz = np.random.uniform(face_radius + 0.05, 1.0 - face_radius - 0.05, size=(2,))
        if random_yz[1] < 0.4:
            continue
        random_center[1] = random_yz[0]
        random_center[2] = random_yz[1]

        if face_centers.shape[0] == 0:
            face_centers = np.array([random_center])
        else:
            dist_to_prev = np.linalg.norm(face_centers - random_center, axis=1)
            if np.min(dist_to_prev) > 2 * face_radius + face_radius:
                face_centers = np.concatenate([face_centers, [random_center]], axis=0)
    
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


def generate_right_pipes(num_pipes: int, start_marker: int, face_radius=0.05):
    face_norm_vector=np.array([1.0, 0.0, 0.0])
    face_centers = np.array([])
    while len(face_centers) < num_pipes:
        random_center = np.ones(3)
        random_yz = np.random.uniform(face_radius + 0.05, 1.0 - face_radius - 0.05, size=(2,))
        if random_yz[1] < 0.4:
            continue
        random_center[1] = random_yz[0]
        random_center[2] = random_yz[1]

        if face_centers.shape[0] == 0:
            face_centers = np.array([random_center])
        else:
            dist_to_prev = np.linalg.norm(face_centers - random_center, axis=1)
            if np.min(dist_to_prev) > 2 * face_radius + face_radius:
                face_centers = np.concatenate([face_centers, [random_center]], axis=0)
    
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
        value=[0.0, 0.0, 1.0],
        marker=1,
        face_center=np.array([0.5, 0.5, 0.0]),
        face_norm_vector=np.array([0.0, 0.0, 1.0]),
        face_radius=0.1
    )
]

start_marker = 1

top_pipes, start_marker = generate_top_pipes(6, start_marker)
front_pipes, start_marker = generate_front_pipes(6, start_marker)
back_pipes, start_marker = generate_back_pipes(6, start_marker)
left_pipes, start_marker = generate_left_pipes(6, start_marker)
right_pipes, start_marker = generate_right_pipes(6, start_marker)

boundaries.extend(top_pipes)
boundaries.extend(front_pipes)
boundaries.extend(back_pipes)
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

num_output_pipes = 0
outlet_markers = []
for bry in boundaries:
    if bry.boundary_type == 'pressure':
        outlet_markers.append(bry.marker)
        num_output_pipes += 1
outlet_marker_ratios = np.ones(num_output_pipes) / num_output_pipes
outlet_marker_weights = 1.0 * np.ones(num_output_pipes) / num_output_pipes

assert len(outlet_markers) == num_output_pipes, 'Number of output pipes does not match'
assert outlet_marker_ratios.shape[0] == num_output_pipes, 'Number of output pipe ratios does not match'
assert outlet_marker_weights.shape[0] == num_output_pipes, 'Number of output pipe weights does not match'
print('outlet_markers:', outlet_markers)
print('outlet_marker_ratios:', outlet_marker_ratios)
print('outlet_marker_weights:', outlet_marker_weights)


optim = TopoFluidHierarchyOptimizer(
    name='t48',
    run_dir=base_dir,
    space_dim=3,
    space_left_bottom_corner=np.array([0.0, 0.0, 0.0]),
    space_right_top_corner=np.array([1.0, 1.0, 1.0]),
    space_resolutions=[0.15, 0.075, 0.05],
    mesh_min_sizes=[0.02, 0.02, 0.01],
    mesh_max_sizes=[0.08, 0.04, 0.02],
    boundary_list=boundaries,
    fluid_type='stoke',
    viscosity_init=1.0 / 400.,
    topo_weight_init=0.2,
    init_topo_weight_method='filter',  # TODO: in 3D example, it seem filter is the mose stable
    us_fipy=True,
    extract_threshold=0.3,
    diffusion_step=0,
)
optim.optimize(
    run_cfgs=[
        {
            'obj_cost_args_list':[
                {
                    'style': 'MinDissipation+BrinkmanPenalized',
                    'with_viscosity': True,
                    'dissipation_weight': 0.1,  # TODO: 试下再缩小这个
                    'penalized_weight': 1.0,
                },
                {
                    'style': 'FixedFlux',
                    'inlet_markers': [1],
                    'outlet_markers': outlet_markers,
                    'outlet_marker_ratios': outlet_marker_ratios.tolist(),
                    'outlet_marker_weights': outlet_marker_weights.tolist(),
                    'with_rescale': True
                }
            ],
            'reg_cost_args':{
                'style': 'Volume',
                'weight': 0.5,
                'target_volume': 0.15,  # TODO: 第一次是不是应该比较大
                'reg_rescale': True,
            },
            'obstacle_cost_args':None,
            'fipy_simulator_cfg': {
                'simulation_iterations': 200,
                'convergence_threshold': 1e-4,
                'velocityRelaxation': 0.6,
                'pressureRelaxation': 0.8,
                'hot_steps': 60,
                'log_step': 2,
                'parallel_num': 8,
                'use_tqdm': True,
                'use_log': True,
                'max_iteration': 500,
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
            'fipy_simulator_cfg': {
                'simulation_iterations': 200,
                'convergence_threshold': 1e-4,
                'velocityRelaxation': 0.6,
                'pressureRelaxation': 0.8,
                'hot_steps': 30,
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
                'weight': 1.0,
                'target_volume': 0.05,
                'reg_rescale': True,
            },
            'fipy_simulator_cfg': {
                'simulation_iterations': 200,
                'convergence_threshold': 1e-4,
                'velocityRelaxation': 0.6,
                'pressureRelaxation': 0.8,
                'hot_steps': 20,
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

