import os
import numpy as np
import pyvista

from scripts_py.auto_dolfinx.opt_solver.topo_fluid import AutoFluidBoundary
from scripts_py.auto_dolfinx.opt_solver.topo_fluid_hierarchy import TopoFluidHierarchyOptimizer


"""TODO: 仿真工具不足够鲁棒性,关于累积->决策->迈步的加速收敛方法
NOTE: FixedFlux本身的震荡很严重,所以必须是最小的,否则无法收敛.该问题主要是平衡reg_cost与FixedFlux之间的权重.
"""

base_dir = '/home/admin123456/Desktop/work/gmsh_test/t48_fipy_2'

right_boundary = 1.0
back_boundary = 1.0
top_boundary = 1.0


def generate_top_pipes(num_pipes: int, start_marker: int, face_radius=0.05):
    face_norm_vector=np.array([0.0, 0.0, 1.0])
    face_centers = np.array([])
    while len(face_centers) < num_pipes:
        random_center = np.array([0.0, 0.0, top_boundary])
        random_center[:2] = np.random.uniform(face_radius + 0.05, 1.0 - face_radius - 0.05, size=(2,))
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
        random_center = np.array([0.0, 0.0, 0.0])
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
        random_center = np.array([0.0, back_boundary, 0.0])
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
        random_center = np.array([0., 0., 0.])
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
        random_center = np.array([right_boundary, 0.0, 0.0])
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

top_pipes, start_marker = generate_top_pipes(4, start_marker)
front_pipes, start_marker = generate_front_pipes(4, start_marker)
back_pipes, start_marker = generate_back_pipes(4, start_marker)
left_pipes, start_marker = generate_left_pipes(4, start_marker)
right_pipes, start_marker = generate_right_pipes(4, start_marker)

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
outlet_marker_ratios = np.ones(len(outlet_markers)) / len(outlet_markers)
outlet_marker_weights = np.ones(len(outlet_markers))

assert len(outlet_markers) == num_output_pipes, 'Number of output pipes does not match'
assert outlet_marker_ratios.shape[0] == num_output_pipes, 'Number of output pipe ratios does not match'
assert outlet_marker_weights.shape[0] == num_output_pipes, 'Number of output pipe weights does not match'
print('outlet_markers:', outlet_markers)
print('outlet_marker_ratios:', outlet_marker_ratios)
print('outlet_marker_weights:', outlet_marker_weights)

target_ratio = 0.05
target_volume = (right_boundary * top_boundary * back_boundary) * target_ratio
optim = TopoFluidHierarchyOptimizer(
    name='t48',
    run_dir=base_dir,
    space_dim=3,
    space_left_bottom_corner=np.array([0.0, 0.0, 0.0]),
    space_right_top_corner=np.array([1.0, 1.0, 1.0]),
    space_resolutions=[0.15],
    mesh_min_sizes=[0.02],
    mesh_max_sizes=[0.08],
    boundary_list=boundaries,
    fluid_type='stoke',
    viscosity_init=1.0 / 400.,
    topo_weight_init=max(target_ratio, 0.3),
    init_topo_weight_method='raw',
    us_fipy=True,
    extract_threshold=0.4,
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
                    'outlet_markers': outlet_markers,
                    'outlet_marker_ratios': outlet_marker_ratios.tolist(),
                    # NOTE: FixedFlux的振荡会造成无法全部管道都不相连,互相约束.但过小的weight会造成无法均衡. 所以这里的大小必须严格测试
                    'outlet_marker_weights': (outlet_marker_weights * 1e-1).tolist(),
                    'with_rescale': True,
                    'with_max_dif_cost': False,
                    'max_dif_weight': 1e-3,
                },
            ],
            'reg_cost_args':{
                'style': 'Volume',
                'weight': 1e-1,
                'target_volume': target_volume,
                'reg_rescale': True,
            },
            'obstacle_cost_args':None,
            'fipy_simulator_cfg': {
                'simulation_iterations': 200,
                'convergence_threshold': 1e-4,
                'velocityRelaxation': 0.6,
                'pressureRelaxation': 0.8,
                'hot_steps': 80,
                'simulation_hot_steps': 4,
                'log_step': 2,
                'parallel_num': 8,
                'use_tqdm': True,
                'use_log': True,
                'max_iteration': 1000,
                'line_search_steps': [0.05],
                'momentum_weight': 0.8,
                'full_run': True,
            },
            'dolfinx_simulator_cfg': {
                'max_iteration': 300,
                'line_search_steps': [0.2, 0.1, 0.05],
                'momentum_weight': 0.8,
                'full_run': False,
                'hot_steps': 80,
            },
        }
    ],
    residual_error_threshold=1e-3,
    inout_pipe_length=0.03
)

