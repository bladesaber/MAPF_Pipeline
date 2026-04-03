
import os
import shutil

import numpy as np
import pyvista
from mpi4py import MPI

from scripts_py.auto_dolfinx.opt_solver.topo_fluid import AutoFluidBoundary
from scripts_py.auto_dolfinx.opt_solver.topo_fluid_shrink import TopoFluidShrinkOptimizer
from scripts_py.auto_dolfinx.opt_solver.grid_mesher import GridMesher


base_dir = '/home/admin123456/Desktop/work/gmsh_test/t60_2'
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(base_dir, exist_ok=True)

right_boundary = 1.5
top_boundary = 1.5


def prepare_design_space(
    space_dim: int, 
    space_left_bottom_corner: np.ndarray, 
    space_right_top_corner: np.ndarray, 
    space_resolution: float, 
    boundary_list: list[AutoFluidBoundary], 
    inout_pipe_length: float,
    geo_file_path: str,
    msh_file_path: str,
    mesh_min_size: float,
    mesh_max_size: float,
    dist_max_ratio: float = 1.0,
    compress_mesh=False
):
    grid_mesher = GridMesher(
        dim=space_dim,
        left_bottom_corner=space_left_bottom_corner,
        right_top_corner=space_right_top_corner,
        resolution=space_resolution,
        boundaries=boundary_list,  # type: ignore
        inout_pipe_length=inout_pipe_length,
    )
    init_field_value = np.ones(shape=(grid_mesher.cell_num,))
    grid_mesher.extract_field(
        field_value=init_field_value,
        threshold=0.5,
        output_type='geo',
        geo_file_path=geo_file_path,
        min_size=mesh_min_size,
        max_size=max(mesh_max_size, mesh_min_size * 1.1),
        dist_min=mesh_min_size,
        dist_max=mesh_min_size * dist_max_ratio,
        msh_file_path=msh_file_path,
        compress_mesh_extend_from_boundary=compress_mesh
    )


def generate_top_pipes(num_pipes: int, start_marker: int, face_radius=0.1):
    face_norm_vector=np.array([0.0, 1.0])
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
        value=[0.0, 0.1],
        marker=1,
        face_center=np.array([right_boundary / 2.0, 0.0]),
        face_norm_vector=np.array([0.0, 1.0]),
        face_radius=0.1
    )
]

start_marker = 1
top_pipes, start_marker = generate_top_pipes(10, start_marker, face_radius=0.05)
left_pipes, start_marker = generate_left_pipes(10, start_marker, face_radius=0.05)
right_pipes, start_marker = generate_right_pipes(10, start_marker, face_radius=0.05)
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

prepare_design_space(
    space_dim=2,
    space_left_bottom_corner=np.array([0.0, 0.0]),
    space_right_top_corner=np.array([right_boundary, top_boundary]),
    space_resolution=0.025,
    boundary_list=boundaries,
    inout_pipe_length=0.05,
    geo_file_path=os.path.join(base_dir, 'model.geo'),
    msh_file_path=os.path.join(base_dir, 'model.msh'),
    mesh_min_size=0.02,
    mesh_max_size=0.02,
    dist_max_ratio=1.1,
    compress_mesh=True
)

boundary_dict = {}
for bry in boundaries:
    boundary_dict[bry.name] = bry

target_ratio = 0.2
target_volume = (right_boundary * top_boundary) * target_ratio

outlet_markers = []
for bry in boundaries:
    if bry.boundary_type == 'pressure':
        outlet_markers.append(bry.marker)
outlet_marker_ratios = np.ones(len(outlet_markers)) / len(outlet_markers)
outlet_marker_weights = np.ones(len(outlet_markers))
print('outlet_markers:', outlet_markers)
print('outlet_marker_ratios:', outlet_marker_ratios)
print('outlet_marker_weights:', outlet_marker_weights)

optim = TopoFluidShrinkOptimizer(
    name='t60_2',
    run_dir=base_dir,
    msh_file_name='model.msh',
    mesh_dimension=2,
    boundary_dict=boundary_dict,
    fluid_type='stoke',
    viscosity_init=1.0 / 400.,
    weight_filter_radius=0.075,
    obj_cost_args_list=[
        {
            'style': 'MinDissipation+BrinkmanPenalized',
            'with_viscosity': True,
            'dissipation_weight': 1.0,
            'penalized_weight': 1.0,
            'penalized_alpha_max': 1e4,
            'penalized_alpha_min': 1e-12,
        },
        {
            'style': 'FixedFlux',
            'inlet_markers': [1],
            'outlet_markers': outlet_markers,
            'outlet_marker_ratios': outlet_marker_ratios.tolist(),
            'outlet_marker_weights': (outlet_marker_weights * 1.0).tolist(),
            'with_rescale': True,
            'with_max_dif_cost': False,
            'max_dif_weight': 1.0,
        },
    ],
    reg_cost_args={
        'style': 'Volume',
        'weight': 1.0,
        'target_volume': target_volume,
        'reg_rescale': True,
    },
    
    min_lr=2e-2,
    max_lr=2e-2,
    # each_step_iteration=50,
    # optimize_iteration=300,
)

optim.run_optimize(
    # volume_stages=[1.93139056, 1.63511245, 1.33883434, 1.04255623, 0.74627811, 0.45],
    # step_iterations=[50, 50, 50, 50, 50, 50],
    volume_stages=[1.63511245, 1.04255623, 0.45, 0.25, 0.15],
    step_iterations=[100, 100, 100, 100, 100],
    fuzzy_alpha_boundary=-1.0, 
    filter_threshold=0.4
)
