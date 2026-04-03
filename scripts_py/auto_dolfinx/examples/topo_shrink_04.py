import os
import shutil

import numpy as np
import pyvista
from mpi4py import MPI
import copy

from scripts_py.auto_dolfinx.opt_solver.topo_fluid_weight import AutoFluidBoundary
from scripts_py.auto_dolfinx.opt_solver.topo_fluid_weight_fipy import TopoFluidWeightOptimizerFipy
from scripts_py.auto_dolfinx.opt_solver.grid_mesher import GridMesher
from scripts_py.mesh_utils import MeshUtils, MeshObj
# from scripts_py.auto_dolfinx.opt_solver.seq_mesh import SequenceMeshObj

from thirdparty.pyadjoint.pyadjoint import Tape
from scripts_py.vtu_recorder import TensorBoardRecorder

base_dir = '/home/admin123456/Desktop/work/gmsh_test/t60_3'

right_boundary = 1.0
back_boundary = 1.0
top_boundary = 1.0


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
    new_boundary_list = []
    for boundary in boundary_list:
        boundary_copy = copy.deepcopy(boundary)
        boundary_copy.origin_face_center = boundary_copy.face_center  # type: ignore
        if boundary_copy.boundary_type == 'velocity':
            boundary_copy.face_center = boundary_copy.face_center - boundary_copy.face_norm_vector * inout_pipe_length
        elif boundary_copy.boundary_type == 'pressure':
            boundary_copy.face_center = boundary_copy.face_center + boundary_copy.face_norm_vector * inout_pipe_length
        new_boundary_list.append(boundary_copy)
                
    grid_mesher = GridMesher(
        dim=space_dim,
        left_bottom_corner=space_left_bottom_corner,
        right_top_corner=space_right_top_corner,
        resolution=space_resolution,
        boundaries=new_boundary_list,  # type: ignore
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
    
    return new_boundary_list


def generate_top_pipes(num_pipes: int, start_marker: int, face_radius=0.05, min_ratio=0.3, max_ratio=0.7):
    face_norm_vector=np.array([0.0, 0.0, 1.0])
    if num_pipes == 1:
        face_centers = np.array([[right_boundary / 2.0, back_boundary / 2.0, top_boundary]])
    else:
        row_num = max(int(np.sqrt(num_pipes)), 1)
        x_centers = np.linspace(right_boundary * min_ratio, right_boundary * max_ratio, row_num)
        y_centers = np.linspace(back_boundary * min_ratio, back_boundary * max_ratio, row_num)
        x_centers, y_centers = np.meshgrid(x_centers, y_centers)        
        face_centers = np.concatenate([
            x_centers[..., np.newaxis], 
            y_centers[..., np.newaxis],
            np.ones(shape=(x_centers.shape[0], x_centers.shape[1], 1)) * top_boundary
        ], axis=-1).reshape(-1, 3)
        
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


def generate_front_pipes(num_pipes: int, start_marker: int, face_radius=0.05, min_ratio=0.3, max_ratio=0.7, min_top_ratio=0.3):
    face_norm_vector=np.array([0.0, -1.0, 0.0])
    if num_pipes == 1:
        face_centers = np.array([[right_boundary / 2.0, 0.0, top_boundary / 2.0]])
    else:
        row_num = max(int(np.sqrt(num_pipes)), 1)
        x_centers = np.linspace(right_boundary * min_ratio, right_boundary * max_ratio, row_num)
        z_centers = np.linspace(top_boundary * min_top_ratio, top_boundary * max_ratio, row_num)
        x_centers, z_centers = np.meshgrid(x_centers, z_centers)
        face_centers = np.concatenate([
            x_centers[..., np.newaxis],
            np.zeros(shape=(x_centers.shape[0], x_centers.shape[1], 1)),
            z_centers[..., np.newaxis],
        ], axis=-1).reshape(-1, 3)
    
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


def generate_back_pipes(num_pipes: int, start_marker: int, face_radius=0.05, min_ratio=0.3, max_ratio=0.7, min_top_ratio=0.3):
    face_norm_vector=np.array([0.0, 1.0, 0.0])
    if num_pipes == 1:
        face_centers = np.array([[right_boundary / 2.0, back_boundary, top_boundary / 2.0]])
    else:
        row_num = max(int(np.sqrt(num_pipes)), 1)
        x_centers = np.linspace(right_boundary * min_ratio, right_boundary * max_ratio, row_num)
        z_centers = np.linspace(top_boundary * min_top_ratio, top_boundary * max_ratio, row_num)
        x_centers, z_centers = np.meshgrid(x_centers, z_centers)
        face_centers = np.concatenate([
            x_centers[..., np.newaxis],
            np.ones(shape=(x_centers.shape[0], x_centers.shape[1], 1)) * back_boundary,
            z_centers[..., np.newaxis],
        ], axis=-1).reshape(-1, 3)
    
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


def generate_left_pipes(num_pipes: int, start_marker: int, face_radius=0.05, min_ratio=0.3, max_ratio=0.7, min_top_ratio=0.3):
    face_norm_vector=np.array([-1.0, 0.0, 0.0])
    if num_pipes == 1:
        face_centers = np.array([[0.0, back_boundary / 2.0, top_boundary / 2.0]])
    else:
        row_num = max(int(np.sqrt(num_pipes)), 1)
        y_centers = np.linspace(back_boundary * min_ratio, back_boundary * max_ratio, row_num)
        z_centers = np.linspace(top_boundary * min_top_ratio, top_boundary * max_ratio, row_num)
        y_centers, z_centers = np.meshgrid(y_centers, z_centers)
        face_centers = np.concatenate([
            np.zeros(shape=(y_centers.shape[0], y_centers.shape[1], 1)),
            y_centers[..., np.newaxis],
            z_centers[..., np.newaxis],
        ], axis=-1).reshape(-1, 3)
    
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


def generate_right_pipes(num_pipes: int, start_marker: int, face_radius=0.05, min_ratio=0.3, max_ratio=0.7, min_top_ratio=0.3):
    face_norm_vector=np.array([1.0, 0.0, 0.0])
    if num_pipes == 1:
        face_centers = np.array([[right_boundary, back_boundary / 2.0, top_boundary / 2.0]])
    else:
        row_num = max(int(np.sqrt(num_pipes)), 1)
        y_centers = np.linspace(back_boundary * min_ratio, back_boundary * max_ratio, row_num)
        z_centers = np.linspace(top_boundary * min_top_ratio, top_boundary * max_ratio, row_num)
        y_centers, z_centers = np.meshgrid(y_centers, z_centers)
        face_centers = np.concatenate([
            np.ones(shape=(y_centers.shape[0], y_centers.shape[1], 1)) * right_boundary,
            y_centers[..., np.newaxis],
            z_centers[..., np.newaxis],
        ], axis=-1).reshape(-1, 3)
    
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

min_ratio = 0.2
max_ratio = 0.8
face_radius = 0.06
top_pipes, start_marker = generate_top_pipes(9, start_marker, face_radius, min_ratio, max_ratio)
front_pipes, start_marker = generate_front_pipes(9, start_marker, face_radius, min_ratio, max_ratio, 0.4)
back_pipes, start_marker = generate_back_pipes(9, start_marker, face_radius, min_ratio, max_ratio, 0.4)
left_pipes, start_marker = generate_left_pipes(9, start_marker, face_radius, min_ratio, max_ratio, 0.4)
right_pipes, start_marker = generate_right_pipes(9, start_marker, face_radius, min_ratio, max_ratio, 0.4)

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

boundaries = prepare_design_space(
    space_dim=3,
    space_left_bottom_corner=np.array([0.0, 0.0, 0.0]),
    space_right_top_corner=np.array([right_boundary, back_boundary, top_boundary]),
    space_resolution=0.1,
    boundary_list=boundaries,
    inout_pipe_length=0.05,
    geo_file_path=os.path.join(base_dir, 'model.geo'),
    msh_file_path=os.path.join(base_dir, 'model.msh'),
    mesh_min_size=0.02,
    mesh_max_size=0.03,
    dist_max_ratio=1.1,
    compress_mesh=True
)
# for bry in boundaries:
#     print(bry.name, bry.face_center)

domain, cell_tags, facet_tags = MeshUtils.read_msh(os.path.join(base_dir, 'model.msh'), 3)
mesh = MeshObj(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)

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

optim = TopoFluidWeightOptimizerFipy(
    name='t60_3',
    msh_file=os.path.join(base_dir, 'model.msh'),
    mesh=mesh,
    fluid_type='stoke',
    viscosity_init=1.0 / 400.,
    weight_filter_radius=0.075,
    result_dir=os.path.join(base_dir, 'fipy_topo'),
    alpha_fun={
        'name': 'sigmoid',
        'args': {
            'alpha_min': 1e-12,
            'alpha_max': 300.0,
            'x0': 0.5,
            'k': 10.0,
            'inverse': True,
        }
    }
)

optim.setup_tape(Tape())
for boundary in boundaries:
    optim.add_boundary(boundary)
    
optim.define_fluid_problem(
    obj_cost_args_list=[
        {
            'style': 'MinDissipation+BrinkmanPenalized',
            'with_viscosity': True,
            'dissipation_weight': 0.0,  # 多联通时这里设置为0
            'penalized_weight': 1.0,
            'penalized_alpha_max': 1e5,
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
    simulation_iterations=200,
    convergence_threshold=1e-4,
    velocityRelaxation=0.6,
    pressureRelaxation=0.8,
    log_step=2,
    hot_steps=10,
    parallel_num=8,
    use_tqdm=True,
    use_log=True,
)

recorder = TensorBoardRecorder(os.path.join(base_dir, 'log'))
result = optim.run_along_volume(
    min_lr=2e-2,
    max_lr=2e-2,
    clip_lr=False,
    max_iteration=500,
    recorder=recorder,
)


