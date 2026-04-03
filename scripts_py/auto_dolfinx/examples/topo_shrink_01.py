import os
import shutil

import numpy as np
import pyvista
from mpi4py import MPI

from scripts_py.auto_dolfinx.opt_solver.topo_fluid_weight import AutoFluidBoundary, TopoFluidOptimizer
from scripts_py.auto_dolfinx.opt_solver.grid_mesher import GridMesher
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.auto_dolfinx.opt_solver.seq_mesh import SequenceMeshObj

from thirdparty.pyadjoint.pyadjoint import Tape
from scripts_py.vtu_recorder import TensorBoardRecorder


base_dir = '/home/admin123456/Desktop/work/gmsh_test/t60_1'
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


def generate_top_pipes(num_pipes: int, start_marker: int, boundary_centers: list, face_radius=0.1):
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
    
    boundary_centers.append(face_centers)
    return bry_list, int(marker_series[-1])


def generate_left_pipes(num_pipes: int, start_marker: int, boundary_centers: list, face_radius=0.1):
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
    
    boundary_centers.append(face_centers)
    return bry_list, int(marker_series[-1])


def generate_right_pipes(num_pipes: int, start_marker: int, boundary_centers: list, face_radius=0.1):
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
    
    boundary_centers.append(face_centers)
    return bry_list, int(marker_series[-1])


boundary_centers = []

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
boundary_centers.append(np.array([boundaries[0].face_center]))

start_marker = 1
top_pipes, start_marker = generate_top_pipes(10, start_marker, boundary_centers, face_radius=0.05)
left_pipes, start_marker = generate_left_pipes(10, start_marker, boundary_centers, face_radius=0.05)
right_pipes, start_marker = generate_right_pipes(10, start_marker, boundary_centers, face_radius=0.05)
# top_pipes, start_marker = generate_top_pipes(1, start_marker, face_radius=0.1)
# left_pipes, start_marker = generate_left_pipes(1, start_marker, face_radius=0.1)
# right_pipes, start_marker = generate_right_pipes(1, start_marker, face_radius=0.1)
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
domain, cell_tags, facet_tags = MeshUtils.read_msh(os.path.join(base_dir, 'model.msh'), 2)
mesh = SequenceMeshObj(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)

outlet_markers = []
for bry in boundaries:
    if bry.boundary_type == 'pressure':
        outlet_markers.append(bry.marker)
outlet_marker_ratios = np.ones(len(outlet_markers)) / len(outlet_markers)
outlet_marker_weights = np.ones(len(outlet_markers))
print('outlet_markers:', outlet_markers)
print('outlet_marker_ratios:', outlet_marker_ratios)
print('outlet_marker_weights:', outlet_marker_weights)

boundary_centers_positions = np.concatenate(boundary_centers, axis=0)
# print('boundary_centers_positions:', boundary_centers_positions)

target_ratio = 0.2
target_volume = (right_boundary * top_boundary) * target_ratio

optim = TopoFluidOptimizer(
    name='t50_1',
    mesh=mesh,
    fluid_type='stoke',
    viscosity_init=1.0 / 400.,
    weight_filter_radius=0.075,
    result_dir=os.path.join(base_dir, 'dolfinx_topo'),
    alpha_fun={
        'name': 'sigmoid',
        'args': {
            'alpha_min': 1e-12,
            'alpha_max': 300.0,
            'x0': 0.5,
            'k': 10.0,
            'inverse': True,
        }
    },
    # alpha_fun={
    #     'name': 'heaviside',
    #     'args': {
    #         'alpha_min': 1e-12,
    #         'alpha_max': 300.0,
    #         'eta': 0.5,
    #         'beta': 20.0,
    #         'inverse': True,
    #     }
    # },
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
    heat_problem_args={
        'use_heat_problem': False,
        'weight': 0.0,
    },
)

recorder = TensorBoardRecorder(os.path.join(base_dir, 'log'))
result = optim.run_along_volume(
    min_lr=2e-2,
    max_lr=2e-2,
    clip_lr=False,
    max_iteration=300,
    recorder=recorder,
)

# result = optim.run_raw(
#     min_lr=2e-2,
#     max_lr=2e-2,
#     max_iteration=200,
#     recorder=recorder,
# )

