import os
import shutil

import numpy as np
import pyvista
from mpi4py import MPI

from scripts_py.auto_dolfinx.opt_solver.topo_fluid_hierarchy import AutoFluidBoundary
from scripts_py.auto_dolfinx.opt_solver.grid_mesher import GridMesher
from scripts_py.auto_dolfinx.opt_solver.topo_fluid_v2 import TopoFluidOptimizer
from scripts_py.mesh_utils import MeshUtils, MeshObj

from thirdparty.pyadjoint.pyadjoint import Tape
from scripts_py.vtu_recorder import TensorBoardRecorder



"""
TODO: 这里的实验说明渗透率的问题,由于alpha_max的限制为300.0,当目标通量很小时,就会造成优化失效
"""


base_dir = '/home/admin123456/Desktop/work/gmsh_test/t50_5'
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
domain, cell_tags, facet_tags = MeshUtils.read_msh(os.path.join(base_dir, 'model.msh'), 2)
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

optim = TopoFluidOptimizer(
    name='t50_1',
    mesh=mesh,
    fluid_type='stoke',
    viscosity_init=1.0 / 400.,
    topo_weight_init=0.9,  # TODO: 这究竟有没影响
    init_topo_weight_method='raw',
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
    #     'name': 'exponential',
    #     'args': {
    #         'alpha_min': 1e-12,
    #         'alpha_max': 300.0,
    #         'p': 3.0,
    #     }
    # },
)
optim.setup_tape(Tape())
for boundary in boundaries:
    optim.add_boundary(boundary)
        
optim.define_problem(
    obj_cost_args_list=[
        {
            'style': 'MinDissipation+BrinkmanPenalized',
            # 'style': 'MinDissipation+GradBrinkmanPenalized',
            'with_viscosity': True,
            'dissipation_weight': 1.0,  # 这个才是控制联通域的关键, 因为只要流体方程可求解, 那么流量进出就是必然的
            'penalized_weight': 0.0,
            'penalized_alpha_max': 1e3,
            'penalized_alpha_min': 1e-12,
        },
        {
            'style': 'FixedFlux',
            'inlet_markers': [1],
            'outlet_markers': outlet_markers,
            'outlet_marker_ratios': outlet_marker_ratios.tolist(),
            'outlet_marker_weights': (outlet_marker_weights * 1e2).tolist(),
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
    obstacle_cost_args=None,
)
recorder = TensorBoardRecorder(os.path.join(base_dir, 'dolfinx_topo', 'log'))
result = optim.run_with_lr_scheduler(
    min_lr=2e-2,
    max_lr=2e-2,
    max_iteration=500,
    hot_step=30,
    momentum_weight=0.8,
    full_run=False,
    recorder=recorder,
)
