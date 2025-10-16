'''
拓扑优化受到先验topology weight影响,当预设好路径(guess path)时,拓扑优化不会产生新的路径,只会在原设路径上进行优化.
当不存在预设好路径时,即topo_weight初始值均为0.5时,尽管存在障碍物因素,拓扑优化依然会自动产生新的路径.

一个想法是多次交替expand和shrink:
    1.在expand阶段, guess path附近topology weight设置为1.0
    2.在shrink阶段, 进行拓扑优化, 并只保留高权重区域.
    3.重复上述过程, 直到收敛.
'''

import dolfinx
import ufl

import numpy as np
from sklearn.neighbors import KDTree

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.simulator.dolfin_simulator import FluidBoundary
from scripts_py.dolfinx_adjoint.topology_utils.topology_fluid import (
    ConstrainedTopologyOptimizer,
    FluidCost,
    WeightRegularizationItem,
)
from scripts_py.dolfinx_utils import DolfinUtils

mesh_file = '/home/admin123456/Desktop/work/gmsh_test/t7/topo_fluid_04.msh'
xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t7/topo_fluid_04.xdmf'
# MeshUtils.msh_to_XDMF(mesh_file, xdmf_file, dim=2)

# --- 1. define mesh ---
mesh = MeshObj(xdmf_file=xdmf_file)

# --- 2. define cost and regularization ---
cost_item = FluidCost(
    name='cost_1',
    style='MinDissipation',
    weight=1.0,
)
regularization_item = WeightRegularizationItem(
    name='weight_1',
    style='Volume',
    weight=10.0,
    target_ratio=0.35
)

optim = ConstrainedTopologyOptimizer(
    name='topo_fluid_04',
    mesh=mesh,
    fluid_type='stoke',
    cost_item=cost_item,
    regularization_item=regularization_item,
    topo_weight_init=0.5,
    nut_init=1.0 / 100.,
    weight_filter_radius=0.2,
    weight_filter_is_cell_field=True,
    result_dir='/home/admin123456/Desktop/work/gmsh_test/t7/cache',
    obstacle_cost_weight=100.0,
)

# --- 3. define boundary ---
inlet_bry = FluidBoundary(
    name='inlet',
    boundary_type='velocity',
    value=[1.0, 0.],
    marker=18,
    face_center=[0.0, 0.5],
    face_norm_vector=[1.0, 0.0],
    face_radius=0.5,
)
outlet_bry = FluidBoundary(
    name='outlet',
    boundary_type='pressure',
    value=0.0,
    marker=19
)
wall_bry = FluidBoundary(
    name='wall',
    boundary_type='wall',
    value=0.0,
    marker=20
)

optim.add_boundary(inlet_bry)
optim.add_boundary(outlet_bry)
optim.add_boundary(wall_bry)

# --- 4. define problem ---
optim.define_problem()

# --- 5. define obstacle ---
cell_center = DolfinUtils.tabulate_dof_coordinates(optim.cell_scalar_func_space)

obstacle_center = np.array([3.5, 2.0, 0.0])
cell2obs_dist = np.linalg.norm(cell_center - obstacle_center, axis=1)
obs_dofs = np.where(cell2obs_dist < 0.25)[0]
optim.set_obstacle_source(obs_dofs)

# --- 6. define init topo weight ---
init_path = np.array([
    [0, 0.5, 0],
    [4.5, 0.5, 0],
    [4.5, 3.5, 0],
    [7, 3.5, 0]
])
detail_path = []
for i in range(init_path.shape[0] - 1):
    step = int(np.linalg.norm(init_path[i + 1] - init_path[i]) / 0.1)
    detail_path.append(np.linspace(init_path[i], init_path[i + 1], step))
detail_path = np.concatenate(detail_path, axis=0)

tree = KDTree(cell_center)
inner_idxs_list = tree.query_radius(detail_path, r=0.25, return_distance=False)
path_cell_idxs = np.unique(np.concatenate(inner_idxs_list, axis=-1))

# optim.topo_weight_init = 0.3
# optim.set_topo_weight(path_cell_idxs, 0.7)

# --- 7. test state solution at first step ---
# optim.optimizer.compute_state(
#     comm=optim.mesh.domain.comm,
#     mat_assemble_method="lift",
#     with_residual=True,
# )
# optim.record_Upw(0)

# --- 8. run optimization ---
optim.run(max_iteration=150)
