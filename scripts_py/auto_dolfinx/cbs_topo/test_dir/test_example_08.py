import os
import shutil
import time

import dolfinx

import numpy as np
from mpi4py import MPI

from scripts_py.mesh_utils import MeshObj, MeshUtils
from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import AutoFluidBoundary, AutoFluidCost, AutoFluidRegularization
from scripts_py.auto_dolfinx.cbs_topo.utils import generate_rectangle_mesh
from scripts_py.simulator.fipy_simulator import FipyStokeSimulator
from scripts_py.auto_dolfinx.cbs_topo.cbs_base import TopoFluidNode, CbsMeshObj


base_dir = '/home/admin123456/Desktop/work/gmsh_test/t41'
fluid_node_dir = os.path.join(base_dir, 'fluid_node_dir')
if os.path.exists(fluid_node_dir):
    shutil.rmtree(fluid_node_dir)
os.mkdir(fluid_node_dir)

msh_file = os.path.join(base_dir, 't30_model.msh')
xdmf_file = os.path.join(base_dir, 't30_model.xdmf')
MeshUtils.msh_to_XDMF(msh_file=msh_file, xdmf_file=xdmf_file, dim=2)

boundary_list = [
    AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0],
        marker=22,
        face_center=[-0.05, 0.2],
        face_norm_vector=[1.0, 0.0],
        face_radius=0.1
    ),
    AutoFluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=23
    ),
    AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=24
    )
]

mesh = CbsMeshObj(msh_file=msh_file, cell_mode=True, xdmf_file=xdmf_file, is_standard_mesh=False)
mesh.create_neighbors_info()

fluid_node = TopoFluidNode(
    mesh=mesh,
    name='fluid_node',
    save_dir=fluid_node_dir,
    node_idx=0,
    weight_filter_radius=0.02,
    topo_weight_init=1.0,
    
    # ------ simulator related ------
    viscosity=1.0 / 400.,
    
    # ------ computational related ------
    simulation_iterations=200,
    convergence_threshold=1e-3,
    velocityRelaxation=0.6,
    pressureRelaxation=0.8,
    log_step=2,
    hot_steps=10,
    parallel_num=1,
    use_tqdm=True,
    use_log=True,
)

for boundary in boundary_list:
    if boundary.boundary_type == 'velocity':
        fluid_node.inlet_brys[boundary.name] = boundary
    elif boundary.boundary_type == 'pressure':
        fluid_node.outlet_brys[boundary.name] = boundary
    elif boundary.boundary_type == 'wall':
        fluid_node.wall_brys[boundary.name] = boundary

fluid_node.object_cost = AutoFluidCost(
    name='obj_cost',
    style='MinDissipation',
    weight=1.0,
)
fluid_node.regularization_cost = AutoFluidRegularization(
    name='reg_cost',
    style='Volume',
    weight=1.0,
    target_ratio=0.1,
)

start_time = time.time()
fluid_node.solve()
end_time = time.time()
print(f'Time taken: {end_time - start_time} seconds')
# fluid_node.save()
