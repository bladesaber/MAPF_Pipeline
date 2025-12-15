import os
import shutil

import dolfinx

import numpy as np
from mpi4py import MPI

from scripts_py.mesh_utils import MeshObj, MeshUtils
from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import AutoFluidBoundary, AutoFluidCost, AutoFluidRegularization
from scripts_py.auto_dolfinx.cbs_topo.utils import generate_box_mesh
from scripts_py.simulator.fipy_simulator import FipyStokeSimpleSimulator
from scripts_py.auto_dolfinx.cbs_topo.cbs_base import TopoFluidNode, CbsMeshObj


boundary_list = [
    AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0, 0.0],
        marker=1,
        face_center=np.array([0.0, 0.2, 0.15]),
        face_norm_vector=np.array([1.0, 0.0, 0.0]),
        face_radius=0.1,
    ),
    AutoFluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=2,
        face_center=np.array([1.0, 0.8, 0.15]),
        face_norm_vector=np.array([1.0, 0.0, 0.0]),
        face_radius=0.1,
    ),
    AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=3,
    ),
]

base_dir = '/home/admin123456/Desktop/work/gmsh_test/t37'
xdmf_file = os.path.join(base_dir, 'model.xdmf')
msh_v2_file = os.path.join(base_dir, 'model_v2.msh')
msh_file = os.path.join(base_dir, 'model.msh')

mesh = generate_box_mesh(
    boundary_list=boundary_list,
    left_bottom_corner=np.array([0.0, 0.0, 0.0]),
    right_top_corner=np.array([1.0, 1.0, 0.3]),
    num_cells=[30, 30, 10],
    xdmf_file=xdmf_file,
    cell_type=dolfinx.mesh.CellType.tetrahedron,
    comm=MPI.COMM_WORLD,
)

MeshUtils.generate_msh22_file(
    mesh=mesh,
    filename=msh_v2_file,
    face_tags={
        'inlet': 1,
        'outlet': 2,
        'wall': 3,
    },
    cell_tags={
        'volume': 9,
    },
)

FipyStokeSimpleSimulator.msh_version_convert(
    source_msh_file=msh_v2_file,
    target_msh_file=msh_file,
    target_version='msh4',
)

# # ------ debug simulate ------
# simulator = FipyStokeSimpleSimulator.from_msh(
#     source_msh_file=msh_file,
#     dim=3,
#     viscosity=1.0 / 400.0,
#     velocityRelaxation=0.5,
#     pressureRelaxation=0.8,
# )

# for boundary in boundary_list:
#     simulator.add_boundary(boundary)
# simulator.define_boundary()

# simulator.run(iterations=10000, log_step=2, save_dir=os.path.join(base_dir, 'cache'))
# # ------ debug simulate ------

fluid_node_dir = os.path.join(base_dir, 'fluid_node_dir')
if os.path.exists(fluid_node_dir):
    shutil.rmtree(fluid_node_dir)
os.mkdir(fluid_node_dir)

mesh = CbsMeshObj(
    msh_file=msh_file,
    cell_mode=True,
    xdmf_file=xdmf_file,
)
fluid_node = TopoFluidNode(
    mesh=mesh,
    name='fluid_node',
    save_dir=fluid_node_dir,
    node_idx=0,
    weight_filter_radius=0.04,
    topo_weight_init=1.0,
    simulation_iterations=400,
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
    weight=10.0,
    target_ratio=0.1,
)
fluid_node.solve()
# fluid_node.save()
