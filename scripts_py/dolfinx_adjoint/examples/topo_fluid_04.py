import dolfinx
import ufl
from mpi4py import MPI

import numpy as np

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.simulator.dolfin_simulator import FluidBoundary
from scripts_py.dolfinx_adjoint.topology_utils.topology_fluid import (
    FluidTopologyOptimizer,
    FluidCost,
    WeightRegularizationItem,
)

mesh_file = '/home/admin123456/Desktop/work/gmsh_test/t4/topo_fluid_04.msh'
xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t4/topo_fluid_04.xdmf'
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
    target_ratio=0.25
)

optim = FluidTopologyOptimizer(
    name='topo_fluid_04',
    mesh=mesh,
    fluid_type='stoke',
    cost_item=cost_item,
    regularization_item=regularization_item,
    topo_weight_init=0.5,
    nut_init=1.0 / 100.,
    weight_filter_radius=0.2,
    weight_filter_is_cell_field=True,
    result_dir='/home/admin123456/Desktop/work/gmsh_test/t4/cache'
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

# --- 5. run optimization ---
optim.run(max_iteration=100)