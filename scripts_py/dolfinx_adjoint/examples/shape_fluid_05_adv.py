import dolfinx
import ufl
from mpi4py import MPI

import numpy as np

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_adjoint.shape_fluid_utils.shape_fluid import (
    FluidShapeOptimizer,
    FluidPipeBoundary,
    FluidCost,
    ShapeRegularizationItem,
)


msh_file = "/home/admin123456/Desktop/work/gmsh_test/t5/shape_fluid_05.msh"
xdmf_file = "/home/admin123456/Desktop/work/gmsh_test/t5/shape_fluid_05.xdmf"
# MeshUtils.msh_to_XDMF(name='model', msh_file=msh_file, xdmf_file=xdmf_file, dim=2)

# --- 1. define mesh ---
mesh = MeshObj(xdmf_file=xdmf_file, mesh_name="model")

# --- 2. define cost and regularization ---
cost_item = FluidCost(
    name='cost_1',
    style='MinDissipation',
    weight=1.0,
)
regularization_item = ShapeRegularizationItem(
    name='regularization_1',
    style='Volume',
    weight=1.0,
    target_ratio=0.5
)

optim = FluidShapeOptimizer(
    name='shape_fluid_05',
    mesh=mesh,
    fluid_type='stoke',
    nut_init=1.0 / 400.0,
    cost_item=cost_item,
    regularization_item=regularization_item,
    with_vol_check=True,
    vol_change=0.25,
    with_angle_check=True,
    angle_tol_max=160.0,
    angle_tol_min=20.0,
    result_dir='/home/admin123456/Desktop/work/gmsh_test/t5/cache',
    scalar_product_kwargs={"method": "default"}
)

# --- 3. define boundary ---
inlet_bry = FluidPipeBoundary(
    name='inlet',
    boundary_type='velocity',
    value=[1.0, 0.],
    marker=35,
    is_free=False,
    face_center=[0.0, 0.25],
    face_norm_vector=[1.0, 0.0],
    face_radius=0.25
)
outlet_bry = FluidPipeBoundary(
    name='outlet',
    boundary_type='pressure',
    value=0.0,
    marker=36,
    is_free=False
)
wall_bry = FluidPipeBoundary(
    name='wall',
    boundary_type='wall',
    value=0.0,
    marker=37,
    is_free=True
)

optim.add_boundary(inlet_bry)
optim.add_boundary(outlet_bry)
optim.add_boundary(wall_bry)

# --- 4. define problem ---
optim.define_problem()

# --- 5. run optimization ---
optim.run(max_iteration=100)
