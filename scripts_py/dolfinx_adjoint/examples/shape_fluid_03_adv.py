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


# --- 1. define mesh ---
xdmf_file = "/home/admin123456/Desktop/work/gmsh_test/t3/optimal_pipe_03.xdmf"
mesh = MeshObj(xdmf_file=xdmf_file, mesh_name="model")

# --- 2. define cost and regularization
cost_item = FluidCost(
    name='cost_1',
    style='MinDissipation',
    weight=1.0,
)
regularization_item = ShapeRegularizationItem(
    name='regularization_1',
    style='Volume',
    weight=10.0,
    target_ratio=1.0
)
# regularization_item = ShapeRegularizationItem(
#     name='regularization_1',
#     style='Volume',
#     weight=0.5,
#     target_ratio=0.8
# )

optim = FluidShapeOptimizer(
    name='shape_fluid_03',
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
    result_dir='/home/admin123456/Desktop/work/gmsh_test/t3/cache',
    scalar_product_kwargs={"method": "default"}
)

# --- 3. define boundary ---
inlet_bry = FluidPipeBoundary(
    name='inlet',
    boundary_type='velocity',
    value=[6.0, 0.],
    marker=14,
    is_free=False,
    face_center=[0.0, 0.5],
    face_norm_vector=[1.0, 0.0],
    face_radius=0.5
)
outlet_bry = FluidPipeBoundary(
    name='outlet',
    boundary_type='pressure',
    value=0.0,
    marker=15,
    is_free=False
)
wall_fix_bry = FluidPipeBoundary(
    name='wall_fix_bry',
    boundary_type='wall',
    value=0.0,
    marker=16,
    is_free=False
)
wall_free_bry = FluidPipeBoundary(
    name='wall_free_bry',
    boundary_type='wall',
    value=0.0,
    marker=17,
    is_free=True
)

optim.add_boundary(inlet_bry)
optim.add_boundary(outlet_bry)
optim.add_boundary(wall_fix_bry)
optim.add_boundary(wall_free_bry)

# --- 4. define problem ---
optim.define_problem()

# --- 5. run optimization ---
optim.run(max_iteration=600)