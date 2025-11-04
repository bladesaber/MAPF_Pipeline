"""
reference: https://github.com/sblauth/cashocs/tree/main/demos/documented/shape_optimization/shape_poisson
"""

import dolfinx
import ufl
from mpi4py import MPI

import numpy as np

from scripts_py.dolfinx_adjoint.base_type import (
    StateProblemDataBase,
    ShapeProblemDataBase,
    BoundaryCell,
)
from scripts_py.dolfinx_adjoint.problem_state import StateProblem
from scripts_py.dolfinx_adjoint.cost_functions import IntegralFunctional
from scripts_py.dolfinx_adjoint.solver_optimize import ShapeOptimizer
from scripts_py.dolfinx_adjoint.regularization_shape import ShapeRegularizationPipe
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils
from scripts_py.vtu_recorder import VtuRecorder


# --- 1. define mesh ---
mesh = MeshObj(domain=dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 25, 25))

# --- 2.1 define function space, state, f_form ---
point_scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
u1 = dolfinx.fem.Function(point_scalar_func_space, name="State_1")
v1 = dolfinx.fem.Function(point_scalar_func_space, name="adjoint_1")

coord = MeshUtils.define_coordinate(mesh.domain)  # This is global coordinate
source_exp = 2.5 * np.power(coord[0] + 0.4 - np.power(coord[1], 2), 2) + np.power(coord[0], 2) + np.power(coord[1], 2) - 1  # type: ignore
F_form = ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx - ufl.inner(source_exp, v1) * ufl.dx

state_database_1 = StateProblemDataBase(
    name="state_problem_1",
    F_form=F_form,
    state=u1,
    adjoint=v1,
    is_linear=True,
    state_ksp_option={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
    },
    adjoint_ksp_option={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
    },
)

# --- 2.2 define boundary condition ---
bc_cell = BoundaryCell.define_bry_cell(
    name="bc1",
    bc_func_space=point_scalar_func_space,
    value_func_space=point_scalar_func_space,
    dofs=MeshUtils.extract_entity_dofs(
        point_scalar_func_space,
        mesh.fdim,
        MeshUtils.extract_entity_indices(mesh.domain, style="facet", marker=None),
    ),
    bc_value=0.0,
    is_mixed_element=False,
)
state_database_1.add_bc(bc_cell)

# --- 3.1 define control problem ---
control_database_1 = ShapeProblemDataBase(
    domain=mesh.domain,
    gradient_ksp_option={
        'ksp_type': 'preonly', 
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps", 
        "mat_mumps_icntl_24": 1
    }
)

# --- 3.2 define cost function ---
cost1_fun = IntegralFunctional(
    name='cost1', 
    domain=mesh.domain, 
    cost_form=u1 * ufl.dx, 
    weight=1.0
)

# --- 4. define optimal problem ---
opt_problem = ShapeOptimizer(
    state_problem=StateProblem([state_database_1]),
    shape_database=control_database_1,
    cost_functional_list=[cost1_fun],
    shape_regularization_list=ShapeRegularizationPipe([]),
    scalar_product_kwargs={"method": "default"}
)

# --- 5. Test State Problem ---
# opt_problem.compute_state(
#     comm=mesh.domain.comm,
#     mat_assemble_method="lift",
#     with_residual=True,
# )

recorder = VtuRecorder(mesh, save_dir='/home/admin123456/Desktop/work/gmsh_test/test/t2')
recorder.update_point_data('u1', u1.x.array)  # type: ignore
recorder.save(0)

loss_current = opt_problem.evaluate_cost_functional(mesh.domain.comm, update_state=True)
print(f"step/{0} loss:{loss_current}")

line_search_step = [0.5, 0.2, 0.1, 0.05, 0.01]
is_solved, step = False, 0
while True:
    step += 1
    if step > 1:
        loss = opt_problem.evaluate_cost_functional(mesh.domain.comm, update_state=True)
        loss_residual = loss_current - loss
        if np.abs(loss_residual) < 1e-6:
            print(f"step/{step} loss:{loss_current} residual:{loss_residual:.8f}.")
            is_solved = True
            break
        else:
            loss_current = loss
            print(f"step/{step} loss:{loss_current} residual:{loss_residual:.8f}.")

    report_dict = opt_problem.compute_gradient(
        mesh.domain.comm,
        state_kwargs={
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        adjoint_kwargs={
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        gradient_kwargs={
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        scalar_product_kwargs={
            'with_residual': False
        },
        update_state=True,
        update_adjoint=True
    )
    is_success = report_dict['is_success']
    if not is_success:
        print(f'[Error]: {report_dict["log"]}')
        is_solved = False
        break

    dire_np: np.ndarray = report_dict['shape_grad'].x.array.copy()
    dire_np = dire_np.reshape((-1, mesh.tdim))    
    dire_np = dire_np / np.linalg.norm(dire_np, ord=2)
    
    find_best_step = False
    for step_length in line_search_step:
        displacement_np = dire_np * step_length * -1.0
        mesh.deform(displacement_np)
        loss_new = opt_problem.evaluate_cost_functional(mesh.domain.comm, update_state=True)
        if loss_new < loss_current:
            find_best_step = True
            recorder.update_point_data('u1', u1.x.array)  # type: ignore
            recorder.save(step)
            break
        else:
            mesh.deform(displacement_np * -1.0)  # recover mesh
        
    if not find_best_step:
        print(f"[Warning]: line search fail.")
        is_solved = False
        break

