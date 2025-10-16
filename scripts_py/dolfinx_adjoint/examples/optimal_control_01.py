"""
reference: https://github.com/sblauth/cashocs/tree/main/demos/documented/optimal_control/poisson
"""

import dolfinx
import ufl
from mpi4py import MPI

import numpy as np

from scripts_py.dolfinx_adjoint.base_type import (
    StateProblemDataBase,
    ControlProblemDataBase,
    BoundaryCell,
)
from scripts_py.dolfinx_adjoint.problem_state import StateProblem
from scripts_py.dolfinx_adjoint.solver_optimize import ControlOptimizer
from scripts_py.dolfinx_adjoint.cost_functions import IntegralFunctional
from scripts_py.dolfinx_adjoint.regularization_control import ControlRegularizationPipe
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils

# from scripts_py.dolfinx_adjoint.recorder_utils import XDMFRecorder
# from scripts_py.dolfinx_adjoint.vis_mesh_utils import VisUtils

# --- 1. define mesh ---
mesh = MeshObj(domain=dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 25, 25))

# --- 2.1 define function space, state, f_form ---
point_scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
u1 = dolfinx.fem.Function(point_scalar_func_space, name="State_1")
v1 = dolfinx.fem.Function(point_scalar_func_space, name="adjoint_1")
f1 = dolfinx.fem.Function(point_scalar_func_space, name="control_1")

F1_form = ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx - ufl.inner(f1, v1) * ufl.dx
state_database_1 = StateProblemDataBase(
    name="state_problem_1",
    F_form=F1_form,
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
control_database_1 = ControlProblemDataBase(
    controls=[f1],  # type: ignore
    gradient_ksp_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
    },
)

# --- 3.2 define cost function ---
source_f = dolfinx.fem.Function(point_scalar_func_space)
source_f.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))  # type: ignore
cost1_fun = IntegralFunctional(
    name="cost1_fun",
    domain=mesh.domain,
    cost_form=0.5 * ufl.inner(u1 - source_f, u1 - source_f) * ufl.dx,  # type: ignore
    weight=1.0,
)

# --- 4. define optimal problem ---
opt_problem = ControlOptimizer(
    state_problem=StateProblem([state_database_1]),
    control_database=control_database_1,
    cost_functional_list=[cost1_fun],
    control_regularization_pipe=ControlRegularizationPipe([]),
    scalar_product_kwargs={},
)

# --- 5. Test State Problem ---
# opt_problem.compute_state(
#     comm=mesh.domain.comm,
#     mat_assemble_method="lift",
#     with_residual=True,
# )
# print(u1.x.array)

loss_current = opt_problem.evaluate_cost_functional(
    mesh.domain.comm, 
    update_state=True,
    with_residual=False,
)
print(f"step/{0} loss:{loss_current}")

line_search_step = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.01]
is_solved, step = False, 0
while True:
    step += 1
    if step > 1:
        loss = opt_problem.evaluate_cost_functional(mesh.domain.comm, update_state=True)
        loss_ratio = loss / loss_current
        if np.abs(loss_ratio - 1.0) < 1e-2:
            print(f"step/{step} loss:{loss_current} ratio:{loss_ratio:.8f}.")
            is_solved = True
            break
        else:
            loss_current = loss
            print(f"step/{step} loss:{loss_current} ratio:{loss_ratio:.8f}.")

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
        update_state=False,
        update_adjoint=True
    )
    
    is_success = report_dict['is_success']
    if not is_success:
        print(f'[Error]: {report_dict["log"]}')
        is_solved = False
        break

    grads_dict = report_dict['gradients']
    for name, control_par in opt_problem.control_database.control_pars.items():
        grad = grads_dict[name]

        grad_np: np.array = grad.x.array.copy()  # type: ignore
        grad_np = grad_np / np.linalg.norm(grad_np, ord=2)

        find_best_step = False
        control_origin: np.array = control_par.control.x.array.copy()  # type: ignore
        for step_length in line_search_step:
            control_par.control.x.array[:] = control_origin - grad_np * step_length
            loss_new = opt_problem.evaluate_cost_functional(mesh.domain.comm, update_state=True)            
            if loss_new < loss_current:
                find_best_step = True
                break
        
        if not find_best_step:
            print(f"[Warning]: {name} line search fail.")
            is_solved = False
            break

if is_solved:
    mesh.grid.point_data['control'] = f1.x.array  # type: ignore
    mesh.grid.plot()
