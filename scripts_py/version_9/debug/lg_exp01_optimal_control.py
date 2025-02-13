"""
https://github.com/sblauth/cashocs/tree/main/demos/documented/optimal_control/poisson
"""

import numpy as np
import ufl
from mpi4py import MPI
import dolfinx
from typing import Dict

from scripts_py.version_9.dolfinx_grad.lagrange_method.type_database import create_control_problem, create_state_problem
from scripts_py.version_9.dolfinx_grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.solver_optimize import OptimalControlProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.cost_functions import IntegralFunctional
from scripts_py.version_9.dolfinx_grad.lagrange_method.control_regularization import ControlRegularizationPipe
from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils, MeshObj


mesh = MeshObj(domain=dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 25, 25))
V = dolfinx.fem.FunctionSpace(mesh.domain, ("Lagrange", 1))
state_problems = []
cost_func_list = []

# ------ Define State Problem 1
u1 = dolfinx.fem.Function(V, name='State_1')
v1 = dolfinx.fem.Function(V, name='adjoint_1')
f1 = dolfinx.fem.Function(V, name='control_1')

F1_form = ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx - f1 * v1 * ufl.dx

bc1_dofs = MeshUtils.extract_entity_dofs(
    V, mesh.facet_dim,
    MeshUtils.extract_entity_indices(mesh.domain, style='facet', marker=None)
)
bc1: dolfinx.fem.DirichletBC = dolfinx.fem.dirichletbc(0.0, bc1_dofs, V)

state_problem_1 = create_state_problem(
    name='state_1', F_form=F1_form, state=u1, adjoint=v1, is_linear=True,
    bcs_info=[
        (bc1, V, bc1_dofs, 0.0)
    ],
    state_ksp_option={
        'ksp_type': 'preonly', 'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_24": 1
    },
    adjoint_ksp_option={
        'ksp_type': 'preonly', 'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_24": 1
    }
)
state_problems.append(state_problem_1)

# ------ Define Control problem
control_problem = create_control_problem(
    controls=[f1],
    bcs_info={f1.name: []},
    gradient_ksp_options={
        f1.name: {
            'ksp_type': 'preonly', 'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_24": 1
        }
    }
)

# ------ Define Cost Function
f_d = dolfinx.fem.Function(V)
f_d.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
cost1_form = 0.5 * ufl.inner(u1 - f_d, u1 - f_d) * ufl.dx
cost1_fun = IntegralFunctional(name='cost1', domain=mesh.domain, form=cost1_form, weight=1.0)
cost_func_list.append(cost1_fun)

# ------ Define Optimal Problem
state_system = StateProblem(state_problems)
opt_problem = OptimalControlProblem(
    state_system=state_system,
    control_problem=control_problem,
    cost_functional_list=cost_func_list,
    control_regularization_pipe=ControlRegularizationPipe([])
)

loss_init = opt_problem.evaluate_cost_functional(mesh.domain.comm, update_state=True)
print(f"step/{0} loss:{loss_init}")

is_solved, step = False, 0
while True:
    step += 1

    is_solved, grads_dict = opt_problem.compute_gradient(
        mesh.domain.comm,
        state_kwargs={
            'mat_assemble_method': 'lift',
            'with_debug': False
        },
        adjoint_kwargs={
            'mat_assemble_method': 'lift',
            'with_debug': False
        },
        gradient_kwargs={
            'mat_assemble_method': 'lift',
            'with_debug': False
        },
        update_state=True,
        update_adjoint=True
    )
    if not is_solved:
        print(f'[Info]: {grads_dict["log"]}')
        break

    for control in control_problem.controls:
        control_name = control.name
        grad = grads_dict[control_name]

        grad_np: np.array = grad.x.array
        grad_np = grad_np / np.linalg.norm(grad_np, ord=2)

        control_np: np.array = control.x.array
        control_np = control_np - 2.0 * grad_np

        control.x.array[:] = control_np

    loss = opt_problem.evaluate_cost_functional(mesh.domain.comm, update_state=False)
    print(f"{step} loss:{loss} rho:{loss/loss_init}")

    if loss / loss_init < 0.01:
        break

if is_solved:
    mesh.grid.point_data['control'] = f1.x.array
    mesh.grid.plot()
