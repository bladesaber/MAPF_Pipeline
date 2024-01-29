"""
Fail: I don't know how to homogenize boundary in boundary
"""

import numpy as np
import ufl
from mpi4py import MPI
import dolfinx
from typing import Dict

from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_control_problem, create_state_problem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalControlProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import IntegralFunction
from scripts_py.version_9.dolfinx_Grad.recorder_utils import XDMFRecorder
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils

domain: dolfinx.mesh.Mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 25, 25)
tdim = domain.topology.dim
fdim = tdim - 1
grid = VisUtils.convert_to_grid(domain)

state_problems = []
# ------ Define State Problem 1
V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))

u1 = dolfinx.fem.Function(V, name='State_1')
v1 = dolfinx.fem.Function(V, name='adjoint_1')
f1 = dolfinx.fem.Function(V, name='control_1')

F1_form = ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx - f1 * v1 * ufl.dx

facets = MeshUtils.extract_facet_entities(domain, marker=None)
bc1_dofs = MeshUtils.extract_entity_dofs(V, fdim, facets)
bc1: dolfinx.fem.DirichletBC = dolfinx.fem.dirichletbc(0.0, bc1_dofs, V)

state_problem_1 = create_state_problem(
    name='state_1', F_form=F1_form, state=u1, adjoint=v1, is_linear=True,
    bcs_info=[
        (bc1, V, bc1_dofs, 0.0)
    ],
    state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu'},
    adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu'}
)
state_problems.append(state_problem_1)

# ------ Define Control problem
f1_bc_dofs = MeshUtils.extract_entity_dofs(V, fdim, facets)
f1_bc0: dolfinx.fem.DirichletBC = dolfinx.fem.dirichletbc(0.0, f1_bc_dofs, V)

control_problem = create_control_problem(
    controls=[f1],
    bcs_info={
        f1.name: [(f1_bc0, V, f1_bc_dofs, 0.0)]
    },
    gradient_ksp_options={
        f1.name: {'ksp_type': 'preonly', 'pc_type': 'lu'}
    }
)

# ------ Define Cost Function
f_d = dolfinx.fem.Function(V)
f_d.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
cost1_form = 0.5 * ufl.inner(u1 - f_d, u1 - f_d) * ufl.dx
cost1_fun = IntegralFunction(cost1_form)

# ------ Define Optimal Problem
opt_problem = OptimalControlProblem(
    state_problems=state_problems,
    control_problem=control_problem,
    cost_functional_list=[cost1_fun]
)

orig_loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)

step = 0
while True:
    step += 1

    grads_dict: Dict[str, dolfinx.fem.Function] = opt_problem.compute_gradient(domain.comm)
    for control in control_problem.controls:
        control_name = control.name
        grad = grads_dict[control_name]

        grad_np: np.array = grad.x.array
        grad_np = grad_np / np.linalg.norm(grad_np, ord=2)

        control_np: np.array = control.x.array
        control_np = control_np - 2.0 * grad_np

        control.x.array[:] = control_np

    loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=False)
    opt_pcg = loss/orig_loss
    print(f"{step} loss: {loss}, Enhance:{opt_pcg}")

    # if step % 30 == 0:
    #     recorder.write_function(u.block_variable.checkpoint, step)

    if opt_pcg < 0.01:
        break

# recorder.write_function(u.block_variable.checkpoint, step=step)
VisUtils.show_scalar_res_vtk(grid, 'u_opt', u1)
