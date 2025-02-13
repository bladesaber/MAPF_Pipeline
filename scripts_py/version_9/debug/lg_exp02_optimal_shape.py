"""
Ref: https://github.com/sblauth/cashocs/tree/main/demos/documented/shape_optimization/shape_poisson
"""

import numpy as np
import dolfinx
import ufl
from mpi4py import MPI

from scripts_py.version_9.dolfinx_grad.lagrange_method.type_database import create_state_problem, create_shape_problem
from scripts_py.version_9.dolfinx_grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.cost_functions import IntegralFunctional
from scripts_py.version_9.dolfinx_grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.shape_regularization import ShapeRegularizationPipe
from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils, MeshObj
from scripts_py.version_9.dolfinx_grad.dynamic_recorder import VtuRecorder

mesh = MeshObj(mesh_file='/home/admin123456/Desktop/work/topopt_exps/possion_shape/model.msh', dim=2)
# mesh = MeshObj(domain=dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 25, 25))
V = dolfinx.fem.FunctionSpace(mesh.domain, ("Lagrange", 1))
state_problems = []
cost_func_list = []

# ------ Define State Problem 1
u1 = dolfinx.fem.Function(V, name='state_1')
v1 = dolfinx.fem.Function(V, name='adjoint_1')

coord = MeshUtils.define_coordinate(mesh.domain)  # This is global coordinate
f_exp = 2.5 * np.power(coord[0] + 0.4 - np.power(coord[1], 2), 2) + np.power(coord[0], 2) + np.power(coord[1], 2) - 1
F1_form = ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx - f_exp * v1 * ufl.dx

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
control_problem = create_shape_problem(
    domain=mesh.domain, bcs_info=[],
    gradient_ksp_option={
        'ksp_type': 'preonly', 'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_24": 1
    }
)

# ------ Define Cost Function
cost1_form = u1 * ufl.dx
cost1_fun = IntegralFunctional(name='cost1', domain=mesh.domain, form=cost1_form, weight=1.0)
cost_func_list.append(cost1_fun)

# ------ Define Optimal Problem
state_system = StateProblem(state_problems)
opt_problem = OptimalShapeProblem(
    state_system=state_system,
    shape_problem=control_problem,
    cost_functional_list=cost_func_list,
    shape_regularization_list=ShapeRegularizationPipe([]),
    scalar_product=None
)

recorder = VtuRecorder(mesh, save_dir='/home/admin123456/Desktop/work/topopt_exps/new_test/example02')
recorder.add_point_scalar('u1', u1.x.array)
recorder.save(0)
loss_cur = opt_problem.evaluate_cost_functional(mesh.domain.comm, update_state=True)
print(f"step/{0} loss:{loss_cur}")

is_solved, step = False, 0
while True:
    step += 1

    is_solved, shape_grad = opt_problem.compute_gradient(
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
        scalar_product_kwargs={
            'with_debug': False
        },
        update_state=True,
        update_adjoint=True
    )
    if not is_solved:
        print(f'[Info]: {shape_grad["log"]}')
        break

    shape_grad_np = shape_grad.x.array
    shape_grad_np = shape_grad_np / np.linalg.norm(shape_grad_np, ord=2)
    shape_grad_np = shape_grad_np * -0.2

    displacement_np = np.zeros(mesh.domain.geometry.x.shape)
    displacement_np[:, :mesh.topology_dim] = shape_grad_np.reshape((-1, mesh.topology_dim))

    mesh.deform(displacement_np)
    recorder.add_point_scalar('u1', u1.x.array)
    recorder.save(step)

    loss = opt_problem.evaluate_cost_functional(mesh.domain.comm, update_state=True)
    print(f"{loss:.3f}, {loss/loss_cur:.3f}")

    if step > 50:
        break

    loss_cur = loss
