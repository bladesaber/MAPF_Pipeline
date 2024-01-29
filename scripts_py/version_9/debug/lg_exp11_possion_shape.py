import numpy as np
import os
import dolfinx
import ufl

from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_state_problem, create_shape_problem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, UFLUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/possion_shape'
model_xdmf_file = os.path.join(proj_dir, 'model.xdmf')

# ------ create xdmf
MeshUtils.msh_to_XDMF(
    name='model',
    msh_file=os.path.join(proj_dir, 'model.msh'),
    output_file=model_xdmf_file,
    dim=2
)
# ------

domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=model_xdmf_file,
    mesh_name='model',
    cellTag_name='model_cells',
    facetTag_name='model_facets'
)
grid = VisUtils.convert_to_grid(domain)

tdim = domain.topology.dim
fdim = tdim - 1

state_problems = []

# ------ Define State Problem 1
V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
u1 = dolfinx.fem.Function(V, name='state_1')
v1 = dolfinx.fem.Function(V, name='adjoint_1')
boundary_marker = 3

"""
Must Use Coordinate, The x of interpolate function of class(dolfinx.fem.Function) means the coordinate of Mesh
"""
coodr = MeshUtils.define_coordinate(domain)
f_exp = 2.5 * np.power(coodr[0] + 0.4 - np.power(coodr[1], 2), 2) + \
        np.power(coodr[0], 2) + np.power(coodr[1], 2) - 1

# vis_f = dolfinx.fem.Function(V, name='vis_f')
# vis_f.interpolate(UFLUtils.create_expression(f_exp, V))
# VisUtils.show_scalar_res_vtk(grid, 'vis_f', vis_f)

F1_form = ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx - f_exp * v1 * ufl.dx

facets = MeshUtils.extract_facet_entities(domain, facet_tags, boundary_marker)
bc1_dofs = MeshUtils.extract_entity_dofs(V, fdim, facets)
bc1: dolfinx.fem.DirichletBC = dolfinx.fem.dirichletbc(0.0, bc1_dofs, V)

state_problem_1 = create_state_problem(
    name='state_1', F_form=F1_form, state=u1, adjoint=v1, is_linear=True,
    bcs_info=[
        (bc1, V, bc1_dofs, 0.0)
    ],
    state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu'}
)
state_problems.append(state_problem_1)

# ------ Define Control problem
control_problem = create_shape_problem(
    domain=domain,
    bcs_info=[],
    lambda_lame=0.0,
    damping_factor=0.0,
    gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'ksp'}
)

# ------ Define Cost Function
cost1_form = u1 * ufl.dx
cost1_fun = IntegralFunction(cost1_form)

# ------ Define Optimal Problem
opt_problem = OptimalShapeProblem(
    state_problems=state_problems,
    shape_problem=control_problem,
    shape_regulariztions=ShapeRegularization(regularization_list=[]),
    cost_functional_list=[cost1_fun],
    scalar_product=None
)

# opt_problem.compute_state_problem(domain.comm)
last_loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
# VisUtils.show_scalar_res_vtk(grid, 'u1', u1)

step = 0
while True:
    step += 1

    shape_grad: dolfinx.fem.Function = opt_problem.compute_gradient(domain.comm)
    # VisUtils.show_scalar_res_vtk(grid, 'u_opt', u1)

    shape_grad_np = shape_grad.x.array
    # shape_grad_np = shape_grad_np / np.linalg.norm(shape_grad_np, ord=2)
    shape_grad_np = shape_grad_np * -0.2

    displacement_np = np.zeros(domain.geometry.x.shape)
    displacement_np[:, :tdim] = shape_grad_np.reshape((-1, tdim))

    if step % 50 == 0:
        dJ = dolfinx.fem.Function(shape_grad.function_space)
        dJ.x.array[:] = shape_grad_np.reshape(-1)
        VisUtils.show_vector_res_vtk(grid, dJ, dim=2, with_wrap=True)

        VisUtils.show_scalar_res_vtk(grid, 'u1', u1)

    MeshUtils.move(domain, displacement_np)
    cur_loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
    print(f"{cur_loss:.6f} / {last_loss:.6f}")

    if step > 300:
        break

    last_loss = cur_loss

# VisUtils.show_scalar_res_vtk(grid, 'u_opt', u1)
