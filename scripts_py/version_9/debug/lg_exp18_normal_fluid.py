import numpy as np
import ufl
from basix.ufl import element
import dolfinx
from typing import Dict
from functools import partial
import os
from ufl import div, inner, grad

from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_shape_problem, create_state_problem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import ScalarTrackingFunctional, IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import VolumeRegularization
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, AssembleUtils

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/fluid_shape3'
model_xdmf = os.path.join(proj_dir, 'model.xdmf')

# ------ create xdmf
msh_file = os.path.join(proj_dir, 'model.msh')
MeshUtils.msh_to_XDMF(
    name='model',
    msh_file=os.path.join(proj_dir, 'model.msh'),
    output_file=model_xdmf,
    dim=2
)
# -------------------

domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=model_xdmf,
    mesh_name='model',
    cellTag_name='model_cells',
    facetTag_name='model_facets'
)
tdim = domain.topology.dim
fdim = tdim - 1
grid = VisUtils.convert_to_grid(domain)

state_problems = []
# ------ Define State Problem 1
input_marker = 26
output_markers = [28, 29, 30]
bry_fixed_marker = 27
bry_free_marker = 31

P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
TH = ufl.MixedElement([P2, P1])
W = dolfinx.fem.FunctionSpace(domain, TH)
W0, W1 = W.sub(0), W.sub(1)
V, V_to_W = W0.collapse()
Q, Q_to_W = W1.collapse()

up = dolfinx.fem.Function(W, name='state_1')
u, p = ufl.split(up)  # please don't use up.split()
vq = dolfinx.fem.Function(W, name='adjoint_1')
v, q = ufl.split(vq)  # please don't use vq.split()
f = dolfinx.fem.Constant(domain, np.zeros(tdim))

nu = 1. / 400.
F1_form = nu * inner(grad(u), grad(v)) * ufl.dx - \
          p * div(v) * ufl.dx + div(u) * q * ufl.dx - \
          inner(f, v) * ufl.dx

# ------ define state problem 1 boundary
bc0_value = dolfinx.fem.Function(V, name='bry_u0')
bc0_dofs = MeshUtils.extract_entity_dofs(
    (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, bry_fixed_marker)
)
bc0 = dolfinx.fem.dirichletbc(bc0_value, bc0_dofs, W0)

bc1_value = dolfinx.fem.Function(V, name='bry_u1')
bc1_dofs = MeshUtils.extract_entity_dofs(
    (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, bry_free_marker)
)
bc1 = dolfinx.fem.dirichletbc(bc1_value, bc1_dofs, W0)


def inflow_velocity_exp(x):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[0] = 5.0 * (1.5 - x[1]) * x[1]
    return values


bc2_value = dolfinx.fem.Function(V, name='inflow_u')
bc2_value.interpolate(inflow_velocity_exp)
bc2_dofs = MeshUtils.extract_entity_dofs(
    (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
)
bc2 = dolfinx.fem.dirichletbc(bc2_value, bc2_dofs, W0)


bc3_value = dolfinx.fem.Function(Q, name='outflow1_p')
bc3_dofs = MeshUtils.extract_entity_dofs(
    (W1, Q), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, output_markers[0])
)
bc3 = dolfinx.fem.dirichletbc(bc3_value, bc3_dofs, W1)

# bc4_value = dolfinx.fem.Function(Q, name='outflow2_p')
# bc4_dofs = MeshUtils.extract_entity_dofs(
#     (W1, Q), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, output_markers[1])
# )
# bc4 = dolfinx.fem.dirichletbc(bc4_value, bc4_dofs, W1)
#
# bc5_value = dolfinx.fem.Function(Q, name='outflow3_p')
# bc5_dofs = MeshUtils.extract_entity_dofs(
#     (W1, Q), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, output_markers[2])
# )
# bc5 = dolfinx.fem.dirichletbc(bc5_value, bc5_dofs, W1)

state_problem_1 = create_state_problem(
    name='state_1', F_form=F1_form, state=up, adjoint=vq, is_linear=True,
    bcs_info=[
        (bc0, W0, bc0_dofs, bc0_value),
        (bc1, W0, bc1_dofs, bc1_value),
        (bc2, W0, bc2_dofs, bc2_value),
        (bc3, W1, bc3_dofs, bc3_value),
        # (bc4, W1, bc4_dofs, bc4_value),
        # (bc5, W1, bc5_dofs, bc5_value),
    ],
    state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'ksp', 'pc_factor_mat_solver_type': 'mumps'},
    adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'ksp', 'pc_factor_mat_solver_type': 'mumps'}
)
state_problems.append(state_problem_1)

# ------ Define Control problem
coordinate_space = domain.ufl_domain().ufl_coordinate_element()
V_S = dolfinx.fem.FunctionSpace(domain, coordinate_space)

bc6_value = dolfinx.fem.Function(V_S, name='fix_bry_shape')
bc6_dofs = MeshUtils.extract_entity_dofs(
    V_S, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, bry_fixed_marker)
)
bc6 = dolfinx.fem.dirichletbc(bc6_value, bc6_dofs, None)

bc7_value = dolfinx.fem.Function(V_S, name='fix_bry_input')
bc7_dofs = MeshUtils.extract_entity_dofs(
    V_S, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
)
bc7 = dolfinx.fem.dirichletbc(bc7_value, bc7_dofs, None)

bc8_value = dolfinx.fem.Function(V_S, name='fix_bry_output1')
bc8_dofs = MeshUtils.extract_entity_dofs(
    V_S, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, output_markers[0])
)
bc8 = dolfinx.fem.dirichletbc(bc8_value, bc8_dofs, None)

bc9_value = dolfinx.fem.Function(V_S, name='fix_bry_output2')
bc9_dofs = MeshUtils.extract_entity_dofs(
    V_S, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, output_markers[1])
)
bc9 = dolfinx.fem.dirichletbc(bc9_value, bc9_dofs, None)

bc10_value = dolfinx.fem.Function(V_S, name='fix_bry_output3')
bc10_dofs = MeshUtils.extract_entity_dofs(
    V_S, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, output_markers[2])
)
bc10 = dolfinx.fem.dirichletbc(bc10_value, bc10_dofs, None)

control_problem = create_shape_problem(
    domain=domain,
    bcs_info=[
        (bc6, V_S, bc6_dofs, bc6_value),
        (bc7, V_S, bc7_dofs, bc7_value),
        (bc8, V_S, bc8_dofs, bc8_value),
        (bc9, V_S, bc9_dofs, bc9_value),
        (bc10, V_S, bc10_dofs, bc10_value),
    ],
    lambda_lame=0.0,
    damping_factor=0.0,
    gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'ksp'}
)

# ------ Define Cost Function
# cost1_form = nu * inner(grad(u), grad(u)) * ufl.dx
# cost1_fun = IntegralFunction(cost1_form)

ds = MeshUtils.define_ds(domain, facet_tags)
n_vec = MeshUtils.define_facet_norm(domain)
tracking_goal = AssembleUtils.assemble_scalar(dolfinx.fem.form(
    ufl.dot(bc2_value, n_vec) * ds(input_marker)
)) / 3.0

cost_functional_list = []
for output_marker in output_markers:
    integrand_form = ufl.dot(u, n_vec) * ds(output_marker)
    cost_functional_list.append(
        ScalarTrackingFunctional(domain, integrand_form, tracking_goal)
    )

# ------ Define Optimal Problem
# volume_reg = VolumeRegularization(control_problem, mu=0.2, target_volume_rho=1.0)

opt_problem = OptimalShapeProblem(
    state_problems=state_problems,
    shape_problem=control_problem,
    # shape_regulariztions=ShapeRegularization(regularization_list=[
    #     volume_reg
    # ]),
    shape_regulariztions=ShapeRegularization([]),
    cost_functional_list=cost_functional_list,
    scalar_product=None
)

# opt_problem.compute_state(domain.comm, with_debug=True)
# u_res = up.sub(0).collapse()
# p_res = up.sub(1).collapse()
# VisUtils.show_arrow_res_vtk(grid, u_res, V, scale=0.1)

last_loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)

# ------ Recorder Init
u_recorder = VTKRecorder(file=os.path.join(proj_dir, 'opt', 'u_res.pvd'))
u_recorder.write_mesh(domain, 0)

# ------ begin to optimize
best_loss = np.inf
step = 0
while True:
    step += 1

    shape_grad: dolfinx.fem.Function = opt_problem.compute_gradient(domain.comm)

    shape_grad_np = shape_grad.x.array
    # shape_grad_np = shape_grad_np / np.linalg.norm(shape_grad_np, ord=2)
    shape_grad_np = shape_grad_np * -0.35

    displacement_np = np.zeros(domain.geometry.x.shape)
    displacement_np[:, :tdim] = shape_grad_np.reshape((-1, tdim))

    # dJ = dolfinx.fem.Function(shape_grad.function_space)
    # dJ.x.array[:] = shape_grad_np.reshape(-1)
    # VisUtils.show_vector_res_vtk(grid, dJ, dim=2, with_wrap=True)

    if step % 10 == 0:
        u_res = up.sub(0).collapse()
        u_recorder.write_function(u_res, step)

    MeshUtils.move(domain, displacement_np)
    loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)

    best_loss = np.minimum(loss, best_loss)
    print(f"[###Step {step}] loss:{loss:.5f} / best_loss:{best_loss:.5f}")

    if loss > best_loss * 1.25:
        break

    if step > 50:
        break

step += 1
u_res = up.sub(0).collapse()
u_recorder.write_function(u_res, step)

