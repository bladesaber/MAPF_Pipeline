import numpy as np
import ufl
import dolfinx
import shutil
import os
from ufl import div, inner, grad

from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_shape_problem, create_state_problem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import ScalarTrackingFunctional, IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import VolumeRegularization
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder, TensorBoardRecorder
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, AssembleUtils

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/fluid_shape4'
model_xdmf = os.path.join(proj_dir, 'model.xdmf')

# # ------ create xdmf
# msh_file = os.path.join(proj_dir, 'model.msh')
# MeshUtils.msh_to_XDMF(
#     name='model',
#     msh_file=os.path.join(proj_dir, 'model.msh'),
#     output_file=model_xdmf,
#     dim=2
# )
# # -------------------

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
input_marker = 1
output_markers = [5, 6, 7]
bry_markers = [2, 3, 4]

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
bcs_info = []

for marker in bry_markers:
    bc_value = dolfinx.fem.Function(V, name=f"bry_u{marker}")
    bc_dofs = MeshUtils.extract_entity_dofs(
        (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
    )
    bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, W0)
    bcs_info.append((bc, W0, bc_dofs, bc_value))


def inflow_velocity_exp(x):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[0] = 6.0 * (0.0 - x[1]) * (x[1] + 1.0)
    return values


bc_in1_value = dolfinx.fem.Function(V, name='inflow_u')
bc_in1_value.interpolate(inflow_velocity_exp)
bc_in1_dofs = MeshUtils.extract_entity_dofs(
    (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
)
bc_in1 = dolfinx.fem.dirichletbc(bc_in1_value, bc_in1_dofs, W0)
bcs_info.append((bc_in1, W0, bc_in1_dofs, bc_in1_value))

bc_out1_value = dolfinx.fem.Function(Q, name='outflow1_p')
bc_out1_dofs = MeshUtils.extract_entity_dofs(
    (W1, Q), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, output_markers[0])
)
bc_out1 = dolfinx.fem.dirichletbc(bc_out1_value, bc_out1_dofs, W1)
bcs_info.append((bc_out1, W1, bc_out1_dofs, bc_out1_value))

state_problem_1 = create_state_problem(
    name='state_1', F_form=F1_form, state=up, adjoint=vq, is_linear=True,
    bcs_info=bcs_info,
    state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'ksp', 'pc_factor_mat_solver_type': 'mumps'},
    adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'ksp', 'pc_factor_mat_solver_type': 'mumps'}
)
state_problems.append(state_problem_1)

# ------ Define Control problem
bry_fixed_markers = [1, 4, 5, 6, 7]
bry_free_marker = [2, 3]

coordinate_space = domain.ufl_domain().ufl_coordinate_element()
V_S = dolfinx.fem.FunctionSpace(domain, coordinate_space)

bcs_info = []
for marker in bry_fixed_markers:
    bc_value = dolfinx.fem.Function(V_S, name=f"fix_bry_shape_{marker}")
    bc_dofs = MeshUtils.extract_entity_dofs(V_S, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker))
    bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, None)
    bcs_info.append((bc, V_S, bc_dofs, bc_value))

control_problem = create_shape_problem(
    domain=domain,
    bcs_info=bcs_info,
    lambda_lame=0.0,
    damping_factor=0.0,
    gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'ksp'}
)

# ------ Define Cost Function
# cost1_form = nu * inner(grad(u), grad(u)) * ufl.dx
# cost1_fun = IntegralFunction(cost1_form)

ds = MeshUtils.define_ds(domain, facet_tags)
n_vec = MeshUtils.define_facet_norm(domain)
tracking_goal = -1.0 * AssembleUtils.assemble_scalar(dolfinx.fem.form(
    ufl.dot(bc_in1_value, n_vec) * ds(input_marker)
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
# VisUtils.show_arrow_res_vtk(grid, u_res, V, scale=0.3)

last_loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)

# ------ Recorder Init
u_recorder = VTKRecorder(file=os.path.join(proj_dir, 'opt', 'u_res.pvd'))
u_recorder.write_mesh(domain, 0)

log_dir = os.path.join(proj_dir, 'log')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.mkdir(log_dir)
tensor_recorder = TensorBoardRecorder(log_dir=log_dir)

# ------ begin to optimize
best_loss = np.inf
step = 0
while True:
    step += 1

    shape_grad: dolfinx.fem.Function = opt_problem.compute_gradient(domain.comm)

    shape_grad_np = shape_grad.x.array
    # shape_grad_np = shape_grad_np / np.linalg.norm(shape_grad_np, ord=2)
    shape_grad_np = shape_grad_np * -0.5

    displacement_np = np.zeros(domain.geometry.x.shape)
    displacement_np[:, :tdim] = shape_grad_np.reshape((-1, tdim))

    # dJ = dolfinx.fem.Function(shape_grad.function_space)
    # dJ.x.array[:] = shape_grad_np.reshape(-1)
    # VisUtils.show_vector_res_vtk(grid, dJ, dim=2, with_wrap=True)

    if step % 1 == 0:
        # dJ = dolfinx.fem.Function(shape_grad.function_space)
        # dJ.x.array[:] = shape_grad_np.reshape(-1)
        # VisUtils.show_vector_res_vtk(grid, dJ, dim=2, with_wrap=True)

        u_res = up.sub(0).collapse()
        u_recorder.write_function(u_res, step)

    MeshUtils.move(domain, displacement_np)
    loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
    tensor_recorder.write_scalar('loss', loss, step)

    # ------ record velocity
    out_flow_dict = {}
    for marker in output_markers:
        out_vel_form = ufl.dot(u, n_vec) * ds(marker)
        out_flow = AssembleUtils.assemble_scalar(dolfinx.fem.form(out_vel_form))
        out_flow_dict[f"out_{marker}"] = out_flow
    tensor_recorder.write_scalars('flow', out_flow_dict, step)

    best_loss = np.minimum(loss, best_loss)
    print(f"[###Step {step}] loss:{loss:.5f} / best_loss:{best_loss:.5f}")

    if loss > best_loss * 1.5:
        break

    if step > 50:
        break

step += 1
u_res = up.sub(0).collapse()
u_recorder.write_function(u_res, step)
