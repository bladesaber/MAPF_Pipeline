import numpy as np
import ufl
from basix.ufl import element
import dolfinx
from typing import Dict
from functools import partial
from sklearn.neighbors import KDTree
import os
from ufl import div, inner, grad
from scipy import sparse
from tqdm import tqdm

from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_control_problem, create_state_problem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalControlProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import IntegralFunction
from scripts_py.version_9.dolfinx_Grad.recorder_utils import XDMFRecorder
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/fluid_top2'
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
input_marker = 18
output_marker = 19
boundary_marker = 20

P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
TH = ufl.MixedElement([P2, P1])
W = dolfinx.fem.FunctionSpace(domain, TH)
W0, W1 = W.sub(0), W.sub(1)
V, V_to_W = W0.collapse()
Q, Q_to_W = W1.collapse()

V_control = dolfinx.fem.FunctionSpace(domain, ("DG", 0))

up = dolfinx.fem.Function(W, name='state_1')
u, p = ufl.split(up)  # please don't use up.split()
vq = dolfinx.fem.Function(W, name='adjoint_1')
v, q = ufl.split(vq)  # please don't use vq.split()

f = dolfinx.fem.Constant(domain, np.zeros(tdim))
rho = dolfinx.fem.Function(V_control, name='control_1')
rho.x.array[:] = 0.5


def alpha_func(rho, alpha_bar, alpha_under_bar, adj_q):
    """
    rho is smaller, alpha is larger.
    """
    return alpha_bar + (alpha_under_bar - alpha_bar) * rho * (1 + adj_q) / (rho + adj_q)


alpha_func = partial(alpha_func, alpha_bar=1000., alpha_under_bar=0.01, adj_q=0.1)

F1_form = alpha_func(rho) * inner(u, v) * ufl.dx + \
         (inner(grad(u), grad(v)) - div(v) * p - q * div(u)) * ufl.dx - \
         inner(f, v) * ufl.dx

bc0_value = dolfinx.fem.Function(V, name='boundary_u')
bc0_dofs = MeshUtils.extract_entity_dofs(
    (W0, V), fdim,
    MeshUtils.extract_facet_entities(domain, facet_tags, boundary_marker)
)
bc0 = dolfinx.fem.dirichletbc(bc0_value, bc0_dofs, W0)


def inflow_velocity_exp(x):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[0] = 1.0
    return values


bc1_value = dolfinx.fem.Function(V, name='inflow_u')
bc1_value.interpolate(inflow_velocity_exp)
bc1_dofs = MeshUtils.extract_entity_dofs(
    (W0, V), fdim,
    MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
)
bc1 = dolfinx.fem.dirichletbc(bc1_value, bc1_dofs, W0)

bc2_value = dolfinx.fem.Function(Q, name='outflow_p')
bc2_dofs = MeshUtils.extract_entity_dofs(
    (W1, Q), fdim,
    MeshUtils.extract_facet_entities(domain, facet_tags, output_marker)
)
bc2 = dolfinx.fem.dirichletbc(bc2_value, bc2_dofs, W1)

state_problem_1 = create_state_problem(
    name='state_1', F_form=F1_form, state=up, adjoint=vq, is_linear=True,
    bcs_info=[
        (bc0, W0, bc0_dofs, bc0_value),
        (bc1, W0, bc1_dofs, bc1_value),
        (bc2, W1, bc2_dofs, bc2_value),
    ],
    state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'ksp', 'pc_factor_mat_solver_type': 'mumps'},
    adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'ksp', 'pc_factor_mat_solver_type': 'mumps'}
)
state_problems.append(state_problem_1)

state_system = StateProblem(state_problems)

# ------ Define Control problem
control_problem = create_control_problem(
    controls=[rho],
    bcs_info={rho.name: []},
    gradient_ksp_options={rho.name: {'ksp_type': 'preonly', 'pc_type': 'ksp'}}
)

# ------ Define Cost Function
cost1_form = 0.5 * alpha_func(rho) * inner(u, u) * ufl.dx + inner(grad(u), grad(u)) * ufl.dx
cost1_fun = IntegralFunction(domain=domain, form=cost1_form)

# ------ Define Optimal Problem
opt_problem = OptimalControlProblem(
    state_system=state_system,
    control_problem=control_problem,
    cost_functional_list=[cost1_fun]
)
last_loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)

# ------ define distance matrix
rmin = 0.15

cell_centers = grid.cell_centers().points
cells_num = cell_centers.shape[0]

center_tree = KDTree(cell_centers, metric='minkowski')
neighbour_idxs, dists = center_tree.query_radius(cell_centers, r=rmin, return_distance=True)

row_idxs = []
for row_id, neig_idxs in tqdm(enumerate(neighbour_idxs)):
    row_idxs.append(np.full(shape=(neig_idxs.shape[0],), fill_value=row_id))

row_idxs = np.concatenate(row_idxs, axis=-1).reshape(-1)
col_idxs = np.concatenate(neighbour_idxs, axis=-1).reshape(-1)
dists = rmin - np.concatenate(dists, axis=-1).reshape(-1)

dist_mat = sparse.coo_matrix((dists, (row_idxs, col_idxs)), shape=(cells_num, cells_num))
distSum_mat = dist_mat.sum(axis=1).reshape((-1, 1))

# ------ Recorder Init
rho_recorder = XDMFRecorder(file=os.path.join(proj_dir, 'rho_res.xdmf'))
rho_recorder.write_mesh(domain)

u_recorder = XDMFRecorder(file=os.path.join(proj_dir, 'u_res.xdmf'))
u_recorder.write_mesh(domain)

# ------ begin to optimize
step = 0
while True:
    step += 1

    grads_dict: Dict[str, dolfinx.fem.Function] = opt_problem.compute_gradient(domain.comm)
    grad_fun_np: np.array = grads_dict[rho.name].x.array
    grad_fun_np = grad_fun_np.reshape((-1, 1))
    rho_np: np.array = rho.x.array
    rho_np = rho_np.reshape((-1, 1))

    grad_fun_np: np.matrix = np.divide(
        dist_mat.dot(grad_fun_np * rho_np),
        np.multiply(rho_np, distSum_mat)
    )
    grad_fun_np = np.asarray(grad_fun_np).reshape(-1)

    grad_fun_np = grad_fun_np - np.minimum(np.mean(grad_fun_np), 0.0)
    grad_fun_np = np.minimum(grad_fun_np, 0.0)
    grad_fun_np = grad_fun_np / np.max(np.abs(grad_fun_np))  # may be better

    rho_np = rho_np.reshape(-1)
    rho_np = rho_np - np.maximum(np.minimum(0.35 * grad_fun_np, 0.1), -0.1)
    rho_np = np.maximum(np.minimum(rho_np, 0.99), 0.01)

    rho.x.array[:] = rho_np

    loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=False)
    print(f"Step:{step} loss:{loss:.6f}")

    if step % 5 == 0:
        rho_recorder.write_function(rho, step)

        u_res = up.sub(0).collapse()
        u_res.x.scatter_forward()
        P3 = element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
        u_wri_res = dolfinx.fem.Function(dolfinx.fem.functionspace(domain, P3))
        u_wri_res.interpolate(u_res)
        u_recorder.write_function(u_wri_res, step)

    if step > 200:
        break

    if (loss > last_loss * 1.5) or (np.abs(loss - last_loss) / last_loss < 1e-8):
        break
    last_loss = loss

# ---------------------
step += 1
rho_recorder.write_function(rho, step)

u_res = up.sub(0).collapse()
u_res.x.scatter_forward()
P3 = element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
u_wri_res = dolfinx.fem.Function(dolfinx.fem.functionspace(domain, P3))
u_wri_res.interpolate(u_res)
u_recorder.write_function(u_wri_res, step)

