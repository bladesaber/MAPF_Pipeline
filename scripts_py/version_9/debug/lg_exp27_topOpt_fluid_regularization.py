import shutil
import numpy as np
import ufl
from basix.ufl import element
import dolfinx
from typing import Dict, Union
from functools import partial
from sklearn.neighbors import KDTree
import os
from ufl import div, inner, grad
from scipy import sparse
from tqdm import tqdm
from mpi4py import MPI

from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_control_problem, create_state_problem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalControlProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.control_regularization import ControlRegularization, \
    ControlIntegralRegularization
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import AssembleUtils

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/multi_fluid_01'
model_xdmf = os.path.join(proj_dir, 'model.xdmf')

res_dir = os.path.join(proj_dir, 'res')
if os.path.exists(res_dir):
    shutil.rmtree(res_dir)
os.mkdir(res_dir)

# MeshUtils.msh_to_XDMF(
#     name='model',
#     msh_file=os.path.join(proj_dir, 'model.msh'),
#     output_file=model_xdmf,
#     dim=2
# )

domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=model_xdmf,
    mesh_name='model',
    cellTag_name='model_cells',
    facetTag_name='model_facets'
)
tdim = domain.topology.dim
fdim = tdim - 1
grid = VisUtils.convert_to_grid(domain)

# ------
P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
TH = ufl.MixedElement([P2, P1])
W = dolfinx.fem.FunctionSpace(domain, TH)
W0, W1 = W.sub(0), W.sub(1)
V, V_to_W = W0.collapse()
Q, Q_to_W = W1.collapse()
V_control = dolfinx.fem.FunctionSpace(domain, ("DG", 0))


def inflow_velocity_exp(x, dire_axis: int):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[dire_axis] = 1.0
    return values


# def alpha_fun(rho: Union[np.ndarray, dolfinx.fem.Function], alpha_min=0.01, alpha_max=1000.0, p=0.1):
#     # rho is smaller, alpha is larger.
#     return alpha_max + (alpha_min - alpha_max) * rho * (1 + p) / (rho + p)


def alpha_fun(rho: Union[np.ndarray, dolfinx.fem.Function], alpha_min=0.01, alpha_max=1000.0, p=3.0):
    return alpha_min + ufl.elem_pow(1.0 - rho, p) * (alpha_max - alpha_min)


def generate_problem(
        tag: str, rho: dolfinx.fem.Function,
        input_marker, output_marker, boundary_markers,
        dire_axis
):
    up = dolfinx.fem.Function(W, name=f"state_{tag}")
    u, p = ufl.split(up)  # please don't use up.split()
    vq = dolfinx.fem.Function(W, name=f"adjoint_{tag}")
    v, q = ufl.split(vq)  # please don't use vq.split()
    f = dolfinx.fem.Constant(domain, np.zeros(tdim))
    eq_form = (
            alpha_fun(rho) * inner(u, v) * ufl.dx
            + (inner(grad(u), grad(v)) - div(v) * p - q * div(u)) * ufl.dx
            - inner(f, v) * ufl.dx
    )  # Stokes Equation

    bcs_info_state = []
    for marker in boundary_markers:
        bc_value = dolfinx.fem.Function(V, name=f"boundary_{marker}")
        bc_dofs = MeshUtils.extract_entity_dofs(
            (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
        )
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, W0)
        bcs_info_state.append((bc, W0, bc_dofs, bc_value))

    inlet_value = dolfinx.fem.Function(V, name='inlet_u')
    inlet_value.interpolate(partial(inflow_velocity_exp, dire_axis=dire_axis))
    inlet_dofs = MeshUtils.extract_entity_dofs(
        (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
    )
    inlet_bc = dolfinx.fem.dirichletbc(inlet_value, inlet_dofs, W0)
    bcs_info_state.append((inlet_bc, W0, inlet_dofs, inlet_value))

    outlet_value = dolfinx.fem.Function(Q, name='outlet_p')
    outlet_dofs = MeshUtils.extract_entity_dofs(
        (W1, Q), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, output_marker)
    )
    outlet_bc = dolfinx.fem.dirichletbc(outlet_value, outlet_dofs, W1)
    bcs_info_state.append((outlet_bc, W1, outlet_dofs, outlet_value))

    state_problem = create_state_problem(
        name=f"problem{tag}",
        F_form=eq_form, state=up, adjoint=vq,
        is_linear=True,
        bcs_info=bcs_info_state,
        state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
    )

    volume = AssembleUtils.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(domain, 1.0) * ufl.dx))
    reg_volume = ControlIntegralRegularization(domain, rho_1, mu=100.0, target_control_volume=volume * 0.2)

    cost_form = 0.5 * alpha_fun(rho) * inner(u, u) * ufl.dx + inner(grad(u), grad(u)) * ufl.dx
    # cost_form = inner(grad(u), grad(u)) * ufl.dx  # todo why can't
    cost_fun = IntegralFunction(domain=domain, form=cost_form)

    sub_res_dir = os.path.join(res_dir, f"res_problem{tag}")
    os.mkdir(sub_res_dir)

    rho_recorder = VTKRecorder(file=os.path.join(sub_res_dir, 'rho.pvd'))
    u_recorder = VTKRecorder(file=os.path.join(sub_res_dir, 'u.pvd'))

    return state_problem, cost_fun, [reg_volume], {rho.name: rho_recorder, f"{up.name}_u": u_recorder}


state_problems = []
recorder_dict = {}
regularization_list = []

rho_1 = dolfinx.fem.Function(V_control, name='control_1')
rho_1.x.array[:] = 0.1
problem1, cost1_fun, reg_list1, record_info1 = generate_problem(
    tag='1', rho=rho_1,
    input_marker=18, output_marker=19, boundary_markers=[20],
    dire_axis=0
)
state_problems.append(problem1)
recorder_dict.update(record_info1)
regularization_list.extend(reg_list1)

state_system = StateProblem(state_problems)
control_problem = create_control_problem(
    controls=[rho_1],
    bcs_info={rho_1.name: []},
    gradient_ksp_options={
        rho_1.name: {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
    }
)

opt_problem = OptimalControlProblem(
    state_system=state_system,
    control_problem=control_problem,
    cost_functional_list=[cost1_fun],
    control_regularization=ControlRegularization(regularization_list)
)

# # ------ debug for solving state problem
# state_system.solve(comm=MPI.COMM_WORLD, with_debug=True)
# for problem in state_system.state_problems:
#     state_dir = os.path.join(proj_dir, f"state_{problem.name}")
#     if os.path.exists(state_dir):
#         shutil.rmtree(state_dir)
#     os.mkdir(state_dir)
#
#     u_recorder = VTKRecorder(os.path.join(state_dir, f"u_0.pvd"))
#     u_recorder.write_function(problem.state.sub(0).collapse(), step=0)
#     p_recorder = VTKRecorder(os.path.join(state_dir, f"p_0.pvd"))
#     p_recorder.write_function(problem.state.sub(1).collapse(), step=0)
# # ------

radius_filter = 0.2
cell_centers = grid.cell_centers().points
cells_num = cell_centers.shape[0]
row_idxs = []
center_tree = KDTree(cell_centers, metric='minkowski')
neighbour_idxs, dists = center_tree.query_radius(cell_centers, r=radius_filter, return_distance=True)
for row_id, neg_idxs in tqdm(enumerate(neighbour_idxs)):
    row_idxs.append(np.full(shape=(neg_idxs.shape[0],), fill_value=row_id))
row_idxs = np.concatenate(row_idxs, axis=-1).reshape(-1)
col_idxs = np.concatenate(neighbour_idxs, axis=-1).reshape(-1)
dists = radius_filter - np.concatenate(dists, axis=-1).reshape(-1)
dist_mat = sparse.coo_matrix((dists, (row_idxs, col_idxs)), shape=(cells_num, cells_num))
distSum_mat = dist_mat.sum(axis=1).reshape((-1, 1))


def update_grad(control: dolfinx.fem.Function, grad_dict: Dict[str, dolfinx.fem.Function]):
    grad_np: np.array = grad_dict[control.name].x.array
    grad_np = grad_np.reshape((-1, 1))
    grad_np = np.divide(dist_mat.dot(grad_np), distSum_mat + 1e-8)
    grad_np = np.asarray(grad_np).reshape(-1)
    grad_np = grad_np / np.linalg.norm(grad_np, ord=2)
    grad_np = grad_np * 0.1

    control_np: np.array = control.x.array
    control_np = control_np.reshape(-1)

    control_np = control_np - grad_np
    control_np = np.maximum(np.minimum(control_np, 0.99), 0.01)

    control.x.array[:] = control_np


# ------ debug for cost info
cost_info = opt_problem.get_cost_info(comm=MPI.COMM_WORLD, update_state=True)
print(cost_info)
# ------

last_loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True, with_debug=True)
print(f"[Debug]: Init loss:{last_loss}")

step = 0
while True:
    step += 1

    grads_dict: Dict[str, dolfinx.fem.Function] = opt_problem.compute_gradient(domain.comm)
    for control in control_problem.controls:
        update_grad(control, grads_dict)

    loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=False)
    print(f"[Debug]: Step_{step} loss:{loss:.6f}")

    if step % 3 == 0:
        for control in control_problem.controls:
            recorder_dict[control.name].write_function(control, step=step)
        for problem in state_system.state_problems:
            recorder_dict[f"{problem.state.name}_u"].write_function(problem.state.sub(0).collapse(), step=step)

    if step % 10 == 0:
        cost_info = opt_problem.get_cost_info(comm=MPI.COMM_WORLD, update_state=False)
        print(cost_info)

    if step > 500:
        break
