import numpy as np
import ufl
from basix.ufl import element
import dolfinx
from ufl import inner, grad, div
from functools import partial
from sklearn.neighbors import KDTree
from scipy import sparse
from tqdm import tqdm

from Thirdparty.pyadjoint.pyadjoint import *

from scripts_py.version_9.dolfinx_Grad.autoGrad_method.type_Function import Function
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.type_Mesh import Mesh
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.block_solve import solve
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.block_assemble import assemble
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.type_utils import start_annotation
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.type_DirichletBC import dirichletbc
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.recorder_utils import XDMFRecorder, TensorBoardRecorder

# ------ create xdmf
msh_file = '/home/admin123456/Desktop/work/topopt_exps/fluid_top1/model.msh'
MeshUtils.msh_to_XDMF(
    name='model',
    msh_file=msh_file,
    output_file='/home/admin123456/Desktop/work/topopt_exps/fluid_top1/model.xdmf',
    dim=2
)
# -------------------

tape = Tape()
set_working_tape(tape)


def alpha_func(rho, alpha_bar, alpha_under_bar, adj_q):
    """
    rho is smaller, alpha is larger.
    """
    return alpha_bar + (alpha_under_bar - alpha_bar) * rho * (1 + adj_q) / (rho + adj_q)


alpha_func = partial(
    alpha_func, alpha_bar=1000., alpha_under_bar=0.01, adj_q=0.1
)

with start_annotation():
    # ------ load domain
    input_marker = 18
    output_marker = 19
    noslip_marker = 20

    domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
        file='/home/admin123456/Desktop/work/topopt_exps/fluid_top1/model.xdmf',
        mesh_name='model',
        cellTag_name='model_cells',
        facetTag_name='model_facets'
    )
    domain = Mesh(domain)
    grid = VisUtils.convert_to_grid(domain)

    tdim = domain.topology.dim
    fdim = tdim - 1

    P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    TH = ufl.MixedElement([P2, P1])
    W = dolfinx.fem.FunctionSpace(domain, TH)
    W0, W1 = W.sub(0), W.sub(1)
    V, V_to_W = W0.collapse()
    Q, Q_to_W = W1.collapse()

    noslip: Function = Function(V, name='noslip')
    bc_noslip_facets = MeshUtils.extract_facet_entities(domain, facet_tags, noslip_marker)
    bc_noslip_dofs = MeshUtils.extract_entity_dofs((W0, V), fdim, bc_noslip_facets)
    bc0 = dirichletbc(noslip, bc_noslip_dofs, W0)


    def inflow_velocity_exp(x):
        num = x.shape[1]
        values = np.zeros((tdim, num))
        values[0] = 1.0
        return values


    inflow_velocity = Function(V, name='inflow_velocity')
    inflow_velocity.interpolate(inflow_velocity_exp)
    bc_input_facets = MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
    bc_input_dofs = MeshUtils.extract_entity_dofs((W0, V), fdim, bc_input_facets)
    bc1 = dirichletbc(inflow_velocity, bc_input_dofs, W0)

    zero_pressure = Function(Q, name='outflow_pressure')
    bc_output_facets = MeshUtils.extract_facet_entities(domain, facet_tags, output_marker)
    bc_output_dofs = MeshUtils.extract_entity_dofs((W1, Q), fdim, bc_output_facets)
    bc2 = dirichletbc(zero_pressure, bc_output_dofs, W1)

    bcs = [bc0, bc1, bc2]

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    f = dolfinx.fem.Constant(domain, np.zeros(tdim))
    # f: Function = Function(V)
    # f.x.array[:] = 0.0

    V_control = dolfinx.fem.FunctionSpace(domain, ("DG", 0))
    rho: Function = Function(V_control, name='control')

    rho.x.array[:] = 0.5
    F_form = alpha_func(rho) * inner(u, v) * ufl.dx + \
             (inner(grad(u), grad(v)) - div(v) * p - q * div(u)) * ufl.dx - \
             inner(f, v) * ufl.dx

    a_form: ufl.form.Form = ufl.lhs(F_form)
    L_form: ufl.form.Form = ufl.rhs(F_form)

    uh: Function = Function(W, name='state')
    uh = solve(
        uh, a_form, L_form, bcs,
        domain=domain, is_linear=True,
        tlm_ksp_option={
            'ksp_type': 'cg', 'pc_type': 'ksp',
        },
        adj_ksp_option={
            'ksp_type': 'preonly', 'pc_type': 'ksp',
        },
        forward_ksp_option={
            'ksp_type': 'preonly', 'pc_type': 'ksp', 'pc_factor_mat_solver_type': 'mumps',
        },
        with_debug=True,
        tlm_with_debug=False,
        adj_with_debug=False,
        recompute_with_debug=False,
    )

    u, p = ufl.split(uh)
    cost_form = 0.5 * alpha_func(rho) * inner(u, u) * ufl.dx + inner(grad(u), grad(u)) * ufl.dx

    Jhat = assemble(cost_form, domain)
    print(f"[### Test Jhat Cost]: {Jhat}")

control = Control(rho)
opt_problem = ReducedFunctional(Jhat, [control])

# grad_fun = opt_problem.derivative(adj_input=1.0)[0]
# print(np.any(np.isnan(grad_fun.x.array)), np.any(np.isinf(grad_fun.x.array)))

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
rho_recorder = XDMFRecorder(file='/home/admin123456/Desktop/work/topopt_exps/fluid_top1/rho_res.xdmf')
rho_recorder.write_mesh(domain)

u_recorder = XDMFRecorder(file='/home/admin123456/Desktop/work/topopt_exps/fluid_top1/u_res.xdmf')
u_recorder.write_mesh(domain)

# p_recorder = XDMFRecorder(file='/home/admin123456/Desktop/work/topopt_exps/fluid_top1/p_res.xdmf')
# p_recorder.write_mesh(domain)

# loss_recorder = TensorBoardRecorder(log_dir='/home/admin123456/Desktop/work/topopt_exps/fluid_top1/log')
# -----------------------

trial_rho: Function = Function(V_control)
trial_rho.assign(rho)
last_loss = opt_problem([trial_rho])
print(f"[### Original Cost]: {last_loss}")
# loss_recorder.write_scalar('loss', last_loss, 1)

step = 1
while True:
    step += 1

    grad_fun = opt_problem.derivative(adj_input=1.0)[0]

    grad_fun_np: np.array = grad_fun.x.array
    grad_fun_np = grad_fun_np.reshape((-1, 1))
    trial_np: np.array = trial_rho.x.array
    trial_np = trial_np.reshape((-1, 1))

    grad_fun_np: np.matrix = np.divide(
        dist_mat.dot(grad_fun_np * trial_np),
        np.multiply(trial_np, distSum_mat)
    )
    grad_fun_np = np.asarray(grad_fun_np).reshape(-1)

    grad_fun_np = grad_fun_np - np.minimum(np.mean(grad_fun_np), 0.0)
    grad_fun_np = np.minimum(grad_fun_np, 0.0)
    grad_fun_np = grad_fun_np / np.max(np.abs(grad_fun_np))  # may be better

    # grad_fun.x.array[:] = grad_fun_np
    # VisUtils.show_scalar_res_vtk(grid, 'grad', grad_fun, is_point_data=False)

    trial_np = trial_np.reshape(-1)
    trial_np = trial_np - np.maximum(np.minimum(0.1 * grad_fun_np, 0.1), -0.1)
    trial_np = np.maximum(np.minimum(trial_np, 0.99), 0.01)

    trial_rho.x.array[:] = trial_np
    # VisUtils.show_scalar_res_vtk(grid, 'rho', trial_rho, is_point_data=False)

    # alpha_np = alpha_func(trial_np)
    # alpha_field = dolfinx.fem.Function(trial_rho.function_space)
    # alpha_field.x.array[:] = alpha_np
    # VisUtils.show_scalar_res_vtk(grid, 'alpha', alpha_field, is_point_data=False)

    loss = opt_problem([trial_rho])
    print(f"Step:{step} loss:{loss:.6f}")

    if step % 5 == 0:
        rho_recorder.write_function(trial_rho, step)

        latest_uh: dolfinx.fem.Function = uh.block_variable.checkpoint

        u_res = latest_uh.sub(0).collapse()
        u_res.x.scatter_forward()
        P3 = element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
        u_wri_res = Function(dolfinx.fem.functionspace(domain, P3))
        u_wri_res.interpolate(u_res)
        u_recorder.write_function(u_wri_res, step)

        # p_res = latest_uh.sub(1).collapse()
        # p_res.x.scatter_forward()
        # p_recorder.write_function(p_res, step)

        # loss_recorder.write_scalar('loss', loss, step)

    if step > 200:
        break

    if (loss > last_loss * 1.5) or (np.abs(loss - last_loss) / last_loss < 1e-8):
        break
    last_loss = loss

# -----------------------------
step += 1

latest_uh: dolfinx.fem.Function = uh.block_variable.checkpoint

u_res = latest_uh.sub(0).collapse()
u_res.x.scatter_forward()
P3 = element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
u_wri_res = Function(dolfinx.fem.functionspace(domain, P3))
u_wri_res.interpolate(u_res)
u_recorder.write_function(u_wri_res, step)

# p_res = latest_uh.sub(1).collapse()
# p_res.x.scatter_forward()
# p_recorder.write_function(p_res, step)

# loss_recorder.write_scalar('loss', loss, step)
# ----------------------------
