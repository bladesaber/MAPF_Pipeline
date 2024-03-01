import numpy as np
import ufl
import dolfinx
from ufl import inner, dot
import os
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
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, AssembleUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.recorder_utils import XDMFRecorder, TensorBoardRecorder

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/solid_2D_01/'
model_xdmf_file = os.path.join(proj_dir, 'model.xdmf')


def epsilon(u):
    # return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
    return ufl.sym(ufl.grad(u))


def sigma(u, lambda_par, mu_par):
    return lambda_par * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu_par * epsilon(u)


def psi(u, lambda_par, mu_par):
    return 0.5 * lambda_par * (ufl.tr(epsilon(u)) ** 2) + mu_par * ufl.tr(epsilon(u) * epsilon(u))


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
domain = Mesh(domain)
grid = VisUtils.convert_to_grid(domain)

tdim = domain.topology.dim
fdim = tdim - 1
support_marker = 5
load_marker = 99
volfrac = 0.5


def load_location_marker(x):
    return np.logical_and(np.isclose(x[0], 60.0), x[1] < 2.0)


bc_load_facets = MeshUtils.extract_facet_entities(domain, None, load_location_marker)
bc_load_markers = np.full_like(bc_load_facets, fill_value=load_marker)

# Be Careful. neuuman_facet_tags必须独立出来定义，原因不明
neuuman_facet_tags = MeshUtils.define_meshtag(
    domain,
    indices_list=[bc_load_facets],
    markers_list=[bc_load_markers],
    dim=fdim
)

tape = Tape()
set_working_tape(tape)
with start_annotation():
    V = dolfinx.fem.VectorFunctionSpace(domain, element=('Lagrange', 1))
    Q = dolfinx.fem.FunctionSpace(domain, element=('DG', 0))

    # ---------- Boundary Define
    bcs = []

    bc_support_facets = MeshUtils.extract_facet_entities(domain, facet_tags, support_marker)
    bc_support_dofs = MeshUtils.extract_entity_dofs(V, fdim, bc_support_facets)
    # support_func = dolfinx.fem.Function(V, name='support_%d' % support_marker)
    # bc0 = dolfinx.fem.dirichletbc(support_func, bc_support_dofs)
    bc0 = dirichletbc(np.array([0., 0.]), bc_support_dofs, V)
    bcs.append(bc0)

    # ------ Linear Elastic Equation
    lambda_v = 1.25
    mu = 1.0
    g = 0.045
    gravity_np = np.array([0., 0.])
    load_force_np = np.array([0., -1.0])
    density_poly = 3.0

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = dolfinx.fem.Constant(domain, gravity_np)
    load_force = dolfinx.fem.Constant(domain, load_force_np)
    ds = MeshUtils.define_ds(domain, neuuman_facet_tags)

    density: Function = Function(Q, name='density')
    density.x.array[:] = volfrac

    a_form = inner(density ** density_poly * sigma(u, lambda_v, mu), epsilon(v)) * ufl.dx
    L_form = dot(f, v) * ufl.dx
    L_form += dot(load_force, v) * ds(load_marker)

    uh: Function = Function(V, name='state')
    uh = solve(
        uh, a_form, L_form, bcs,
        domain=domain, is_linear=True,
        tlm_ksp_option={
            'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'
        },
        adj_ksp_option={
            'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'
        },
        forward_ksp_option={
            'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'
        },
        with_debug=True,
        tlm_with_debug=False,
        adj_with_debug=False,
        recompute_with_debug=False,
    )

    # VisUtils.show_vector_res_vtk(grid, uh, dim=tdim, with_wrap=True, factor=0.1)

    cost_form = density ** density_poly * inner(sigma(uh, lambda_v, mu), epsilon(uh)) * ufl.dx
    # cost_form = density ** density_poly * psi(uh, lambda_v, mu)

    Jhat = assemble(cost_form, domain)
    # print(f"[### Test J Cost]: {Jhat}")

control = Control(density)
opt_problem = ReducedFunctional(Jhat, [control])

# ------ define distance matrix
rmin = 2.0

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

# ------
recorder = XDMFRecorder(file=os.path.join(proj_dir, 'density_opt.xdmf'))
recorder.write_mesh(domain)

trial_density: Function = Function(Q)
trial_density.assign(density)
last_loss = opt_problem([trial_density])
print(f"[### Original Cost]: {last_loss}")

volume_orig = AssembleUtils.assemble_scalar(dolfinx.fem.form(
    dolfinx.fem.Constant(domain, 1.0) * ufl.dx
))

step = 1
while True:
    step += 1

    grad_fun = opt_problem.derivative(adj_input=1.0)[0]

    grad_fun_np: np.array = grad_fun.x.array
    # grad_fun_np = grad_fun_np / np.linalg.norm(grad_fun_np, ord=2)

    # grad_fun.x.array[:] = grad_fun_np
    # VisUtils.show_scalar_res_vtk(grid, 'grad', grad_fun, is_point_data=False)

    grad_fun_np = grad_fun_np.reshape((-1, 1))
    density_np: np.array = trial_density.x.array
    density_np = density_np.reshape((-1, 1))

    sensitivity_np: np.matrix = np.divide(
        dist_mat.dot(grad_fun_np * density_np),
        np.multiply(density_np, distSum_mat)
    )
    sensitivity_np = np.asarray(sensitivity_np).reshape(-1)
    density_np = density_np.reshape(-1)

    # grad_fun.x.array[:] = sensitivity_np
    # VisUtils.show_scalar_res_vtk(grid, 'grad', grad_fun, is_point_data=False)

    l1, l2, move = 0, 100000, 0.1
    current_vol = np.inf
    while l2 - l1 > 1e-4:
        l_mid = 0.5 * (l2 + l1)

        update_density = density_np * np.sqrt(-sensitivity_np / volume_orig / l_mid)
        update_density = np.minimum(density_np + move, np.maximum(density_np - move, update_density))
        update_density = np.minimum(1.0, np.maximum(1e-4, update_density))

        tmp_density = dolfinx.fem.Function(Q)
        tmp_density.x.array[:] = update_density
        current_vol = AssembleUtils.assemble_scalar(dolfinx.fem.form(tmp_density * ufl.dx))

        if current_vol > volfrac * volume_orig:
            l1, l2 = l_mid, l2
        else:
            l1, l2 = l1, l_mid

    trial_density.x.array[:] = update_density
    # VisUtils.show_scalar_res_vtk(grid, 'trial_density', trial_density, is_point_data=False)

    loss = opt_problem([trial_density])
    print(f"Step:{step} loss:{loss:.6f}")

    if step % 5 == 0:
        recorder.write_function(trial_density, step)

    if step > 100:
        break
