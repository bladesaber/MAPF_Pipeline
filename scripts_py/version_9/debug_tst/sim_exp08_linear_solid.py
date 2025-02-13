import dolfinx
import ufl
import os
import numpy as np

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.equation_solver import LinearProblemSolver
from scripts_py.version_9.dolfinx_Grad.recorder_utils import XDMFRecorder
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/solid_2D_01/'
model_xdmf_file = os.path.join(proj_dir, 'model.xdmf')


def epsilon(u):
    # return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
    return ufl.sym(ufl.grad(u))


def sigma(u, lambda_par, mu_par):
    return lambda_par * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu_par * epsilon(u)


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
support_marker = 5
load_marker = 99


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

V = dolfinx.fem.VectorFunctionSpace(domain, element=('Lagrange', 1))

# ---------- Boundary Define
bcs = []

bc_support_facets = MeshUtils.extract_facet_entities(domain, facet_tags, support_marker)
bc_support_dofs = MeshUtils.extract_entity_dofs(V, fdim, bc_support_facets)
support_func = dolfinx.fem.Function(V, name='support_%d' % support_marker)
bc0 = dolfinx.fem.dirichletbc(support_func, bc_support_dofs)
bcs.append(bc0)

# ------ Linear Elastic Equation
lambda_v = 1.25
mu = 1.0
g = 0.045
gravity_np = np.array([0., 0.])
load_force_np = np.array([0., -1.0])

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dolfinx.fem.Constant(domain, gravity_np)
load_force = dolfinx.fem.Constant(domain, load_force_np)
ds = MeshUtils.define_ds(domain, neuuman_facet_tags)

a_form = ufl.inner(sigma(u, lambda_v, mu), epsilon(v)) * ufl.dx
L_form = ufl.dot(f, v) * ufl.dx
L_form += ufl.dot(load_force, v) * ds(load_marker)

uh = dolfinx.fem.Function(V)
res_dict = LinearProblemSolver.solve_by_petsc_form(
    domain.comm, uh, a_form, L_form, bcs,
    ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu'},
    with_debug=True
)
print(f"max_error:{res_dict['max_error']:.6f}, cost_time:{res_dict['cost_time']:.6f}")
uh: dolfinx.fem.Function = res_dict['res']

VisUtils.show_vector_res_vtk(grid, uh, dim=tdim, with_wrap=True, factor=0.1)
# VisUtils.show_arrow_res_vtk(grid, uh, V, scale=0.05)

# ------
# recorder = XDMFRecorder(os.path.join(proj_dir, f"model_u.xdmf"))
# recorder.write_mesh(domain)
# recorder.write_function(uh, step=0)
