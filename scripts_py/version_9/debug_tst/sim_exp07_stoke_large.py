import dolfinx
import numpy as np
import os
import ufl
from ufl import inner, grad, div
from basix.ufl import element
from petsc4py import PETSc

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.equation_solver import LinearProblemSolver
from scripts_py.version_9.dolfinx_Grad.recorder_utils import XDMFRecorder, VTKRecorder
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils

"""
1. shape optimization
2. 更好的方式解NS方程，而不是使用联合方程，好的仿真是第一步
3. 更合适的PETSc配置
4. 优化方法不够合理
5. 使用OpenFoam生成网格试下??
"""

# # ------ debug
# VisUtils.show_vtu('/home/admin123456/Desktop/work/topopt_exps/Stoke_tst/model_07_2D_u_p0_000000.vtu', scale=0.03)
# raise ValueError
# # ------

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/Stoke_simulate_02'
model_name = 'model'
model_xdmf_file = f"{model_name}.xdmf"
dim = 3

input_marker = 238
output1_marker, output2_marker, output3_marker, output4_marker = 239, 240, 241, 242
boundary_markers = 9999

petsc_options = {'ksp_type': 'bcgsl', 'pc_type': 'jacobi'}
# petsc_options = {'ksp_type': 'preonly', 'pc_type': 'ksp'}
record_mat = True
record_dir = os.path.join(proj_dir, 'debug')

nu = 1. / 400.

# ------ create xdmf
MeshUtils.msh_to_XDMF(
    name='model',
    msh_file=os.path.join(proj_dir, f"{model_name}.msh"),
    output_file=os.path.join(proj_dir, model_xdmf_file),
    dim=dim
)
# -------------------

domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=os.path.join(proj_dir, model_xdmf_file),
    mesh_name='model',
    cellTag_name='model_cells',
    facetTag_name='model_facets'
)

tdim = domain.topology.dim
fdim = tdim - 1

facet_tags = MeshUtils.extract_inverse_boundary_entities(
    domain, facet_tags, other_marker=boundary_markers, dim=fdim
)

P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
TH = ufl.MixedElement([P2, P1])
W = dolfinx.fem.FunctionSpace(domain, TH)
W0, W1 = W.sub(0), W.sub(1)
V, V_to_W = W0.collapse()
Q, Q_to_W = W1.collapse()

# ---------- Boundary Define
bcs = []

noslip = dolfinx.fem.Function(V, name='noslip_%d' % boundary_markers)
bc_noslip_facets = MeshUtils.extract_facet_entities(domain, facet_tags, boundary_markers)
bc_noslip_dofs = MeshUtils.extract_entity_dofs((W0, V), fdim, bc_noslip_facets)
bc0 = dolfinx.fem.dirichletbc(noslip, bc_noslip_dofs, W0)
bcs.append(bc0)


def inflow_velocity_exp(x):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[2] = 10.0
    return values


inflow_velocity = dolfinx.fem.Function(V, name='inflow_velocity')
inflow_velocity.interpolate(inflow_velocity_exp)
bc_input_facets = MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
bc_input_dofs = MeshUtils.extract_entity_dofs((W0, V), fdim, bc_input_facets)
bc1 = dolfinx.fem.dirichletbc(inflow_velocity, bc_input_dofs, W0)
bcs.append(bc1)

# inflow_pressure = dolfinx.fem.Function(Q, name='inflow_pressure')
# inflow_pressure.x.array[:] = 1.0
# bc_input_facets = MeshUtils.extract_boundary_entities(domain, input_marker, facet_tags)
# bc_input_dofs = MeshUtils.extract_entity_dofs((W1, Q), fdim, bc_input_facets)
# bc1 = dolfinx.fem.dirichletbc(inflow_pressure, bc_input_dofs, W1)
# bcs.append(bc1)

# ------ Output Flow 1
bc_out1_facets = MeshUtils.extract_facet_entities(domain, facet_tags, output1_marker)

# outflow1_velocity = dolfinx.fem.Function(V, name='outflow1_velocity')
# bc_out1_dofs = MeshUtils.extract_entity_dofs((W0, V), fdim, bc_out1_facets)
# bc2 = dolfinx.fem.dirichletbc(outflow1_velocity, bc_out1_dofs, W0)

out1_pressure = dolfinx.fem.Function(Q, name='outflow1_pressure')
bc_out1_dofs = MeshUtils.extract_entity_dofs((W1, Q), fdim, bc_out1_facets)
bc2 = dolfinx.fem.dirichletbc(out1_pressure, bc_out1_dofs, W1)

bcs.append(bc2)

# ------ Output Flow 2
bc_out2_facets = MeshUtils.extract_facet_entities(domain, facet_tags, output2_marker)

outflow2_velocity = dolfinx.fem.Function(V, name='outflow2_velocity')
bc_out2_dofs = MeshUtils.extract_entity_dofs((W0, V), fdim, bc_out2_facets)
bc3 = dolfinx.fem.dirichletbc(outflow2_velocity, bc_out2_dofs, W0)

# out2_pressure = dolfinx.fem.Function(Q, name='outflow2_pressure')
# bc_out2_dofs = MeshUtils.extract_entity_dofs((W1, Q), fdim, bc_out2_facets)
# bc3 = dolfinx.fem.dirichletbc(out2_pressure, bc_out2_dofs, W1)

bcs.append(bc3)

# ------ Output Flow 3
out3_pressure = dolfinx.fem.Function(Q, name='outflow3_pressure')
bc_out3_facets = MeshUtils.extract_facet_entities(domain, facet_tags, output3_marker)
bc_out3_dofs = MeshUtils.extract_entity_dofs((W1, Q), fdim, bc_out3_facets)
bc4 = dolfinx.fem.dirichletbc(out3_pressure, bc_out3_dofs, W1)
bcs.append(bc4)

# ------ Output Flow 4
bc_out4_facets = MeshUtils.extract_facet_entities(domain, facet_tags, output4_marker)

outflow4_velocity = dolfinx.fem.Function(V, name='outflow4_velocity')
bc_out4_dofs = MeshUtils.extract_entity_dofs((W0, V), fdim, bc_out4_facets)
bc5 = dolfinx.fem.dirichletbc(outflow4_velocity, bc_out4_dofs, W0)

# out4_pressure = dolfinx.fem.Function(Q, name='outflow4_pressure')
# bc_out4_dofs = MeshUtils.extract_entity_dofs((W1, Q), fdim, bc_out4_facets)
# bc5 = dolfinx.fem.dirichletbc(out4_pressure, bc_out4_dofs, W1)

bcs.append(bc5)
# ----------

(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = dolfinx.fem.Constant(domain, np.zeros(tdim))

# ------ Stokes Equation
F_form = nu * inner(grad(u), grad(v)) * ufl.dx - p * div(v) * ufl.dx + div(u) * q * ufl.dx - inner(f, v) * ufl.dx
a_form: ufl.form.Form = ufl.lhs(F_form)
L_form: ufl.form.Form = ufl.rhs(F_form)

uh = dolfinx.fem.Function(W, name='state')
res_dict = LinearProblemSolver.solve_by_petsc_form(
    domain.comm, uh, a_form, L_form, bcs,
    ksp_option=petsc_options, with_debug=False,
    # record_mat_dir=record_dir
)
uh = res_dict['res']

u_res = uh.sub(0).collapse()
p_res = uh.sub(1).collapse()

grid = VisUtils.convert_to_grid(domain)
VisUtils.show_arrow_res_vtk(grid, u_res, V, scale=0.1)

# ---------- Record Simulate Result
# ------ XDMF Recorder
u_recorder = XDMFRecorder(os.path.join(proj_dir, f"{model_name}_u.xdmf"))
u_recorder.write_mesh(domain)
u_res.x.scatter_forward()
P3 = element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
u_wri_res = dolfinx.fem.Function(dolfinx.fem.functionspace(domain, P3))
u_wri_res.interpolate(u_res)
u_recorder.write_function(u_wri_res, 0)

p_recorder = XDMFRecorder(os.path.join(proj_dir, f"{model_name}_p.xdmf"))
p_recorder.write_mesh(domain)
p_res.x.scatter_forward()
p_recorder.write_function(p_res, 0)

# ------ VTK Recorder
# vtx_u = VTKRecorder(os.path.join(proj_dir, f"{model_name}_u.pvd"), domain.comm)
# vtx_p = VTKRecorder(os.path.join(proj_dir, f"{model_name}_p.pvd"), domain.comm)
# vtx_u.write_function(u_res, 0)
# vtx_p.write_function(p_res, 0)

