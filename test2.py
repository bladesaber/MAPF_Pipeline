import matplotlib.pyplot as plt
import numpy as np
import ufl
from basix.ufl import element
from petsc4py import PETSc
from mpi4py import MPI
import dolfinx
import os
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
import pyvista
from ufl import inner, grad, div

from Thirdparty.pyadjoint.pyadjoint import *
from Thirdparty.pyadjoint import pyadjoint

from type_Function import Function
from type_Mesh import Mesh
from block_solve import solve
from block_assemble import assemble
from type_utils import start_annotation
from type_DirichletBC import dirichletbc, DirichletBC
from backend_dolfinx import MeshUtils

"""
1. shape optimization
2. 更好的方式解NS方程，而不是使用联合方程，好的仿真是第一步
3. 更合适的PETSc配置
4. 优化方法不够合理
"""

"""
# ------ create xdmf
# msh_file = '/home/admin123456/Desktop/work/topopt_test/fluid_top1/fluid_2D.msh'
# MeshUtils.msh_to_XDMF(
#     name='fluid_2D',
#     msh_file=msh_file,
#     output_file='/home/admin123456/Desktop/work/topopt_test/fluid_top1/fluid_2D.xdmf',
#     dim=2
# )

# ------ simulate stoke process
noslip_marker = 20
input_marker = 18
output_marker = 19

domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file='/home/admin123456/Desktop/work/topopt_test/fluid_top1/fluid_2D.xdmf',
    mesh_name='fluid_2D',
    cellTag_name='fluid_2D_cells',
    facetTag_name='fluid_2D_facets'
)
tdim = domain.topology.dim
fdim = tdim - 1


P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
TH = ufl.MixedElement([P2, P1])
W = dolfinx.fem.FunctionSpace(domain, TH)
V, V_to_W = W.sub(0).collapse()
Q, Q_to_W = W.sub(1).collapse()

noslip = dolfinx.fem.Function(V)
bc_noslip_facets = MeshUtils.extract_boundary_entities(domain, noslip_marker, facet_tags)
bc_noslip_dofs = MeshUtils.extract_entity_dofs((W.sub(0), V), fdim, bc_noslip_facets)
bc_noslip = dolfinx.fem.dirichletbc(noslip, bc_noslip_dofs, W.sub(0))


def inflow_velocity_exp(x):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[0] = 1.0
    return values


inflow_velocity = dolfinx.fem.Function(V)
inflow_velocity.interpolate(inflow_velocity_exp)
bc_input_facets = MeshUtils.extract_boundary_entities(domain, input_marker, facet_tags)
bc_input_dofs = MeshUtils.extract_entity_dofs((W.sub(0), V), fdim, bc_input_facets)
bc_input = dolfinx.fem.dirichletbc(inflow_velocity, bc_input_dofs, W.sub(0))

zero_pressure = dolfinx.fem.Function(Q)
bc_output_facets = MeshUtils.extract_boundary_entities(domain, output_marker, facet_tags)
bc_output_dofs = MeshUtils.extract_entity_dofs((W.sub(1), Q), fdim, bc_output_facets)
bc_output = dolfinx.fem.dirichletbc(zero_pressure, bc_output_dofs, W.sub(1))

(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

# f = dolfinx.fem.Constant(domain, (0., 0.))
f = dolfinx.fem.Function(V)
f.x.array[:] = np.random.random(f.x.array.size)

# ------------------------------
F_form = (inner(grad(u), grad(v)) - div(v) * p - q * div(u)) * ufl.dx - inner(f, v) * ufl.dx
a = ufl.lhs(F_form)
L = ufl.rhs(F_form)

# from functools import partial
#
# def alpha_func(rho, alpha_bar, alpha_under_bar, adj_q):
#     return alpha_bar + (alpha_under_bar - alpha_bar) * rho * (1 + adj_q) / (rho + adj_q)
#
#
# mu = 1.0
# alpha_under_bar = 2.5 * mu / (100 ** 2)
# alpha_bar = 2.5 * mu / (0.01 ** 2)
# adj_q = 0.01
# alpha_func = partial(alpha_func, alpha_bar=alpha_bar, alpha_under_bar=alpha_under_bar, adj_q=adj_q)
#
# rho: Function = Function(Q)
# rho.x.array[:] = 0.85
# F_form = alpha_func(rho) * inner(u, v) * ufl.dx + \
#          (inner(grad(u), grad(v)) - div(v) * p - q * div(u)) * ufl.dx - \
#          inner(dolfinx.fem.Constant(domain, np.array([0.0, 0.0])), v) * ufl.dx
# a = ufl.lhs(F_form)
# L = ufl.rhs(F_form)
# ----------------------------

# --------------- Fail
# from backend_dolfinx import SolverUtils
# F_form_replace = ufl.replace(F_form, {
#     u: dolfinx.fem.Function(u.ufl_function_space()),
#     p: dolfinx.fem.Function(p.ufl_function_space())
# })
# uh = dolfinx.fem.Function(W)
# run_times, is_converged, uh = SolverUtils.solve_nonlinear_variational_problem(
#     uh, F_form, bcs=[bc_noslip, bc_input, bc_output]
# )
# print(f'Status: {is_converged}')
# ---------------

# uh = dolfinx.fem.Function(W)
# problem = LinearProblem(a, L, bcs=[bc_noslip, bc_input, bc_output], u=uh)
# problem.solve()
# u_h = uh.sub(0).collapse()
# p_h = uh.sub(1).collapse()

problem = LinearProblem(a, L, bcs=[bc_noslip, bc_input, bc_output])
res = problem.solve()
u_h = res.sub(0).collapse()
p_h = res.sub(1).collapse()
# norm_u, norm_p = u_h.x.norm(), p_h.x.norm()

# grid = VisUtils.convert_to_grid(domain)
# grid.point_data["p"] = p_h.x.array.real
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
# plotter.show()

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, '/home/admin123456/Desktop/work/topopt_test/fluid_top1/u_out_1.xdmf', "w") as f:
    u_h.x.scatter_forward()
    P3 = element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    u_h_1 = Function(dolfinx.fem.functionspace(domain, P3))
    u_h_1.interpolate(u_h)
    f.write_mesh(domain)
    f.write_function(u_h_1)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, '/home/admin123456/Desktop/work/topopt_test/fluid_top1/p_out_1.xdmf', "w") as f:
    p_h.x.scatter_forward()
    f.write_mesh(domain)
    f.write_function(p_h)
"""
