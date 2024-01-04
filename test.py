import dolfinx
import matplotlib.pyplot as plt
import pyvista
from mpi4py import MPI
import ufl
import basix
import numpy as np
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

from basix import CellType, ElementFamily, LagrangeVariant


from functools import partial


domain: dolfinx.mesh.Mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD, points=((0.0, 0.0), (1.0, 1.0)), n=(8, 8), cell_type=dolfinx.mesh.CellType.quadrilateral
)

def bc_eq_1(x):
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
        np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
    )


V = dolfinx.fem.functionspace(domain, element=('Lagrange', 1))
# dimension of boundary is always equal to dimension of entity - 1
facets = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim - 1, marker=bc_eq_1)
dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim=domain.topology.dim - 1, entities=facets)
bc = dolfinx.fem.dirichletbc(value=0.0, dofs=dofs, V=V)

# u = ufl.TrialFunction(V)
# v = ufl.TestFunction(V)
u = dolfinx.fem.Function(V)
v = dolfinx.fem.Function(V)

u.x.array[:] = 1.0
v.x.array[:] = u.x.array
u.x.array[:] = 2.0
print(v.x.array)
print(u.x.array)

# x = ufl.SpatialCoordinate(domain)
# f = (x[0] - 0.5) ** 2 + (x[1] - 0.5)**2
# a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
# L = ufl.inner(f, v) * ufl.dx
#
# dc = ufl.TestFunction(u.function_space)
# dform = ufl.derivative(a, u, dc)
# output = dolfinx.fem.assemble_vector(dolfinx.fem.form(dform))
# print(output)
# print(output.array)

# print(len(a.coefficients()))

# problem = LinearProblem(a, L, bcs=[bc])
# uh: dolfinx.fem.function.Function = problem.solve()
# # uh = dolfinx.fem.Function(V)
# # problem = LinearProblem(a, L, bcs=[bc], u=uh)
# # a = problem.solve()
#
# cells, cell_types, geometry = dolfinx.plot.vtk_mesh(domain)
# grid = pyvista.UnstructuredGrid(cells, cell_types, geometry)
# grid.point_data['u'] = uh.x.array.real
# grid.set_active_scalars('u')
#
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True)
# plotter.show()


"""
rho = 1  # density of material
g = 0.016  # acceleration of gravity
lambda_ = 1.25

domain = dolfinx.mesh.create_box(
    MPI.COMM_WORLD, points=[np.array([0., 0., 0.]), np.array([1., 0.2, 0.2])],
    n=[30, 6, 6], cell_type=dolfinx.mesh.CellType.hexahedron
)
V = dolfinx.fem.VectorFunctionSpace(domain, element=('Lagrange', 1))


def clamped_boundary(x):
    return np.isclose(x[0], 0)


facets = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim - 1, marker=clamped_boundary)
dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim=domain.topology.dim - 1, entities=facets)
bc = dolfinx.fem.dirichletbc(value=np.array([0., 0., 0.]), dofs=dofs, V=V)


def epsilon(u):
    # return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
    return ufl.sym(ufl.grad(u))


def sigma(u):
    return lambda_ * ufl.nabla_grad(u) * ufl.Identity(len(u)) + (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dolfinx.fem.Constant(domain, np.array([0., 0., -rho * g]))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
ds = ufl.Measure("ds", domain=domain)
T = dolfinx.fem.Constant(domain, np.array([0., 0., 0.]))
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

plotter = pyvista.Plotter()
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid['u'] = uh.x.array.reshape((geometry.shape[0], 3))
actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = plotter.add_mesh(warped, show_edges=True)
plotter.show_axes()
plotter.show()

# with dolfinx.io.XDMFFile(domain.comm, "/home/admin123456/Desktop/work/test/deformation.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     uh.name = "Deformation"
#     xdmf.write_function(uh)
"""

"""
domain = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD, points=((0.0, 0.0), (1.0, 1.0)), n=(8, 8), cell_type=dolfinx.mesh.CellType.quadrilateral
)

V = dolfinx.fem.functionspace(domain, element=('Lagrange', 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx


def bc_dirichlet(x):
    return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))


# facets = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim - 1, marker=bc_dirichlet)
# dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim=domain.topology.dim - 1, entities=facets)
dofs = dolfinx.fem.locate_dofs_geometrical(V, bc_dirichlet)
# u_D = dolfinx.fem.Function(V)
# u_D.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
# bc = dolfinx.fem.dirichletbc(value=u_D, dofs=dofs)
bc = dolfinx.fem.dirichletbc(value=0.0, dofs=dofs, V=V)

f = dolfinx.fem.Constant(domain, c=-6.0)


def bc_neuuman(x):
    return np.isclose(x[1], 1.0)


facet_indices = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim - 1, bc_neuuman)
facet_marker = np.full_like(facet_indices, 1)
facet_indices = facet_indices.astype(np.int32)
facet_marker = facet_marker.astype(np.int32)
facets_tag = dolfinx.mesh.meshtags(domain, domain.topology.dim - 1, facet_indices, facet_marker)

# # ------ Debug
# domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
# with dolfinx.io.XDMFFile(domain.comm, '/home/admin123456/Desktop/work/test/facet_tags.xdmf', 'w') as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_meshtags(facets_tag, domain.geometry)
# # ------

ds = ufl.Measure("ds", domain=domain, subdomain_data=facets_tag)
# x = ufl.SpatialCoordinate(domain)
# n = ufl.FacetNormal(domain)
# g = ufl.dot(n, ufl.grad(lambda x: .....))
g = -4.0

L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds

problem = LinearProblem(a, L, bcs=[bc])
uh = problem.solve()

cells, cell_types, geometry = dolfinx.plot.vtk_mesh(domain)
grid = pyvista.UnstructuredGrid(cells, cell_types, geometry)
grid.point_data['u'] = uh.x.array.real
grid.set_active_scalars('u')
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.show()

"""

"""
t = 0
T = 2.0
num_steps = 30
dt = T / num_steps
alpha = 3
beta = 1.2

# domain = dolfinx.mesh.create_rectangle(
#     MPI.COMM_WORLD, [np.array([-2, -2]), np.array([2, 2])], [50, 50], dolfinx.mesh.CellType.triangle
# )
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 25, 25, dolfinx.mesh.CellType.triangle)
V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))


class exact_solution():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return 1. + x[0]**2 + self.alpha*x[1]**2 + self.beta * self.t


u_exact = exact_solution(alpha, beta, t)

u_D = dolfinx.fem.Function(V)
u_D.interpolate(u_exact)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
bc = dolfinx.fem.dirichletbc(u_D, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))

u_n = dolfinx.fem.Function(V)
u_n.interpolate(u_exact)
f = dolfinx.fem.Constant(domain, beta - 2 - 2 * alpha)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx
a_form = dolfinx.fem.form(ufl.lhs(F))
L_form = dolfinx.fem.form(ufl.rhs(F))

# A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=[bc])
# A.assemble()
A = dolfinx.fem.petsc.create_matrix(a_form)
dolfinx.fem.petsc.assemble_matrix_mat(A, a_form, bcs=[bc])
A.assemble()

b = dolfinx.fem.petsc.create_vector(L_form)
uh = dolfinx.fem.Function(V)

solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.setOperators(A)

for i in range(num_steps):
    u_exact.t += dt
    u_D.interpolate(u_exact)

    with b.localForm() as loc_b:
        loc_b.set(0)
    dolfinx.fem.petsc.assemble_vector(b, L_form)
    # dolfinx.fem.assemble_vector(b, L_form)
    # b = dolfinx.fem.petsc.assemble_vector(L_form)

    dolfinx.fem.apply_lifting(b, [a_form], [[bc]])
    # b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, [bc])

    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    u_n.x.array[:] = uh.x.array

V_ex = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 2))
u_ex = dolfinx.fem.Function(V_ex)
u_ex.interpolate(u_exact)

error_L2 = np.sqrt(dolfinx.fem.assemble_scalar(
    dolfinx.fem.form((uh - u_ex)**2 * ufl.dx)
))
print(error_L2)

# ----------------------------------------------------
# def init_condition(x, a=5):
#     return np.exp(-a * (x[0]**2 + x[1]**2))
#
#
# u_n = dolfinx.fem.Function(V)
# u_n.name = "u_n"
# u_n.interpolate(init_condition)

# fdim = domain.topology.dim - 1
# # boundary_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], T, dtype=bool))
# domain.topology.create_connectivity(fdim, domain.topology.dim)
# boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
# bc = dolfinx.fem.dirichletbc(PETSc.ScalarType(0), dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# xdmf = dolfinx.io.XDMFFile(domain.comm, "/home/admin123456/Desktop/temptory/test_1/diffusion.xdmf", "w")
# xdmf.write_mesh(domain)
#
# uh = dolfinx.fem.Function(V)
# uh.name = 'uh'
# uh.interpolate(init_condition)
# xdmf.write_function(uh, t)
#
# u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
# f = dolfinx.fem.Constant(domain, PETSc.ScalarType(0))
# a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
# L = (u_n + dt * f) * v * ufl.dx
#
# a_form = dolfinx.fem.form(a)
# L_form = dolfinx.fem.form(L)
#
# A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=[bc])
# A.assemble()
# b = dolfinx.fem.petsc.create_vector(L_form)
#
# solver = PETSc.KSP().create(domain.comm)
# solver.setOperators(A)
# solver.setType(PETSc.KSP.Type.PREONLY)
# solver.getPC().setType(PETSc.PC.Type.LU)
#
# for i in range(num_steps):
#     t += dt
#
#     with b.localForm() as loc_b:
#         loc_b.set(0)
#     dolfinx.fem.petsc.assemble_vector(b, L_form)
#
#     dolfinx.fem.apply_lifting(b, [a_form], [[bc]])
#     b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
#     dolfinx.fem.set_bc(b, [bc])
#
#     solver.solve(b, uh.vector)
#     uh.x.scatter_forward()
#
#     u_n.x.array[:] = uh.x.array
#
#     xdmf.write_function(uh, t)
#
# xdmf.close()
"""

"""
from dolfinx.nls.petsc import NewtonSolver


def q(u):
    return 1 + u ** 2


domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
x = ufl.SpatialCoordinate(domain)
u_ufl = 1 + x[0] + 2 * x[1]
f = -ufl.div(q(u_ufl) * ufl.grad(u_ufl))

V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))


def u_exact(x):
    return eval(str(u_ufl))


u_D = dolfinx.fem.Function(V)
u_D.interpolate(u_exact)

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
bc = dolfinx.fem.dirichletbc(u_D, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))

uh = dolfinx.fem.Function(V)
v = ufl.TestFunction(V)
F = q(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

problem = dolfinx.fem.petsc.NonlinearProblem(F, uh, bcs=[bc])

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = 'incremental'
solver.rtol = 1e-6
# solver.report = False

# ksp = solver.krylov_solver
# opts = PETSc.Options()
# option_prefix = ksp.getOptionsPrefix()
# opts[f"{option_prefix}ksp_type"] = "cg"
# opts[f"{option_prefix}pc_type"] = "gamg"
# opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# ksp.setFromOptions()

# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
n, converged = solver.solve(uh)
assert (converged)
print(f"Number of interations: {n:d}", converged)
"""


