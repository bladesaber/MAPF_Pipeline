from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import pyvista
import dolfinx
import ufl
from ufl import VectorElement, FiniteElement
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc

from scripts_py.version_9.AD_dolfinx.backend_dolfinx import VTKRecorder

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 50, 50)
tdim = mesh.topology.dim
fdim = tdim - 1

t = 0
T = 10
num_steps = 500
dt = T / num_steps

v_cg2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
s_cg1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V = dolfinx.fem.FunctionSpace(mesh, v_cg2)
Q = dolfinx.fem.FunctionSpace(mesh, s_cg1)

u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
v, q = ufl.TestFunction(V), ufl.TestFunction(Q)


# ------ Wall Boundary
def walls(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))


wall_dofs = dolfinx.fem.locate_dofs_geometrical(V, walls)
u_noslip = np.array((0,) * tdim, dtype=PETSc.ScalarType)
bc_noslip = dolfinx.fem.dirichletbc(u_noslip, wall_dofs, V)


# ------ Inflow Boundary
def inflow(x):
    return np.isclose(x[0], 0)


inflow_dofs = dolfinx.fem.locate_dofs_geometrical(Q, inflow)
bc_inflow = dolfinx.fem.dirichletbc(PETSc.ScalarType(8.), inflow_dofs, Q)


# ------ Outflow Boundary
def outflow(x):
    return np.isclose(x[0], 1)


outflow_dofs = dolfinx.fem.locate_dofs_geometrical(Q, outflow)
bc_outflow = dolfinx.fem.dirichletbc(PETSc.ScalarType(0), outflow_dofs, Q)

bcu = [bc_noslip]
bcp = [bc_inflow, bc_outflow]

# ------ Init Form
f = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0)))
k = dolfinx.fem.Constant(mesh, PETSc.ScalarType(dt))
mu = dolfinx.fem.Constant(mesh, PETSc.ScalarType(1))
rho = dolfinx.fem.Constant(mesh, PETSc.ScalarType(1))

u_n = dolfinx.fem.Function(V, name='u_n')
p_n = dolfinx.fem.Function(Q, name='p_n')
U = 0.5 * (u_n + u)
n = ufl.FacetNormal(mesh)


# Define strain-rate tensor
def epsilon(u):
    return ufl.sym(ufl.nabla_grad(u))


# Define stress tensor
def sigma(u, p):
    return 2 * mu * epsilon(u) - p * ufl.Identity(len(u))


# Define variational problem for step 1
F1 = rho * ufl.dot((u - u_n) / k, v) * ufl.dx
F1 += rho * ufl.dot(ufl.dot(u_n, ufl.nabla_grad(u_n)), v) * ufl.dx
F1 += ufl.inner(sigma(U, p_n), epsilon(v)) * ufl.dx
F1 += ufl.dot(p_n * n, v) * ufl.ds - ufl.dot(mu * ufl.nabla_grad(U) * n, v) * ufl.ds
F1 -= ufl.dot(f, v) * ufl.dx
a1 = dolfinx.fem.form(ufl.lhs(F1))
L1 = dolfinx.fem.form(ufl.rhs(F1))

A1 = assemble_matrix(a1, bcs=bcu)
A1.assemble()
b1 = create_vector(L1)

# Define variational problem for step 2
u_ = dolfinx.fem.Function(V)
a2 = dolfinx.fem.form(ufl.dot(ufl.nabla_grad(p), ufl.nabla_grad(q)) * ufl.dx)
L2 = dolfinx.fem.form(ufl.dot(ufl.nabla_grad(p_n), ufl.nabla_grad(q)) * ufl.dx - (rho / k) * ufl.div(u_) * q * ufl.dx)
A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = create_vector(L2)

# Define variational problem for step 3
p_ = dolfinx.fem.Function(Q)
a3 = dolfinx.fem.form(rho * ufl.dot(u, v) * ufl.dx)
L3 = dolfinx.fem.form(rho * ufl.dot(u_, v) * ufl.dx - k * ufl.dot(ufl.nabla_grad(p_ - p_n), v) * ufl.dx)
A3 = assemble_matrix(a3)
A3.assemble()
b3 = create_vector(L3)

# ------ init solver
# Solver for step 1
solver1 = PETSc.KSP().create(mesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")

# Solver for step 2
solver2 = PETSc.KSP().create(mesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.BCGS)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(mesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)

# ------ solve the question
vtx_u = VTKRecorder("/home/admin123456/Desktop/work/topopt_exps/NS_simulate_01/res_u.pvd", mesh.comm)
vtx_p = VTKRecorder("/home/admin123456/Desktop/work/topopt_exps/NS_simulate_01/res_p.pvd", mesh.comm)
vtx_u.write_mesh(mesh, t)
vtx_p.write_mesh(mesh, t)

for i in range(num_steps):
    # Update current time step
    t += dt

    # Step 1: Tentative veolcity step
    with b1.localForm() as loc_1:
        loc_1.set(0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_.vector)
    u_.x.scatter_forward()

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc_2:
        loc_2.set(0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, p_.vector)
    p_.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc_3:
        loc_3.set(0)
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_.vector)
    u_.x.scatter_forward()
    # Update variable with solution form this time step
    u_n.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]

    # Write solutions to file
    if i % 10 == 0:
        vtx_u.write_function(u_n, t)
        vtx_p.write_function(p_n, t)

    print(f"Step:{i} finished")


# Close xmdf file
vtx_u.close()
vtx_p.close()

b1.destroy()
b2.destroy()
b3.destroy()
solver1.destroy()
solver2.destroy()
solver3.destroy()

# # ------ Visulation
# pyvista.start_xvfb()
# topology, cell_types, geometry = vtk_mesh(V)
# values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
# values[:, :len(u_n)] = u_n.x.array.real.reshape((geometry.shape[0], len(u_n)))
#
# # Create a point cloud of glyphs
# function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# function_grid["u"] = values
# glyphs = function_grid.glyph(orient="u", factor=0.2)
#
# # Create a pyvista-grid for the mesh
# grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))
#
# # Create plotter
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, style="wireframe", color="k")
# plotter.add_mesh(glyphs)
# plotter.view_xy()
# plotter.show()
