"""
IF you use mpi, please tape 'mpirun -n 8 python3 runfile' in console

Not Sure whether it is Correct
"""

import gmsh
import os
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl
import os, shutil
from tqdm import tqdm

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder

fluid_marker = 1
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5


def create_experiment_msh(msh_file):
    gmsh.initialize()

    L = 2.2
    H = 0.41
    c_x = c_y = 0.2
    r = 0.05
    gdim = 2

    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

    inflow, outflow, walls, obstacle = [], [], [], []
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H / 2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H / 2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

    # ------ create mesh
    res_min = r / 3
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    # ------ save mesh
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(msh_file)


# ------ create msh
save_dir = '/home/admin123456/Desktop/work/topopt_exps/NS_simulate_02'
msh_file = os.path.join(save_dir, '/geometry.msh')
xdmf_file = os.path.join(save_dir, 'geometry.xdmf')

# create_experiment_msh(msh_file)
# MeshUtils.msh_to_XDMF(msh_file, xdmf_file, name='fluid', dim=2)

# ------
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=xdmf_file, mesh_name='fluid', cellTag_name='fluid_cells', facetTag_name='fluid_facets'
)

# ------ Init Parameters
t = 0
T = 8  # Final time
dt = 1 / 1600  # Time step size
num_steps = int(T / dt)
k = dolfinx.fem.Constant(domain, PETSc.ScalarType(dt))
mu = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.001))  # Dynamic viscosity
rho = dolfinx.fem.Constant(domain, PETSc.ScalarType(1))  # Density
f = dolfinx.fem.Constant(domain, PETSc.ScalarType((0, 0)))

# ------ Init FUnction Space
v_cg2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
s_cg1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
V = dolfinx.fem.FunctionSpace(domain, v_cg2)
Q = dolfinx.fem.FunctionSpace(domain, s_cg1)

tdim = domain.topology.dim
fdim = tdim - 1


# ------ Define boundary conditions
class InletVelocity(object):
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((tdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41 ** 2)
        return values


# ------ Inlet
u_inlet = dolfinx.fem.Function(V)
inlet_velocity = InletVelocity(t)
u_inlet.interpolate(inlet_velocity)
bcu_inflow = dolfinx.fem.dirichletbc(
    u_inlet, dolfinx.fem.locate_dofs_topological(V, fdim, facet_tags.find(inlet_marker))
)

# ------ Walls
u_nonslip = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType)
bcu_walls = dolfinx.fem.dirichletbc(
    u_nonslip, dolfinx.fem.locate_dofs_topological(V, fdim, facet_tags.find(wall_marker)), V
)

# ------ Obstacle
bcu_obstacle = dolfinx.fem.dirichletbc(
    u_nonslip, dolfinx.fem.locate_dofs_topological(V, fdim, facet_tags.find(obstacle_marker)), V
)

bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
# ------ Outlet
bcp_outlet = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(0), dolfinx.fem.locate_dofs_topological(Q, fdim, facet_tags.find(outlet_marker)), Q
)
bcp = [bcp_outlet]


# ------ Define Variational Formulations
def method_ipcs_1(save_dir=save_dir):
    """
    Incremental Pressure Correction Scheme (IPCS)
    Based on Adams-Bashforth Method
    """

    def epsilon(u):
        return ufl.sym(ufl.nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2 * mu * epsilon(u) - p * ufl.Identity(len(u))

    u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)

    u_n = dolfinx.fem.Function(V, name='u_n')
    p_n = dolfinx.fem.Function(Q, name='p_n')

    U = 0.5 * (u_n + u)
    n = ufl.FacetNormal(domain)

    # Define variational problem for step 1
    F1 = rho * ufl.dot((u - u_n) / k, v) * ufl.dx
    F1 += rho * ufl.dot(ufl.dot(u_n, ufl.nabla_grad(u_n)), v) * ufl.dx
    F1 += ufl.inner(sigma(U, p_n), epsilon(v)) * ufl.dx
    F1 += ufl.dot(p_n * n, v) * ufl.ds - ufl.dot(mu * ufl.nabla_grad(U) * n, v) * ufl.ds
    F1 -= ufl.dot(f, v) * ufl.dx
    a1 = dolfinx.fem.form(ufl.lhs(F1))
    L1 = dolfinx.fem.form(ufl.rhs(F1))

    A1 = petsc.assemble_matrix(a1, bcs=bcu)
    A1.assemble()
    b1 = petsc.create_vector(L1)

    # Define variational problem for step 2
    u_ = dolfinx.fem.Function(V)
    a2 = dolfinx.fem.form(ufl.dot(ufl.nabla_grad(p), ufl.nabla_grad(q)) * ufl.dx)
    L2 = dolfinx.fem.form(
        ufl.dot(ufl.nabla_grad(p_n), ufl.nabla_grad(q)) * ufl.dx - (rho / k) * ufl.div(u_) * q * ufl.dx)
    A2 = petsc.assemble_matrix(a2, bcs=bcp)
    A2.assemble()
    b2 = petsc.create_vector(L2)

    # Define variational problem for step 3
    p_ = dolfinx.fem.Function(Q)
    a3 = dolfinx.fem.form(rho * ufl.dot(u, v) * ufl.dx)
    L3 = dolfinx.fem.form(rho * ufl.dot(u_, v) * ufl.dx - k * ufl.dot(ufl.nabla_grad(p_ - p_n), v) * ufl.dx)
    A3 = petsc.assemble_matrix(a3)
    A3.assemble()
    b3 = petsc.create_vector(L3)

    # ------ init solver
    # Solver for step 1
    solver1 = PETSc.KSP().create(domain.comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    pc1 = solver1.getPC()
    pc1.setType(PETSc.PC.Type.HYPRE)
    pc1.setHYPREType("boomeramg")

    # Solver for step 2
    solver2 = PETSc.KSP().create(domain.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.BCGS)
    pc2 = solver2.getPC()
    pc2.setType(PETSc.PC.Type.HYPRE)
    pc2.setHYPREType("boomeramg")

    # Solver for step 3
    solver3 = PETSc.KSP().create(domain.comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    pc3 = solver3.getPC()
    pc3.setType(PETSc.PC.Type.SOR)

    # ------ solve the question
    module_dir = os.path.join(save_dir, 'method1')
    # if os.path.exists(module_dir):
    #     shutil.rmtree(module_dir)
    # os.mkdir(module_dir)

    t = 0
    vtx_u = VTKRecorder(os.path.join(module_dir, "res_u.pvd"), domain.comm)
    vtx_p = VTKRecorder(os.path.join(module_dir, "res_p.pvd"), domain.comm)
    vtx_u.write_mesh(domain, 0)
    vtx_p.write_mesh(domain, 0)

    for i in tqdm(range(num_steps)):
        # Update current time step
        t += dt

        # Update inlet velocity
        inlet_velocity.t = t
        u_inlet.interpolate(inlet_velocity)

        # Step 1: Tentative velocity step
        A1.zeroEntries()
        petsc.assemble_matrix(A1, a1, bcs=bcu)
        A1.assemble()
        with b1.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b1, L1)
        petsc.apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b1, bcu)
        solver1.solve(b1, u_.vector)
        u_.x.scatter_forward()

        # Step 2: Pressure corrrection step
        with b2.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b2, L2)
        petsc.apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b2, bcp)
        solver2.solve(b2, p_.vector)
        p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with b3.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.vector)
        u_.x.scatter_forward()

        # Update variable with solution form this time step
        u_n.x.array[:] = u_.x.array[:]
        p_n.x.array[:] = p_.x.array[:]

        if i % 100 == 0:
            # Write solutions to file
            vtx_u.write_function(u_n, t)
            vtx_p.write_function(p_n, t)

    # Close xmdf file
    vtx_u.close()
    vtx_p.close()

    b1.destroy()
    b2.destroy()
    b3.destroy()
    solver1.destroy()
    solver2.destroy()
    solver3.destroy()


def method_ipcs_2(save_dir=save_dir):
    """
    Incremental Pressure Correction Scheme (IPCS)
    Based on semi-implicit Adams-Bashforth Method
    """
    u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)

    u_ = dolfinx.fem.Function(V, name='u')
    p_ = dolfinx.fem.Function(Q, name='p')

    u_s = dolfinx.fem.Function(V)
    u_n = dolfinx.fem.Function(V)
    u_n1 = dolfinx.fem.Function(V)
    phi = dolfinx.fem.Function(Q)

    # Define variational problem for step 1
    f = dolfinx.fem.Constant(domain, PETSc.ScalarType((0, 0)))
    F1 = rho / k * ufl.dot(u - u_n, v) * ufl.dx
    F1 += ufl.inner(ufl.dot(1.5 * u_n - 0.5 * u_n1, 0.5 * ufl.nabla_grad(u + u_n)), v) * ufl.dx
    F1 += 0.5 * mu * ufl.inner(ufl.grad(u + u_n), ufl.grad(v)) * ufl.dx - ufl.dot(p_, ufl.div(v)) * ufl.dx
    F1 += ufl.dot(f, v) * ufl.dx
    a1 = dolfinx.fem.form(ufl.lhs(F1))
    L1 = dolfinx.fem.form(ufl.rhs(F1))
    A1 = petsc.create_matrix(a1)
    b1 = petsc.create_vector(L1)

    # Define variational problem for step 2
    a2 = dolfinx.fem.form(ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx)
    L2 = dolfinx.fem.form(-rho / k * ufl.dot(ufl.div(u_s), q) * ufl.dx)
    A2 = petsc.assemble_matrix(a2, bcs=bcp)
    A2.assemble()
    b2 = petsc.create_vector(L2)

    # Define variational problem for step 3
    a3 = dolfinx.fem.form(rho * ufl.dot(u, v) * ufl.dx)
    L3 = dolfinx.fem.form(rho * ufl.dot(u_s, v) * ufl.dx - k * ufl.dot(ufl.nabla_grad(phi), v) * ufl.dx)
    A3 = petsc.assemble_matrix(a3)
    A3.assemble()
    b3 = petsc.create_vector(L3)

    # Solver for step 1
    solver1 = PETSc.KSP().create(domain.comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    pc1 = solver1.getPC()
    pc1.setType(PETSc.PC.Type.JACOBI)

    # Solver for step 2
    solver2 = PETSc.KSP().create(domain.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.MINRES)
    pc2 = solver2.getPC()
    pc2.setType(PETSc.PC.Type.HYPRE)
    pc2.setHYPREType("boomeramg")

    # Solver for step 3
    solver3 = PETSc.KSP().create(domain.comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    pc3 = solver3.getPC()
    pc3.setType(PETSc.PC.Type.SOR)

    # ------ solve the question
    module_dir = os.path.join(save_dir, 'method2')
    # please delete it when use mpirun
    # if os.path.exists(module_dir):
    #     shutil.rmtree(module_dir)
    # os.mkdir(module_dir)

    t = 0
    vtx_u = VTKRecorder(os.path.join(module_dir, "res_u.pvd"), domain.comm)
    vtx_p = VTKRecorder(os.path.join(module_dir, "res_p.pvd"), domain.comm)
    vtx_u.write_mesh(domain, 0)
    vtx_p.write_mesh(domain, 0)

    for i in tqdm(range(num_steps)):
        # Update current time step
        t += dt

        # Update inlet velocity
        inlet_velocity.t = t
        u_inlet.interpolate(inlet_velocity)

        # Step 1: Tentative velocity step
        A1.zeroEntries()
        petsc.assemble_matrix(A1, a1, bcs=bcu)
        A1.assemble()
        with b1.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b1, L1)
        petsc.apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b1, bcu)
        solver1.solve(b1, u_s.vector)
        u_s.x.scatter_forward()

        # Step 2: Pressure corrrection step
        with b2.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b2, L2)
        petsc.apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b2, bcp)
        solver2.solve(b2, phi.vector)
        phi.x.scatter_forward()

        p_.vector.axpy(1, phi.vector)
        p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with b3.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.vector)
        u_.x.scatter_forward()

        if i % 100 == 0:
            # Write solutions to file
            vtx_u.write_function(u_, t)
            vtx_p.write_function(p_, t)

        # Update variable with solution form this time step
        with u_.vector.localForm() as loc_, u_n.vector.localForm() as loc_n, u_n1.vector.localForm() as loc_n1:
            loc_n.copy(loc_n1)
            loc_.copy(loc_n)

    vtx_u.close()
    vtx_p.close()


if __name__ == '__main__':
    # create_experiment_msh(
    #     # msh_file='/home/admin123456/Desktop/work/topopt_exps/NS_simulate_02/geometry.step'
    #     msh_file=msh_file
    # )
    # method_ipcs_1()
    method_ipcs_2()

    pass
