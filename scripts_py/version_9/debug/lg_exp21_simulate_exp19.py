import os
import shutil
from functools import partial
import dolfinx
import ufl
from ufl import inner, dot, grad, div
import numpy as np

from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.recorder_utils import XDMFRecorder, VTKRecorder
from scripts_py.version_9.dolfinx_Grad.equation_solver import LinearProblemSolver, NonLinearProblemSolver
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import AssembleUtils

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/fluid_shape4'
model_xdmf = os.path.join(proj_dir, 'last_model.xdmf')

input_marker = 1
output_markers = [5, 6, 7]
bry_markers = [2, 3, 4]

Re = 100
nuValue = 1. / Re


def inflow_velocity_exp(x, tdim):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[0] = 12.0 * (0.0 - x[1]) * (x[1] + 1.0)
    return values


def method1_run(
        domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags,
        proj_dir, nstoke_ksp_option,
        **kwargs
):
    tdim = domain.topology.dim
    fdim = tdim - 1

    W = dolfinx.fem.FunctionSpace(
        domain, ufl.MixedElement([
            ufl.VectorElement("Lagrange", domain.ufl_cell(), 2), ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
        ])
    )
    W0, W1 = W.sub(0), W.sub(1)
    V, _ = W0.collapse()
    Q, _ = W1.collapse()
    V_mapping_space = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
    Q_mapping_space = dolfinx.fem.FunctionSpace(domain, ("CG", 1))

    # -----------------------------------------------
    bcs = []
    for marker in bry_markers:
        bc_value = dolfinx.fem.Function(V, name=f"bry_u_{marker}")
        bc_dofs = MeshUtils.extract_entity_dofs(
            (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
        )
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, W0)
        bcs.append(bc)

    bc_in1_value = dolfinx.fem.Function(V, name='inflow_u')
    bc_in1_value.interpolate(partial(inflow_velocity_exp, tdim=tdim))
    bc_in1_dofs = MeshUtils.extract_entity_dofs(
        (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
    )
    bc_in1 = dolfinx.fem.dirichletbc(bc_in1_value, bc_in1_dofs, W0)
    bcs.append(bc_in1)

    for marker in output_markers:
        bc_out_value = dolfinx.fem.Function(Q, name=f"outflow_p_{marker}")
        bc_out_dofs = MeshUtils.extract_entity_dofs(
            (W1, Q), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
        )
        bc_out = dolfinx.fem.dirichletbc(bc_out_value, bc_out_dofs, W1)
        bcs.append(bc_out)

    # -----------------------------------------------
    nu = dolfinx.fem.Constant(domain, nuValue)

    # u1, p1 = ufl.split(ufl.TrialFunction(W))
    # v1, q1 = ufl.split(ufl.TestFunction(W))
    # f1 = dolfinx.fem.Constant(domain, np.zeros(tdim))
    # stoke_form = (
    #         inner(grad(u1), grad(v1)) * ufl.dx
    #         - p1 * div(v1) * ufl.dx
    #         - q1 * div(u1) * ufl.dx
    #         - inner(f1, v1) * ufl.dx
    # )

    up2 = dolfinx.fem.Function(W, name='fine_state')
    u2, p2 = ufl.split(up2)
    v2, q2 = ufl.split(ufl.TestFunction(W))
    f2 = dolfinx.fem.Constant(domain, np.zeros(tdim))
    navier_stoke_lhs = (
            nu * inner(grad(u2), grad(v2)) * ufl.dx
            + inner(grad(u2) * u2, v2) * ufl.dx
            - inner(p2, div(v2)) * ufl.dx
            + inner(div(u2), q2) * ufl.dx
            - inner(f2, v2) * ufl.dx
    )

    ds = MeshUtils.define_ds(domain, facet_tags)
    n_vec = MeshUtils.define_facet_norm(domain)
    outflow_data = {}
    for marker in output_markers:
        outflow_data[f"marker_{marker}"] = {
            'form': dolfinx.fem.form(ufl.dot(u2, n_vec) * ds(marker)),
            'value': 0.0,
        }

    # --------------------------------------------------------
    jacobi_form = ufl.derivative(navier_stoke_lhs, up2, ufl.TrialFunction(up2.function_space))
    res_dict = NonLinearProblemSolver.solve_by_petsc(
        F_form=navier_stoke_lhs, uh=up2, jacobi_form=jacobi_form, bcs=bcs,
        comm=domain.comm, ksp_option=nstoke_ksp_option, with_debug=True,
        **kwargs
    )
    print(f"[DEBUG Navier Stoke]: max_error:{res_dict['max_error']:.8f}")

    for key in outflow_data.keys():
        outflow_data[key]['value'] = AssembleUtils.assemble_scalar(outflow_data[key]['form'])
        print(f"{key}: {outflow_data[key]['value']}")

    record_dir = os.path.join(proj_dir, 'simulate_method1')
    if os.path.exists(record_dir):
        shutil.rmtree(record_dir)
    os.mkdir(record_dir)

    u_recorder = XDMFRecorder(os.path.join(record_dir, "velocity.xdmf"))
    u_recorder.write_mesh(domain)
    up2.sub(0).collapse().x.scatter_forward()
    u_wri_res = dolfinx.fem.Function(V_mapping_space)
    u_wri_res.interpolate(up2.sub(0).collapse())
    u_recorder.write_function(u_wri_res, 0)

    p_recorder = XDMFRecorder(os.path.join(record_dir, f"pressure.xdmf"))
    p_recorder.write_mesh(domain)
    up2.sub(1).collapse().x.scatter_forward()
    p_recorder.write_function(up2.sub(1).collapse(), 0)


def method2_run(
        domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags,
        proj_dir, **kwargs
):
    tdim = domain.topology.dim
    fdim = tdim - 1

    t = 0
    T = 10
    num_steps = 500
    dt = T / num_steps

    v_cg2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
    s_cg1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    V = dolfinx.fem.FunctionSpace(domain, v_cg2)
    Q = dolfinx.fem.FunctionSpace(domain, s_cg1)

    # ---------------------
    bcu = []
    bcp = []

    for marker in bry_markers:
        bc_value = dolfinx.fem.Function(V, name=f"bry_u_{marker}")
        bc_dofs = MeshUtils.extract_entity_dofs(V, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker))
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs)
        bcu.append(bc)

    bc_in1_value = dolfinx.fem.Function(V, name='inflow_u')
    bc_in1_value.interpolate(partial(inflow_velocity_exp, tdim=tdim))
    bc_in1_dofs = MeshUtils.extract_entity_dofs(
        V, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
    )
    bc_in1 = dolfinx.fem.dirichletbc(bc_in1_value, bc_in1_dofs)
    bcu.append(bc_in1)

    for marker in output_markers:
        bc_out_value = dolfinx.fem.Function(Q, name=f"outflow_p_{marker}")
        bc_out_dofs = MeshUtils.extract_entity_dofs(
            Q, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
        )
        bc_out = dolfinx.fem.dirichletbc(bc_out_value, bc_out_dofs)
        bcp.append(bc_out)

    # --------------------------------------------------
    u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)

    f = dolfinx.fem.Constant(domain, PETSc.ScalarType((0, 0)))
    k = dolfinx.fem.Constant(domain, PETSc.ScalarType(dt))
    mu = dolfinx.fem.Constant(domain, PETSc.ScalarType(1))
    rho = dolfinx.fem.Constant(domain, PETSc.ScalarType(1))

    u_n = dolfinx.fem.Function(V, name='u_n')
    p_n = dolfinx.fem.Function(Q, name='p_n')
    U = 0.5 * (u_n + u)
    n = ufl.FacetNormal(domain)
    ds = MeshUtils.define_ds(domain, facet_tags)

    outflow_data = {}
    for marker in output_markers:
        outflow_data[f"marker_{marker}"] = {
            'form': dolfinx.fem.form(ufl.dot(u_n, n) * ds(marker)),
            'value': 0.0,
        }

    def epsilon(u):
        return ufl.sym(ufl.nabla_grad(u))

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
    L2 = dolfinx.fem.form(
        ufl.dot(ufl.nabla_grad(p_n), ufl.nabla_grad(q)) * ufl.dx - (rho / k) * ufl.div(u_) * q * ufl.dx)
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
    record_dir = os.path.join(proj_dir, 'simulate_method2')
    if os.path.exists(record_dir):
        shutil.rmtree(record_dir)
    os.mkdir(record_dir)

    vtx_u = VTKRecorder(os.path.join(record_dir, "velocity.pvd"), domain.comm)
    vtx_p = VTKRecorder(os.path.join(record_dir, "pressure.pvd"), domain.comm)

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

    for key in outflow_data.keys():
        outflow_data[key]['value'] = AssembleUtils.assemble_scalar(outflow_data[key]['form'])
        print(f"{key}: {outflow_data[key]['value']}")

    # Close xmdf file
    vtx_u.close()
    vtx_p.close()

    b1.destroy()
    b2.destroy()
    b3.destroy()
    solver1.destroy()
    solver2.destroy()
    solver3.destroy()


domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=model_xdmf, mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)

method1_run(
    domain=domain, cell_tags=cell_tags, facet_tags=facet_tags,
    proj_dir='/home/admin123456/Desktop/work/topopt_exps/fluid_shape4',
    nstoke_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    # A_assemble_method='Identity_row'
)

# method2_run(
#     domain=domain, cell_tags=cell_tags, facet_tags=facet_tags,
#     proj_dir='/home/admin123456/Desktop/work/topopt_exps/fluid_shape4',
# )
