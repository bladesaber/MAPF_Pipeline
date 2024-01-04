import numpy as np
import ufl
from basix.ufl import element
import dolfinx
from ufl import inner, grad, div
# import matplotlib.pyplot as plt
from functools import partial

from Thirdparty.pyadjoint.pyadjoint import *

from scripts_py.version_9.AD_dolfinx.type_Function import Function
from scripts_py.version_9.AD_dolfinx.type_Mesh import Mesh
from scripts_py.version_9.AD_dolfinx.block_solve import solve
from scripts_py.version_9.AD_dolfinx.block_assemble import assemble
from scripts_py.version_9.AD_dolfinx.type_utils import start_annotation
from scripts_py.version_9.AD_dolfinx.type_DirichletBC import dirichletbc
from scripts_py.version_9.AD_dolfinx.backend_dolfinx import (
    VisUtils, MeshUtils, XDMFRecorder, SolverUtils, TensorBoardRecorder
)

# ------ create xdmf
# msh_file = '/home/admin123456/Desktop/work/topopt_test/fluid_top1/fluid_2D.msh'
# MeshUtils.msh_to_XDMF(
#     name='fluid_2D',
#     msh_file=msh_file,
#     output_file='/home/admin123456/Desktop/work/topopt_test/fluid_top1/fluid_2D.xdmf',
#     dim=2
# )
# -------------------

tape = Tape()
set_working_tape(tape)


def alpha_func(rho, alpha_bar, alpha_under_bar, adj_q):
    return alpha_bar + (alpha_under_bar - alpha_bar) * rho * (1 + adj_q) / (rho + adj_q)


alpha_func = partial(
    alpha_func, alpha_bar=1000., alpha_under_bar=0.01, adj_q=0.1
)

with start_annotation():
    # ------ load domain
    noslip_marker = 20
    input_marker = 18
    output_marker = 19

    domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
        file='/home/admin123456/Desktop/work/topopt_test/fluid_top1/fluid_2D.xdmf',
        mesh_name='fluid_2D',
        cellTag_name='fluid_2D_cells',
        facetTag_name='fluid_2D_facets'
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
    bc_noslip_facets = MeshUtils.extract_boundary_entities(domain, noslip_marker, facet_tags)
    bc_noslip_dofs = MeshUtils.extract_entity_dofs((W0, V), fdim, bc_noslip_facets)
    bc0 = dirichletbc(noslip, bc_noslip_dofs, W0)


    def inflow_velocity_exp(x):
        num = x.shape[1]
        values = np.zeros((tdim, num))
        values[0] = 1.0
        return values


    inflow_velocity = Function(V, name='inflow_velocity')
    inflow_velocity.interpolate(inflow_velocity_exp)
    bc_input_facets = MeshUtils.extract_boundary_entities(domain, input_marker, facet_tags)
    bc_input_dofs = MeshUtils.extract_entity_dofs((W0, V), fdim, bc_input_facets)
    bc1 = dirichletbc(inflow_velocity, bc_input_dofs, W0)

    zero_pressure = Function(Q, name='outflow_pressure')
    bc_output_facets = MeshUtils.extract_boundary_entities(domain, output_marker, facet_tags)
    bc_output_dofs = MeshUtils.extract_entity_dofs((W1, Q), fdim, bc_output_facets)
    bc2 = dirichletbc(zero_pressure, bc_output_dofs, W1)

    bcs = [bc0, bc1, bc2]

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    V_control = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))

    # f = dolfinx.fem.Constant(domain, np.array([0.0, 0.0]))
    f: Function = Function(V)
    # f.x.array[:] = 0.0

    rho: Function = Function(V_control, name='control')

    rho.x.array[:] = 0.5
    F_form = alpha_func(rho) * inner(u, v) * ufl.dx + \
             (inner(grad(u), grad(v)) - div(v) * p - q * div(u)) * ufl.dx - \
             inner(f, v) * ufl.dx

    a_form: ufl.form.Form = ufl.lhs(F_form)
    L_form: ufl.form.Form = ufl.rhs(F_form)

    uh = Function(W, name='state')
    uh = solve(
        uh, a_form, L_form, bcs, domain=domain, is_linear=True,

        petsc_options={'ksp_type': 'cg', 'pc_type': 'lu'},
        linear_system_solver_setting={'ksp_type': 'cg', 'pc_type': 'lu'},

        # petsc_options={'ksp_type': 'cg', 'pc_type': 'ksp'},
        # linear_system_solver_setting={'ksp_type': 'cg', 'pc_type': 'ksp'},
    )

    u, p = ufl.split(uh)
    cost_form = 0.5 * alpha_func(rho) * inner(u, u) * ufl.dx + inner(grad(u), grad(u)) * ufl.dx

    Jhat = assemble(cost_form, domain)
    print(f"[### Test Jhat Cost]: {Jhat}")

control = Control(rho)
opt_problem = ReducedFunctional(Jhat, [control])

# grad = opt_problem.derivative(adj_input=1.0)[0]
# print(np.any(np.isnan(grad.x.array)), np.any(np.isinf(grad.x.array)))

# ------ Recorder Init
rho_recorder = XDMFRecorder(file='/home/admin123456/Desktop/work/topopt_test/fluid_top1/rho_res.xdmf')
rho_recorder.write_mesh(domain)

u_recorder = XDMFRecorder(file='/home/admin123456/Desktop/work/topopt_test/fluid_top1/u_res.xdmf')
u_recorder.write_mesh(domain)

p_recorder = XDMFRecorder(file='/home/admin123456/Desktop/work/topopt_test/fluid_top1/p_res.xdmf')
p_recorder.write_mesh(domain)

loss_recorder = TensorBoardRecorder(log_dir='/home/admin123456/Desktop/work/topopt_test/fluid_top1/log')
# -----------------------

trial_rho: Function = Function(V_control)
trial_rho.assign(rho)
last_loss = opt_problem([trial_rho])
print(f"[### Original Cost]: {last_loss}")
loss_recorder.write_scalar('loss', last_loss, 1)

step = 1
while True:
    step += 1

    grad = opt_problem.derivative(adj_input=1.0)[0]
    grad_np: np.array = grad.x.array
    grad_np = grad_np / np.linalg.norm(grad_np, ord=2)

    thresold = np.percentile(grad_np, q=0.2)
    grad_np = grad_np - thresold
    # grad_np = grad_np - np.mean(grad_np)

    grad.x.array[:] = grad_np
    # VisUtils.show_scalar_res_vtk(grid, 'grad', grad)

    trial_np: np.array = trial_rho.x.array
    # TODO whether we need plus here ???
    trial_np = trial_np + 0.25 * grad_np

    trial_np = np.maximum(np.minimum(trial_np, 1.0), 0.0)

    trial_rho.x.array[:] = trial_np
    loss = opt_problem([trial_rho])

    # VisUtils.show_scalar_res_vtk(grid, 'rho', trial_rho)

    opt_pcg = loss / last_loss
    last_loss = loss
    print(f"[###Step {step}] loss: {loss}, Enhance:{opt_pcg}")

    if step % 3 == 0:
        rho.assign(trial_rho)
        rho_recorder.write_function(rho, step)

        SolverUtils.solve_linear_variational_problem(
            uh, a_form, L_form, bcs=bcs,
            petsc_options={}
        )
        u_res = uh.sub(0).collapse()
        p_res = uh.sub(1).collapse()

        u_res.x.scatter_forward()
        P3 = element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
        u_wri_res = Function(dolfinx.fem.functionspace(domain, P3))
        u_wri_res.interpolate(u_res)
        u_recorder.write_function(u_wri_res, step)

        p_res.x.scatter_forward()
        p_recorder.write_function(p_res, step)

        loss_recorder.write_scalar('loss', loss, step)

    if step > 300:
        break

# -----------------------------
rho.assign(trial_rho)
rho_recorder.write_function(rho, step)

SolverUtils.solve_linear_variational_problem(
    uh, a_form, L_form, bcs=bcs,
    petsc_options={}
)
u_res = uh.sub(0).collapse()
p_res = uh.sub(1).collapse()

u_res.x.scatter_forward()
P3 = element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
u_wri_res = Function(dolfinx.fem.functionspace(domain, P3))
u_wri_res.interpolate(u_res)
u_recorder.write_function(u_wri_res, step)

p_res.x.scatter_forward()
p_recorder.write_function(p_res, step)
# ----------------------------
