"""
reference: Automated shape differentation in the Unified Form Language
"""

import numpy as np
import ufl
import dolfinx
# import matplotlib.pyplot as plt

from Thirdparty.pyadjoint.pyadjoint import *

from scripts_py.version_9.dolfinx_Grad.autoGrad_method.type_Function import Function
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.type_Mesh import Mesh
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.block_solve import solve
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.block_assemble import assemble
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.type_utils import start_annotation
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.type_DirichletBC import dirichletbc
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.block_ALE import move
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder, TensorBoardRecorder
from scripts_py.version_9.dolfinx_Grad.equation_solver import LinearProblemSolver

"""
我认为这个方法需要不断的re-mesh
"""

# # ------ create xdmf
# msh_file = '/home/admin123456/Desktop/work/topopt_exps/fluid_shape1/pipe_1.msh'
# MeshUtils.msh_to_XDMF(
#     name='fluid_2D',
#     msh_file=msh_file,
#     output_file='/home/admin123456/Desktop/work/topopt_exps/fluid_shape1/fluid_2D.xdmf',
#     dim=2
# )
# # -------------------

# tape = RecordTape()
tape = Tape()
set_working_tape(tape)

with start_annotation():
    input_marker = 13
    output_marker = 14
    wallFixed_markers = 15
    wallFree_markers = 16

    domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
        file='/home/admin123456/Desktop/work/topopt_exps/fluid_shape1/fluid_2D.xdmf',
        mesh_name='fluid_2D',
        cellTag_name='fluid_2D_cells',
        facetTag_name='fluid_2D_facets'
    )
    domain: Mesh = Mesh(domain)
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

    coordinate_space = domain.ufl_domain().ufl_coordinate_element()
    S = dolfinx.fem.FunctionSpace(domain, coordinate_space)
    # S = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))  # replace

    # ---------- Boundary Define
    bcs = []

    for noslip_marker in [wallFixed_markers, wallFree_markers]:
        noslip = dolfinx.fem.Function(V, name='noslip_%d' % noslip_marker)
        bc_noslip_facets = MeshUtils.extract_facet_entities(domain, facet_tags, noslip_marker)
        bc_noslip_dofs = MeshUtils.extract_entity_dofs((W0, V), fdim, bc_noslip_facets)
        bc0 = dirichletbc(noslip, bc_noslip_dofs, W0)
        bcs.append(bc0)


    def inflow_velocity_exp(x):
        num = x.shape[1]
        values = np.zeros((tdim, num))
        values[0] = 6 * (1 - x[1]) * x[1]
        return values


    inflow_velocity = Function(V, name='inflow_velocity')
    inflow_velocity.interpolate(inflow_velocity_exp)
    bc_input_facets = MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
    bc_input_dofs = MeshUtils.extract_entity_dofs((W0, V), fdim, bc_input_facets)
    bc1 = dirichletbc(inflow_velocity, bc_input_dofs, W0)
    bcs.append(bc1)

    zero_pressure = Function(Q, name='outflow_pressure')
    bc_output_facets = MeshUtils.extract_facet_entities(domain, facet_tags, output_marker)
    bc_output_dofs = MeshUtils.extract_entity_dofs((W1, Q), fdim, bc_output_facets)
    bc2 = dirichletbc(zero_pressure, bc_output_dofs, W1)
    bcs.append(bc2)
    # ----------

    displacement: Function = Function(S, name='control')
    move(domain, displacement)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f = dolfinx.fem.Constant(domain, np.array([0.0, 0.0]))

    nu = 1. / 400.
    # ------ Stokes Equation
    F_form = nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - \
             p * ufl.div(v) * ufl.dx + ufl.div(u) * q * ufl.dx - \
             ufl.inner(f, v) * ufl.dx
    a_form: ufl.form.Form = ufl.lhs(F_form)
    L_form: ufl.form.Form = ufl.rhs(F_form)

    uh = Function(W, name='state')
    uh = solve(
        uh, a_form, L_form, bcs,
        domain=domain, is_linear=True,
        tlm_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu'},
        adj_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        forward_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        with_debug=True,
    )
    u, p = ufl.split(uh)

    volume_form = dolfinx.fem.Constant(domain, 1.0) * ufl.dx
    with stop_annotating():
        constraint_volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(volume_form)) * 0.95

    cost_form = 1. / 400. * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
    J = assemble(cost_form, domain)
    constraint_J = 0.1 * (assemble(volume_form, domain) - constraint_volume) ** 2
    J += constraint_J

control = Control(displacement)
# opt_problem = RecordFunctional(J, [control])
opt_problem = ReducedFunctional(J, [control])

# grad = opt_problem.derivative(adj_input=1.0)[0]
# print(np.any(np.isnan(grad.x.array)), np.any(np.isinf(grad.x.array)))
# print(grad.x.array)

# ------ init recorder
u_recorder = VTKRecorder(file='/home/admin123456/Desktop/work/topopt_exps/fluid_shape1/opt/u_res.pvd')
u_recorder.write_mesh(domain, 0)

# loss_recorder = TensorBoardRecorder(log_dir='/home/admin123456/Desktop/work/topopt_exps/fluid_shape1/opt/log')

# ------ init deform problem
phi = ufl.TrialFunction(S)
psi = ufl.TestFunction(S)

# ------ Method 1
a_riesz_ufl_1 = ufl.inner(ufl.grad(phi), ufl.grad(psi)) * ufl.dx
moving_factor_1 = 15.0
a_riesz_ufl_2 = ufl.inner(ufl.grad(phi), ufl.grad(psi)) * ufl.dx + ufl.inner(phi, psi) * ufl.dx  # better
moving_factor_2 = 20.0

a_riesz_ufl = a_riesz_ufl_2
moving_factor = moving_factor_2

riesz_bcs = []
for marker in [input_marker, output_marker, wallFixed_markers]:
    fixed_bry = dolfinx.fem.Function(S, name='fixed_bry_%d' % marker)
    bc_fixed_facets = MeshUtils.extract_facet_entities(domain, facet_tags, marker)
    bc_fixed_dofs = MeshUtils.extract_entity_dofs(S, fdim, bc_fixed_facets)  # please don't use sub function space here
    bc_fixed = dirichletbc(fixed_bry, bc_fixed_dofs)
    riesz_bcs.append(bc_fixed)

trial_displacement: Function = Function(displacement.function_space)
trial_displacement.assign(displacement)
last_loss = opt_problem([trial_displacement])
print(f"[### Original Cost]: {last_loss}")

dJ = dolfinx.fem.Function(trial_displacement.function_space)
displacement_cum = np.copy(trial_displacement.x.array)

best_loss = np.inf
step = 0
while True:
    step += 1

    grad = opt_problem.derivative()[0]
    L_riesz_ufl = ufl.inner(grad, psi) * ufl.dx

    res_dict = LinearProblemSolver.solve_by_petsc_form(
        domain.comm, dJ, a_riesz_ufl, L_riesz_ufl, riesz_bcs,
        ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
    )
    dJ = res_dict['res']

    dJ_np: np.ndarray = dJ.x.array
    # dJ_np = dJ_np / np.linalg.norm(dJ_np, ord=2)
    dJ_np = dJ_np * moving_factor * -1

    # dJ.x.array[:] = dJ_np
    # VisUtils.show_vector_res_vtk(grid, dJ, dim=2, with_wrap=True)

    displacement_cum += dJ_np
    # displacement_cum = np.maximum(np.minimum(dJ_np, 1.0), -1.0)
    trial_displacement.x.array[:] = displacement_cum

    loss = opt_problem([trial_displacement])
    # loss_recorder.write_scalar('loss', loss, step)

    if step % 10 == 0:
        latest_uh: dolfinx.fem.Function = uh.block_variable.checkpoint
        u_res = latest_uh.sub(0).collapse()
        # p_res = latest_uh.sub(1).collapse()
        u_recorder.write_function(u_res, step)

    if loss < best_loss:
        best_loss = loss
    print(f"[###Step {step}] loss:{loss:.5f} / best_loss:{best_loss:.5f}")

    if loss > best_loss * 1.5:
        break

    if step > 300:
        break
