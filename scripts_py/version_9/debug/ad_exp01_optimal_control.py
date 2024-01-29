import numpy as np
import ufl
from mpi4py import MPI
import dolfinx

from Thirdparty.pyadjoint.pyadjoint import *

from scripts_py.version_9.dolfinx_Grad.autoGrad_method.type_Function import Function
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.type_Mesh import Mesh
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.block_solve import solve
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.block_assemble import assemble
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.type_utils import start_annotation
from scripts_py.version_9.dolfinx_Grad.autoGrad_method.type_DirichletBC import dirichletbc, DirichletBC
from scripts_py.version_9.dolfinx_Grad.recorder_utils import XDMFRecorder
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils

tape = Tape()
set_working_tape(tape)

"""
# -------------------------------------
with start_annotation():
    domain: dolfinx.mesh.Mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 5, (0.0, 1.0))
    domain = Mesh(domain)
    grid = VisUtils.convert_to_grid(domain)

    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
    f1: Function = Function(V, name='control')
    f1.x.array[:] = 1.0
    F_form = 0.5 * f1 * f1 * ufl.dx
    J = assemble(F_form, domain)

control = Control(f1)
opt_problem = ReducedFunctional(J, control)

# f_opt = minimize(opt_problem, bounds=(0.1, 1.0), tol=1e-10, options={"gtol": 1e-10, "factr": 0.0})
# res_origin = opt_problem([f1])
# res_opt = opt_problem([f_opt])
# print(f"org_score:{res_origin} opt_score:{res_opt} pcg:{res_opt/res_origin}")

trial_f: Function = Function(V)
trial_f.assign(f1)
for _ in range(10):
    grad = opt_problem.derivative(adj_input=1.0)
    trial_f.x.array[:] = trial_f.x.array - 0.1 * grad.x.array
    loss = opt_problem([trial_f])
    print(grad.x.array, loss)
# -------------------------------------
"""

# ----------------------------------------------------
with start_annotation():
    domain: dolfinx.mesh.Mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 25, 25)
    domain = Mesh(domain)
    grid = VisUtils.convert_to_grid(domain)

    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
    u: Function = Function(V, name='State')
    v = ufl.TestFunction(V)
    f1: Function = Function(V, name='control')
    # f1.x.array[:] = 1.0

    F_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f1 * v * ufl.dx

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, facets)
    bc: DirichletBC = dirichletbc(0.0, dofs, V)

    solve(
        u, F_form, [bc],
        domain=domain, is_linear=False,
        tlm_ksp_option={
            'ksp_type': 'preonly', 'pc_type': 'lu',
        },
        adj_ksp_option={
            'ksp_type': 'preonly', 'pc_type': 'lu',
        },
        forward_ksp_option={
            'ksp_type': 'preonly', 'pc_type': 'lu',
        },
    )
    # VisUtils.show_scalar_res_vtk(grid, 'u', u)

    f_d: Function = Function(V)
    f_d.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
    # f_d.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    # VisUtils.show_scalar_res_vtk(grid, 'f_d', f_d)

    alpha = 1e-6
    exp_form = 0.5 * ufl.inner(u - f_d, u - f_d) * ufl.dx
    J = assemble(exp_form, domain)

control = Control(f1)
opt_problem = ReducedFunctional(J, [control])

# f_opt = minimize(opt_problem, bounds=(0.0, 0.8), tol=1e-10, options={"gtol": 1e-10, "factr": 0.0})
# res_origin = opt_problem([f1])
# res_opt = opt_problem([f_opt])
# print(f"org_score:{res_origin} opt_score:{res_opt} pcg:{res_opt/res_origin}")

recorder = XDMFRecorder(file='/home/admin123456/Desktop/work/topopt_exps/opt_control_01/res.xdmf')
recorder.write_mesh(domain)

trial_f: Function = Function(V)
trial_f.assign(f1)
orig_loss = opt_problem([trial_f])

step = 0
while True:
    step += 1

    grad: dolfinx.fem.Function = opt_problem.derivative(adj_input=1.0)[0]
    grad_np: np.array = grad.x.array
    grad_np = grad_np / np.linalg.norm(grad_np, ord=2)

    trial_np: np.array = trial_f.x.array
    trial_np = trial_np - 1.0 * grad_np

    trial_f.x.array[:] = trial_np
    loss = opt_problem([trial_f])

    opt_pcg = loss/orig_loss
    print(f"{step} loss: {loss}, Enhance:{opt_pcg}")

    if step % 30 == 0:
        recorder.write_function(u.block_variable.checkpoint, step)

    if opt_pcg < 0.01:
        break

recorder.write_function(u.block_variable.checkpoint, step=step)
VisUtils.show_scalar_res_vtk(grid, 'u_opt', u.block_variable.checkpoint)
# VisUtils.show_scalar_res_vtk(grid, 'control', trial_f)
