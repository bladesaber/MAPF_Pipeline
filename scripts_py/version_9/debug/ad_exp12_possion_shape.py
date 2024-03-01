"""

Fail

"""
import os
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
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, UFLUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder, TensorBoardRecorder
from scripts_py.version_9.dolfinx_Grad.equation_solver import LinearProblemSolver

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/possion_shape'
model_xdmf = os.path.join(proj_dir, 'model.xdmf')

# # ------ create xdmf
# msh_file = '/home/admin123456/Desktop/work/topopt_exps/fluid_shape1/model.msh'
# MeshUtils.msh_to_XDMF(
#     name='model',
#     msh_file=os.path.join(proj_dir, 'model.msh'),
#     output_file=model_xdmf,
#     dim=2
# )
# # -------------------

# tape = RecordTape()
tape = Tape()
set_working_tape(tape)

with start_annotation():
    boundary_markers = 3

    domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
        file=model_xdmf,
        mesh_name='model',
        cellTag_name='model_cells',
        facetTag_name='model_facets'
    )
    domain: Mesh = Mesh(domain)
    grid = VisUtils.convert_to_grid(domain)

    tdim = domain.topology.dim
    fdim = tdim - 1

    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))

    coordinate_space = domain.ufl_domain().ufl_coordinate_element()
    S = dolfinx.fem.FunctionSpace(domain, coordinate_space)

    # ---------- Boundary Define
    bcs = []
    bc_facets = MeshUtils.extract_facet_entities(domain, facet_tags, boundary_markers)
    bc_dofs = MeshUtils.extract_entity_dofs(V, fdim, bc_facets)
    bc0 = dirichletbc(0.0, bc_dofs, V)
    bcs.append(bc0)
    # ----------

    displacement: Function = Function(S, name='control')
    move(domain, displacement)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    coodr = MeshUtils.define_coordinate(domain)
    f_exp = 2.5 * np.power(coodr[0] + 0.4 - np.power(coodr[1], 2), 2) + \
            np.power(coodr[0], 2) + np.power(coodr[1], 2) - 1
    # f_exp = coodr[0] * coodr[0] + coodr[1] * coodr[1] - 0.25

    # vis_f = dolfinx.fem.Function(V, name='vis_f')
    # vis_f.interpolate(UFLUtils.create_expression(f_exp, V))
    # VisUtils.show_scalar_res_vtk(grid, 'vis_f', vis_f)

    F_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f_exp * v * ufl.dx
    # F_form = ufl.inner(u, v) * ufl.dx - f_exp * v * ufl.dx

    a_form: ufl.form.Form = ufl.lhs(F_form)
    L_form: ufl.form.Form = ufl.rhs(F_form)

    uh = Function(V, name='state')
    uh = solve(
        uh, a_form, L_form, bcs,
        domain=domain, is_linear=True,
        tlm_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu'},
        adj_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        forward_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        with_debug=False,
    )

    cost_form = uh * ufl.dx
    J = assemble(cost_form, domain)

    # VisUtils.show_scalar_res_vtk(grid, 'uh', uh)

# uh0 = dolfinx.fem.Function(uh.function_space)
# uh0.x.array[:] = uh.x.array
# grid0 = grid.copy()
#
# domain_xy = domain.geometry.x[:, :tdim]
# domain_xy += 1.0
# uh = solve(
#     uh, a_form, L_form, bcs,
#     domain=domain, is_linear=True,
#     tlm_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu'},
#     adj_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
#     forward_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
#     with_debug=False,
# )
# uh1 = dolfinx.fem.Function(uh.function_space)
# uh1.x.array[:] = uh.x.array
# grid1 = grid.copy()
#
# import pyvista
# grid0.point_data['uh0'] = uh0.x.array.real
# grid0.set_active_scalars('uh0')
# grid1.point_data['uh1'] = uh1.x.array.real
# grid1.set_active_scalars('uh1')
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid0, show_edges=True)
# plotter.add_mesh(grid1, show_edges=True)
# plotter.show()

control = Control(displacement)
opt_problem = ReducedFunctional(J, [control])

# ------ init recorder
u_recorder = VTKRecorder(file=os.path.join(proj_dir, 'u_res.pvd'))
u_recorder.write_mesh(domain, 0)

# ------ init deform problem
phi = ufl.TrialFunction(S)
psi = ufl.TestFunction(S)

a_riesz_ufl = ufl.inner(ufl.grad(phi), ufl.grad(psi)) * ufl.dx
riesz_bcs = []

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
        ksp_option={'ksp_type': 'preonly', 'pc_type': 'ksp'},
        with_debug=True
    )

    dJ = res_dict['res']

    dJ = dJ.x.array.reshape((-1, tdim))
    domain.geometry.x[:, :tdim] = domain.geometry.x[:, :tdim] + dJ

    print(dJ)
    import pyvista
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.show()

    break

    # dJ_np: np.ndarray = dJ.x.array
    # # dJ_np = dJ_np / np.linalg.norm(dJ_np, ord=2)
    # dJ_np = dJ_np * -0.1
    #
    # # dJ.x.array[:] = dJ_np
    # # VisUtils.show_vector_res_vtk(grid, dJ, dim=2, with_wrap=True)
    #
    # displacement_cum += dJ_np
    # # displacement_cum = np.maximum(np.minimum(dJ_np, 1.0), -1.0)
    # trial_displacement.x.array[:] = displacement_cum
    #
    # loss = opt_problem([trial_displacement])
    # # VisUtils.show_scalar_res_vtk(grid, 'update', uh.block_variable.checkpoint)
    #
    # if step % 10 == 0:
    #     latest_uh: dolfinx.fem.Function = uh.block_variable.checkpoint
    #     u_recorder.write_function(latest_uh, step)
    #
    # if loss < best_loss:
    #     best_loss = loss
    # print(f"[###Step {step}] loss:{loss:.5f} / best_loss:{best_loss:.5f}")
    #
    # if loss > best_loss * 1.1:
    #     break
    #
    # if step > 300:
    #     break
