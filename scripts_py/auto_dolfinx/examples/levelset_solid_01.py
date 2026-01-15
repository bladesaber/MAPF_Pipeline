'''reference: https://github.com/LLNL/letop/tree/main/letop_examples/cantilever'''

import dolfinx
import ufl

import numpy as np
import pyvista

from thirdparty.pyadjoint.pyadjoint import Tape, ReducedFunctional, Control
from thirdparty.pyadjoint.pyadjoint.tape import set_working_tape

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils
from scripts_py.vtu_recorder import VtuRecorder

from scripts_py.auto_dolfinx.types.type_mesh import DolfinxMesh
from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.types.type_dirichletBC import DolfinxDirichletBC
from scripts_py.auto_dolfinx.blocks.block_solve import equation_solve
from scripts_py.auto_dolfinx.blocks.block_assemble import assemble_scalar
from scripts_py.auto_dolfinx.blocks.block_deform import deform
from scripts_py.auto_dolfinx.tape_utils import start_annotation

from scripts_py.auto_dolfinx.derivative_regularization import NormalDerivativeRegularization
from scripts_py.auto_dolfinx.opt_solver.hjb_solver import HamiltonJacobiSolver
from scripts_py.auto_dolfinx.opt_solver.reinitialization_solver import ReinitializationSolver
from scripts_py.auto_dolfinx.derivative_regularization import draw_shape_derivative


tape = Tape()
set_working_tape(tape)

msh_file = "/home/admin123456/Desktop/work/gmsh_test/t15/levelset_01.msh"
xdmf_file = "/home/admin123456/Desktop/work/gmsh_test/t15/levelset_01.xdmf"
# MeshUtils.msh_to_XDMF(msh_file, xdmf_file, dim=2)

mesh = MeshObj(xdmf_file=xdmf_file, mesh_name="model")
grid = mesh.grid
domain = DolfinxMesh(mesh.domain)
mesh.domain = domain

wall_fix_marker = 10
wall_free_marker = 11
wall_force_marker = 12

coordinate_space = mesh.domain.ufl_domain().ufl_coordinate_element()
deformation_space = dolfinx.fem.functionspace(mesh.domain, coordinate_space)
point_scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
point_vector_func_space = DolfinUtils.point_vector_function_space(mesh.domain)

displacement = DolfinxFunction(deformation_space, name='displacement')
displacement.x.array[:] = 0.0  # type: ignore

with start_annotation():
    deform(domain, displacement, tape=tape)  # type: ignore

    # ------ init level set function ------
    def init_levelset_function(pos: np.ndarray):
        pos = pos.T
        lx = 2.0
        ly = 1.0
        phi_expr = (
            -np.cos(6.0 / lx * np.pi * pos[:, 0]) * np.cos(4.0 * np.pi * pos[:, 1])
            - 0.6
            + np.maximum(200.0 * (0.01 - pos[:, 0] ** 2 - (pos[:, 1] - ly / 2) ** 2), 0.0)
            + np.maximum(100.0 * (pos[:, 0] + pos[:, 1] - lx - ly + 0.1), 0.0)
            + np.maximum(100.0 * (pos[:, 0] - pos[:, 1] - lx + 0.1), 0.0)
        )
        return phi_expr

    phi = DolfinxFunction(point_scalar_func_space, name='levelset_phi')
    phi.interpolate(init_levelset_function)  # type: ignore

    def heaviside_step_fun(phi: dolfinx.fem.Function, beta=200.0, rho_min=1e-5, exp_operator=ufl.exp):
        return 1.0 / (1.0 + exp_operator(-beta * phi)) + rho_min  # type: ignore

    # Elasticity parameters
    E, nu = 1.0, 0.3
    mu: float = E / (2 * (1 + nu))
    lmbda: float = E * nu / ((1 + nu) * (1 - 2 * nu))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(2)  # type: ignore


    strain = DolfinxFunction(point_vector_func_space, name='strain')
    strain_v = DolfinxFunction(point_vector_func_space, name='strain_v')

    f = dolfinx.fem.Constant(domain, np.array([0.0, 0.0]))
    T = dolfinx.fem.Constant(domain, np.array([0.0, -75.0]))
    ds = MeshUtils.define_ds(domain, facets_tag=mesh.facet_tags)

    # a_form = ufl.inner(sigma(strain), epsilon(strain_v)) * ufl.dx
    # TODO: check if my implementation is correct ?
    a_form = ufl.inner(heaviside_step_fun(-phi) * sigma(strain), epsilon(strain_v)) * ufl.dx  # type: ignore
    L_form = ufl.dot(f, strain_v) * ufl.dx + ufl.dot(T, strain_v) * ds(wall_force_marker)

    wall_fix_fun = dolfinx.fem.Function(point_vector_func_space, name='wall_fix')
    wall_fix_fun.x.array[:] = 0.0  # type: ignore
    bcs: list[DolfinxDirichletBC] = [
        DolfinxDirichletBC(
            name='wall_fix',
            bc_func_space=point_vector_func_space,
            value_func_space=point_vector_func_space,
            dofs=MeshUtils.extract_entity_dofs(
                point_vector_func_space, mesh.fdim,
                MeshUtils.extract_entity_indices(
                    mesh.domain, style='facet', marker=wall_fix_marker, mesh_tags=mesh.facet_tags
                )
            ),
            bc_value=wall_fix_fun,  # type: ignore
            is_mixed_element=False,
        )
    ]

    strain, linear_elastic_eqn_block = equation_solve(
        domain=domain,
        F_form=a_form - L_form,
        state=strain,  # type: ignore
        adjoint=strain_v,  # type: ignore
        bcs=bcs,  # type: ignore
        is_linear=True,
        tape=tape,
        with_block=True,
    )

    energy_weight = 1e-3
    object_form = energy_weight * ufl.inner(heaviside_step_fun(-phi) * sigma(strain), epsilon(strain)) * ufl.dx  # type: ignore
    object_cost_ = assemble_scalar(object_form, domain=domain, tape=tape)

    volume_space = AssembleUtils.assemble_scalar(dolfinx.fem.Constant(domain, 1.0) * ufl.dx)  # type: ignore
    target_volume = volume_space * 0.3

    volume_form = heaviside_step_fun(-phi) * ufl.dx  # type: ignore
    volume_phi = assemble_scalar(volume_form, domain=domain, tape=tape)

    volume_weight = 1.0
    volume_loss = volume_weight * 0.5 * (volume_phi - target_volume) ** 2

    Jhat = object_cost_ + volume_loss

    control_displacement = Control(displacement)  # type: ignore
    opt_problem = ReducedFunctional(Jhat, [control_displacement])
    

def eval():
    return {
        # TODO: energy_weight use again
        'object_cost': energy_weight * AssembleUtils.assemble_scalar(object_form),
        'volume_cost': volume_weight * 0.5 * (AssembleUtils.assemble_scalar(volume_form) - target_volume) ** 2,
    }


reg_solver = NormalDerivativeRegularization(
    name='normal_derivative_regularization',
    mesh=mesh,
    fix_markers=[wall_fix_marker, wall_force_marker],
    beta=0.1,
    gamma=1.0e4,
    mat_assemble_method='lift'
)


def update_phi(new_phi_np: np.ndarray):
    for block_variable in linear_elastic_eqn_block.get_dependencies():
        if isinstance(block_variable.output, DolfinxFunction):
            if block_variable.output.name == phi.name:  # type: ignore
                new_phi = dolfinx.fem.Function(phi.function_space)  # type: ignore
                new_phi.x.array[:] = new_phi_np.copy()  # type: ignore
                block_variable.checkpoint = new_phi  # type: ignore
                c_ref = block_variable.saved_output


recorder = VtuRecorder(mesh, save_dir='/home/admin123456/Desktop/work/gmsh_test/t15/cache', name='levelset')
run_iter = 0

recorder.update_point_data('phi', phi.x.array)  # type: ignore
recorder.update_point_data('h_phi', heaviside_step_fun(-phi.x.array, exp_operator=np.exp))  # type: ignore
recorder.save(run_iter)

while True:
    run_iter += 1
    
    dJ_dmesh = opt_problem.derivative(adj_input=1.0, apply_riesz=False)[0]
    # draw_shape_derivative(mesh.grid, dJ_dmesh.reshape((-1, 2)), scale=0.1, title='dJ_dmesh')
    dJ_dmesh = reg_solver.solve(dJ_dmesh)
    # draw_shape_derivative(mesh.grid, dJ_dmesh.reshape((-1, 2)), scale=0.1, title='dJ_dmesh_regularized')
    
    dJ_dmesh = dJ_dmesh.reshape((-1, 2))
    dJ_dmesh = dJ_dmesh / np.max(np.linalg.norm(dJ_dmesh, ord=2, axis=1))
    dJ_dmesh = dJ_dmesh * -1.0
    
    dJ_dmesh_fun = dolfinx.fem.Function(deformation_space, name='dJ_dmesh')
    dJ_dmesh_fun.x.array[:] = dJ_dmesh.reshape(-1)  # type: ignore
        
    hjb_solver = HamiltonJacobiSolver(
        domain=domain,
        velocity=dJ_dmesh_fun,  # type: ignore
        phi=phi,  # type: ignore
        with_diffusion=True,
        diffusion_coefficient=1e-4,
        with_GLS_stabilization=True
    )
    solver = hjb_solver.solve(
        dt=0.001, 
        run_steps=10, 
        ksp_option={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        }, 
        snes_option={}, 
        mat_assemble_method='lift',
        only_solver=True
    )
    for ts_step in range(20):
        solver.step()  # type: ignore
        phi_new = hjb_solver.get_solution()
        diff_max = np.max(np.abs(phi_new - phi.x.array))  # type: ignore
        if diff_max >= 0.1:
            break
    
    update_phi(phi_new)
    recorder.update_point_data('phi', phi.x.array)  # type: ignore
    recorder.update_point_data('h_phi', heaviside_step_fun(-phi.x.array, exp_operator=np.exp))  # type: ignore
    
    if run_iter % 5 == 0:
        phi_new = ReinitializationSolver.solve(domain, phi)  # type: ignore
        update_phi(phi_new)
        
        # ------ record reinitialized phi ------
        recorder.update_point_data('phi_reinitialized', phi.x.array)  # type: ignore
        recorder.update_point_data('h_phi_reinitialized', heaviside_step_fun(-phi.x.array, exp_operator=np.exp))  # type: ignore
        recorder.save(run_iter)
    
    loss_current = opt_problem([displacement])  # type: ignore
    recorder.save(run_iter)
    
    loss_dict = eval()
    log = f"step/{run_iter} loss:{loss_current:.3f}"
    for name, cost_value in loss_dict.items():
        log += f" {name}:{cost_value:.5f}"
    print(log)
    
    if run_iter > 100:
        break
    