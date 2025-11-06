import dolfinx
from numpy.ma import true_divide
import ufl
from basix.ufl import element, mixed_element
from ufl import inner, grad, div

import numpy as np
import pyvista
from functools import partial
from sklearn.neighbors import KDTree

from thirdparty.pyadjoint.pyadjoint import Tape, ReducedFunctional, Control
from thirdparty.pyadjoint.pyadjoint.tape import set_working_tape

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils
from scripts_py.vtu_recorder import VtuRecorder
from scripts_py.simulator.dolfin_simulator import FluidBoundary

from scripts_py.auto_dolfinx.types.type_mesh import DolfinxMesh
from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.types.type_dirichletBC import DolfinxDirichletBC
from scripts_py.auto_dolfinx.blocks.block_solve import equation_solve
from scripts_py.auto_dolfinx.blocks.block_assemble import assemble_scalar
from scripts_py.auto_dolfinx.blocks.block_deform import deform
from scripts_py.auto_dolfinx.tape_utils import start_annotation

from scripts_py.auto_dolfinx.derivative_regularization import NormalDerivativeRegularization
from scripts_py.auto_dolfinx.levelset_topo.hjb_solver import HamiltonJacobiSolver
from scripts_py.auto_dolfinx.levelset_topo.reinitialization_solver import ReinitializationSolver
from scripts_py.auto_dolfinx.derivative_regularization import draw_shape_derivative


tape = Tape()
set_working_tape(tape)

mesh_file = '/home/admin123456/Desktop/work/gmsh_test/t16/levelset_fluid_02.msh'
xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t16/levelset_fluid_02.xdmf'
# MeshUtils.msh_to_XDMF(mesh_file, xdmf_file, dim=2)

mesh = MeshObj(xdmf_file=xdmf_file, mesh_name="model")
grid = mesh.grid
domain = DolfinxMesh(mesh.domain)
mesh.domain = domain

inlet_marker = 18
outlet_marker = 19
wall_marker = 20

inlet_face_center = np.array([0.0, 0.25])
inlet_face_norm_vector = np.array([1.0, 0.0])
inlet_face_radius = 0.25

outlet_face_center = np.array([7.0, 3.75])
outlet_face_norm_vector = np.array([1.0, 0.0])
outlet_face_radius = 0.25

# ------ define function space ------
coordinate_space = mesh.domain.ufl_domain().ufl_coordinate_element()
deformation_space = dolfinx.fem.functionspace(mesh.domain, coordinate_space)
point_scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
point_vector_func_space = DolfinUtils.point_vector_function_space(mesh.domain)

velocity_element = element("Lagrange", mesh.cell_types(), degree=2, shape=(mesh.tdim,))
pressure_element = element("Lagrange", mesh.cell_types(), degree=1)
mix_element = mixed_element([velocity_element, pressure_element])

W = dolfinx.fem.functionspace(mesh.domain, mix_element)
W0, W1 = W.sub(0), W.sub(1)
V, V_to_W_dofs = W0.collapse()
Q, Q_to_W_dofs = W1.collapse()
# -----------------------------------

displacement = DolfinxFunction(deformation_space, name='displacement')
displacement.x.array[:] = 0.0  # type: ignore


def init_obstacle_function(pos: np.ndarray):
    obstacle_center = np.array([3.5, 2.0, 0.0])
    pos2obs_dists = np.linalg.norm(pos.T - obstacle_center, axis=1)    
    obs_value = np.zeros(pos.shape[1])
    obs_value[pos2obs_dists < 0.5] = 1.0
    return obs_value


obstacle_fun = DolfinxFunction(point_scalar_func_space, name='obstacle')
obstacle_fun.interpolate(init_obstacle_function)  # type: ignore

with start_annotation():
    # ------ define state condition ------
    state_bcs: list[DolfinxDirichletBC] = []
    
    inflow_velocity_exp = partial(
        FluidBoundary.inflow_velocity_exp,
        velocity=np.array([1.0, 0.0]),
        tdim=mesh.tdim,
        face_center=inlet_face_center,
        face_norm_vector=inlet_face_norm_vector,
        face_radius=inlet_face_radius
    )
    
    inlet_fun = dolfinx.fem.Function(V, name='inlet_bc')
    inlet_fun.interpolate(inflow_velocity_exp)  # type: ignore
    state_bcs.append(
        DolfinxDirichletBC(
            name='inlet_bc',
            bc_func_space=W0,
            value_func_space=V,
            dofs=MeshUtils.extract_entity_dofs(
                (W0, V), mesh.fdim,
                MeshUtils.extract_entity_indices(
                    mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=inlet_marker
                )
            ),
            bc_value=inlet_fun,  # type: ignore
            is_mixed_element=True,
        )
    )
    
    pressure_fun = dolfinx.fem.Function(Q, name='outlet_bc')
    pressure_fun.x.array[:] = 0.0  # type: ignore
    state_bcs.append(
        DolfinxDirichletBC(
            name='outlet_bc',
            bc_func_space=W1,
            value_func_space=Q,
            dofs=MeshUtils.extract_entity_dofs(
                (W1, Q), mesh.fdim,
                MeshUtils.extract_entity_indices(
                    mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=outlet_marker
                )
            ),
            bc_value=pressure_fun,  # type: ignore
            is_mixed_element=True,
        )
    )
    
    wall_fun = dolfinx.fem.Function(V, name='wall_bc')
    wall_fun.x.array[:] = 0.0  # type: ignore
    state_bcs.append(
        DolfinxDirichletBC(
            name='wall_bc',
            bc_func_space=W0,
            value_func_space=V,
            dofs=MeshUtils.extract_entity_dofs(
                (W0, V), mesh.fdim,
                MeshUtils.extract_entity_indices(
                    mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=wall_marker
                )
            ),
            bc_value=wall_fun,  # type: ignore
            is_mixed_element=True,
        )
    )
    
    # ------
    deform(domain, displacement, tape=tape)  # type: ignore
    
    def init_levelset_function(pos: np.ndarray):
        init_path = np.array([
            [0, 0.25, 0],
            [4.75, 0.25, 0],
            [4.75, 3.75, 0],
            [7, 3.75, 0]
        ])
        detail_path = []
        for i in range(init_path.shape[0] - 1):
            step = int(np.linalg.norm(init_path[i + 1] - init_path[i]) / 0.1)
            detail_path.append(np.linspace(init_path[i], init_path[i + 1], step))
        detail_path = np.concatenate(detail_path, axis=0)
        
        tree = KDTree(detail_path)
        dists_list, _ = tree.query(pos.T, k=1, return_distance=True)
        dists_np = np.concatenate(dists_list, axis=-1)
        return dists_np - 0.25
    
    phi = DolfinxFunction(point_scalar_func_space, name='levelset_phi')
    phi.interpolate(init_levelset_function)  # type: ignore
    
    # ------ visualize the initial level set function ------
    # grid.point_data['phi'] = phi.x.array.copy()  # type: ignore
    # grid.save('/home/admin123456/Desktop/work/gmsh_test/t16/init_phi.vtu')
    # ---------------------------------------------------

    def heaviside_step_fun(
        phi: dolfinx.fem.Function|np.ndarray, 
        beta=200.0, 
        rho_max=1.0,
        rho_min=1e-8, 
        exp_operator=ufl.exp
    ):
        return rho_max / (1.0 + exp_operator(-beta * phi)) + rho_min  # type: ignore
    
    uh = DolfinxFunction(W, name='state')
    u, p = ufl.split(uh)  # type: ignore
    vq = DolfinxFunction(W, name='adjoint')
    v, q = ufl.split(vq)  # type: ignore
    
    f = dolfinx.fem.Constant(mesh.domain, np.zeros(mesh.tdim))
    nut = DolfinxFunction(point_scalar_func_space, name='nut')
    nut.x.array[:] = 1.0 / 400.0  # type: ignore
    
    F_form = (
        nut * inner(grad(u), grad(v)) * ufl.dx  # type: ignore
        - inner(p, div(v)) * ufl.dx
        - inner(q, div(u)) * ufl.dx
        - inner(f, v) * ufl.dx
        + heaviside_step_fun(phi, rho_max=1e4) * inner(u, v) * ufl.dx  # type: ignore
        # + 1e4 * obstacle_fun * inner(u, v) * ufl.dx  # type: ignore  TODO: no work
    )
    
    uh, state_eqn_block = equation_solve(
        domain=domain,
        F_form=F_form,
        state=uh,  # type: ignore
        adjoint=vq,  # type: ignore
        bcs=state_bcs,  # type: ignore
        is_linear=True,
        tape=tape,
        with_block=True,
    )
    u, p = ufl.split(uh)  # type: ignore
    
    energy_weight = 1.0
    object_form = energy_weight * nut * inner(grad(u), grad(u)) * ufl.dx  # type: ignore
    object_cost = assemble_scalar(object_form, domain=domain, tape=tape)

    volume_form = heaviside_step_fun(-phi, rho_max=1.0) * ufl.dx  # type: ignore
    target_volume = AssembleUtils.assemble_scalar(volume_form) * 0.9
    
    volume_weight = 1e-1
    volume_ = assemble_scalar(volume_form, domain, tape)
    volume_loss = volume_weight * 0.5 * (volume_ - target_volume) ** 2  # type: ignore
    
    Jhat = object_cost + volume_loss
    
    # # NOTE: obstacle is non't stable, since hjb solver will not consider the obstacle.
    # # TODO: no work
    # obstacle_weight = 1e4
    # obstacle_loss_form = obstacle_weight * obstacle_fun * heaviside_step_fun(-phi, rho_max=1.0) * ufl.dx  # type: ignore
    # obstacle_loss = assemble_scalar(obstacle_loss_form, domain=domain, tape=tape)
    # Jhat = Jhat + obstacle_loss
    
    control_displacement = Control(displacement)  # type: ignore
    opt_problem = ReducedFunctional(Jhat, [control_displacement])


def eval():
    return {
        'object_cost': AssembleUtils.assemble_scalar(object_form),
        'volume_cost': volume_weight * 0.5 * (AssembleUtils.assemble_scalar(volume_form) - target_volume) ** 2,
        # 'obstacle_cost': AssembleUtils.assemble_scalar(obstacle_loss_form),
    }

reg_solver = NormalDerivativeRegularization(
    name='normal_derivative_regularization',
    mesh=mesh,
    fix_markers=[inlet_marker, outlet_marker],
    beta=0.1,
    gamma=1.0e4,
    mat_assemble_method='lift'
)


def update_phi(new_phi_np: np.ndarray):
    for block_variable in state_eqn_block.get_dependencies():
        if isinstance(block_variable.output, DolfinxFunction):
            if block_variable.output.name == phi.name:  # type: ignore
                new_phi = dolfinx.fem.Function(phi.function_space)  # type: ignore
                new_phi.x.array[:] = new_phi_np.copy()  # type: ignore
                block_variable.checkpoint = new_phi  # type: ignore
                c_ref = block_variable.saved_output


recorder = VtuRecorder(mesh, save_dir='/home/admin123456/Desktop/work/gmsh_test/t16/cache', name='levelset')
run_iter = 0


def record_Upw():
    vel_ref = dolfinx.fem.Function(point_vector_func_space, name='vel_ref')
    vel_ref.interpolate(uh.sub(0).collapse())  # type: ignore
    p_ref = uh.sub(1).collapse()  # type: ignore

    vel_ref_np = vel_ref.x.array.copy().reshape((-1, mesh.tdim))  # type: ignore
    if vel_ref_np.shape[1] == 2:
        vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)

    recorder.update_point_data('U', vel_ref_np)
    recorder.update_point_data('p', p_ref.x.array.copy())


record_Upw()  # type: ignore
recorder.update_point_data('phi', phi.x.array)  # type: ignore
recorder.update_point_data('h_phi', heaviside_step_fun(-phi.x.array, rho_max=1.0, exp_operator=np.exp))  # type: ignore
recorder.update_point_data('obstacle', obstacle_fun.x.array)  # type: ignore
recorder.save(0)

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
    
    # NOTE: obstacle is non't stable, since hjb solver will not consider the obstacle.
    # NOTE: so when the loss become larger, we retain phi and abandon this update.
    hjb_solver = HamiltonJacobiSolver(
        domain=domain,
        velocity=dJ_dmesh_fun,  # type: ignore
        phi=phi,  # type: ignore
        with_diffusion=True,
        diffusion_coefficient=1e-4,
        with_GLS_stabilization=True
    )
    solver = hjb_solver.solve(
        dt=0.01, 
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
    recorder.update_point_data('h_phi', heaviside_step_fun(-phi.x.array, rho_max=1.0, exp_operator=np.exp))  # type: ignore  
    
    if run_iter % 5 == 0:
        phi_new = ReinitializationSolver.solve(domain, phi)  # type: ignore
        update_phi(phi_new)
        
        # ------ record reinitialized phi ------
        recorder.update_point_data('phi_reinitialized', phi.x.array)  # type: ignore
        recorder.update_point_data('h_phi_reinitialized', heaviside_step_fun(-phi.x.array, rho_max=1.0, exp_operator=np.exp))  # type: ignore
        recorder.save(run_iter)
    
    loss_current = opt_problem([displacement])  # type: ignore
    record_Upw()
    recorder.save(run_iter)
    
    loss_dict = eval()
    log = f"step/{run_iter} loss:{loss_current:.3f}"
    for name, cost_value in loss_dict.items():
        log += f" {name}:{cost_value:.5f}"
    print(log)
    
    if run_iter > 200:
        break
    
