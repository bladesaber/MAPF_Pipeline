'''
该文件主要对min_opt_03.py的流程做整合,实现固液同步优化
'''

import dolfinx
import ufl
from basix.ufl import element, mixed_element
from ufl import inner, grad, div

import numpy as np
import pyvista
from functools import partial
from sklearn.neighbors import KDTree

from thirdparty.pyadjoint.pyadjoint import Tape, ReducedFunctional, Control
from thirdparty.pyadjoint.pyadjoint.tape import set_working_tape, stop_annotating

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import BoundaryUtils, DolfinUtils, AssembleUtils
from scripts_py.vtu_recorder import VtuRecorder
from scripts_py.simulator.dolfin_simulator import FluidBoundary
from scripts_py.eqn_solver import LinearProblemSolver
from scripts_py.dolfinx_adjoint.topology_utils.topology_filter import DistanceBandFilter

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


def heaviside_step_fun(
    phi: dolfinx.fem.Function|np.ndarray, 
    beta=200.0, 
    rho_max=1.0,
    rho_min=1e-8, 
    exp_operator=ufl.exp
):
    return rho_max / (1.0 + exp_operator(-beta * phi)) + rho_min  # type: ignore


def dirac_delta(phi, beta=200.0, exp_operator=ufl.exp, abs_operator=np.abs):
    return exp_operator(-beta * abs_operator(phi))


mesh_file = '/home/admin123456/Desktop/work/gmsh_test/t20/mix_opt_05.msh'
xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t20/mix_opt_05.xdmf'
# MeshUtils.msh_to_XDMF(mesh_file, xdmf_file, dim=2)

mesh = MeshObj(xdmf_file=xdmf_file, mesh_name="model")
grid = mesh.grid
domain = DolfinxMesh(mesh.domain)
mesh.domain = domain

inlet_marker = 10
outlet_marker = 11
wall_fix_marker = 12
wall_free_marker = 13

inlet_face_center = np.array([0.0, 1.5])
inlet_face_norm_vector = np.array([1.0, 0.0])
inlet_face_radius = 0.25

point_scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
point_vector_func_space = DolfinUtils.point_vector_function_space(mesh.domain)
coordinate_space = domain.ufl_domain().ufl_coordinate_element()
deformation_space = dolfinx.fem.functionspace(domain, coordinate_space)

tape = Tape()
set_working_tape(tape)
recorder = VtuRecorder(mesh, save_dir='/home/admin123456/Desktop/work/gmsh_test/t20/cache', name='mix')

def solve_fluid_problem():
    with start_annotation():
        velocity_element = element("Lagrange", mesh.cell_types(), degree=2, shape=(mesh.tdim,))
        pressure_element = element("Lagrange", mesh.cell_types(), degree=1)
        mix_element = mixed_element([velocity_element, pressure_element])

        W = dolfinx.fem.functionspace(mesh.domain, mix_element)
        W0, W1 = W.sub(0), W.sub(1)
        V, V_to_W_dofs = W0.collapse()
        Q, Q_to_W_dofs = W1.collapse()
        
        bcs: list[DolfinxDirichletBC] = []
            
        inflow_velocity_exp = partial(
            FluidBoundary.inflow_velocity_exp,
            velocity=np.array([1.0, 0.0]),
            tdim=mesh.tdim,
            face_center=inlet_face_center,
            face_norm_vector=inlet_face_norm_vector,
            face_radius=inlet_face_radius
        )
            
        inlet_fun = dolfinx.fem.Function(V, name='fluid_inlet_bc')
        inlet_fun.interpolate(inflow_velocity_exp)  # type: ignore
        bcs.append(
            DolfinxDirichletBC(
                name='fluid_inlet_bc',
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
            
        pressure_fun = dolfinx.fem.Function(Q, name='fluid_outlet_bc')
        pressure_fun.x.array[:] = 0.0  # type: ignore
        bcs.append(
            DolfinxDirichletBC(
                name='fluid_outlet_bc',
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
            
        wall_fix_fun = dolfinx.fem.Function(V, name='fluid_wall_fix_bc')
        wall_fix_fun.x.array[:] = 0.0  # type: ignore
        bcs.append(
            DolfinxDirichletBC(
                name='fluid_wall_fix_bc',
                bc_func_space=W0,
                value_func_space=V,
                dofs=MeshUtils.extract_entity_dofs(
                    (W0, V), mesh.fdim,
                    MeshUtils.extract_entity_indices(
                        mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=wall_fix_marker
                    )
                ),
                bc_value=wall_fix_fun,  # type: ignore
                is_mixed_element=True,
            )
        )
            
        wall_free_fun = dolfinx.fem.Function(V, name='fluid_wall_free_bc')
        wall_free_fun.x.array[:] = 0.0  # type: ignore
        bcs.append(
            DolfinxDirichletBC(
                name='fluid_wall_free_bc',
                bc_func_space=W0,
                value_func_space=V,
                dofs=MeshUtils.extract_entity_dofs(
                    (W0, V), mesh.fdim,
                    MeshUtils.extract_entity_indices(
                        mesh.domain, style='facet', mesh_tags=mesh.facet_tags, marker=wall_free_marker
                    )
                ),
                bc_value=wall_free_fun,  # type: ignore
                is_mixed_element=True,
            )
        )
        
        # ------ deform
        displacement = DolfinxFunction(deformation_space, name='displacement')
        displacement.x.array[:] = 0.0  # type: ignore
        deform(domain, displacement, tape=tape)  # type: ignore

        # ------ define state problem
        def init_levelset_function(pos: np.ndarray):
            # NOTE: please change here
            init_path = np.array([
                [0, 1.5, 0],
                [2.5, 1.5, 0],
                [2.5, 4.25, 0],
                [5, 4.25, 0]
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
        recorder.update_point_data('phi', phi.x.array)  # type: ignore
        
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
        )
        
        uh, state_eqn_block = equation_solve(
            domain=domain,
            F_form=F_form,
            state=uh,  # type: ignore
            adjoint=vq,  # type: ignore
            bcs=bcs,  # type: ignore
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
        
        volume_weight = 1.0
        volume_ = assemble_scalar(volume_form, domain, tape)
        volume_loss = volume_weight * 0.5 * (volume_ - target_volume) ** 2  # type: ignore
        
        Jhat = object_cost + volume_loss
                        
        
    def eval():
        return {
            'fobject': AssembleUtils.assemble_scalar(object_form),
            'fvolume': volume_weight * 0.5 * (AssembleUtils.assemble_scalar(volume_form) - target_volume) ** 2,
        }
    print('[fluid] eval_fun: ', eval())
        
    def record_Upw():
        vel_ref = dolfinx.fem.Function(point_vector_func_space, name='vel_ref')
        vel_ref.interpolate(uh.sub(0).collapse())  # type: ignore
        p_ref = uh.sub(1).collapse()  # type: ignore

        vel_ref_np = vel_ref.x.array.copy().reshape((-1, mesh.tdim))  # type: ignore
        if vel_ref_np.shape[1] == 2:
            vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)

        recorder.update_point_data('U', vel_ref_np)
        recorder.update_point_data('p', p_ref.x.array.copy())
    record_Upw()
        
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
        
    return {
        'Jhat': Jhat,
        'fluid_u': u,
        'fluid_p': p,
        'fluid_phi': phi,
        'displacement': displacement,
        'eval_fun': eval,
        'reg_solver': reg_solver,
        'state_eqn_block': state_eqn_block,
        'record_Upw': record_Upw,
        'update_phi': update_phi,
    }


def solve_topo_solid(fluid_result: dict):
    phi: DolfinxFunction = fluid_result['fluid_phi']
    p: DolfinxFunction = fluid_result['fluid_p']
    
    # using safe distance for avoiding overlap with fluid zone
    safe_dist = 0.075
    surface_phi = dirac_delta(phi - safe_dist, exp_operator=ufl.exp, beta=200.0)  # type: ignore
    surface_nvec = ufl.grad(phi)
    
    # NOTE: very small here
    surface_f: ufl.Form = surface_phi * surface_nvec * (p + 0.1)  # type: ignore
    fluid_zone: ufl.Form = heaviside_step_fun(-phi, rho_max=1.0, rho_min=0.0, exp_operator=ufl.exp)  # type: ignore
    
    def record_surface_force():
        with stop_annotating():
            v = ufl.TestFunction(point_vector_func_space)
            surface_f_form_tmp = ufl.inner(surface_f, v) * ufl.dx
            surface_f_np = AssembleUtils.assemble_vec(surface_f_form_tmp).array
            surface_f_np = surface_f_np.reshape((-1, mesh.tdim))    
            if surface_f_np.shape[1] == 2:
                surface_f_np = np.concatenate([surface_f_np, np.zeros((surface_f_np.shape[0], 1))], axis=1)
            recorder.update_point_data('surface_force', surface_f_np)
    record_surface_force()
    
    def record_fluid_zone():
        with stop_annotating():
            fluid_zone_fun = dolfinx.fem.Function(point_scalar_func_space, name='fluid_zone')
            fluid_zone_fun.interpolate(DolfinUtils.create_expression(fluid_zone, point_scalar_func_space))  # type: ignore
            recorder.update_point_data('fluid_zone', fluid_zone_fun.x.array)  # type: ignore
    record_fluid_zone()
    
    # -------------------------------------------
    E, nu = 1.0, 0.3
    mu: float = E / (2 * (1 + nu))
    lmbda: float = E * nu / ((1 + nu) * (1 - 2 * nu))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(2)  # type: ignore

    def alpha_fun(rho: np.ndarray|dolfinx.fem.Function, alpha_max=1.0, alpha_min=1e-3, p=5.0, pow_operator=ufl.elem_pow):
        return alpha_min + pow_operator(rho, p) * (alpha_max - alpha_min)  # type: ignore

    with start_annotation():
        strain = DolfinxFunction(point_vector_func_space, name='strain')
        strain_v = DolfinxFunction(point_vector_func_space, name='strain_v')
        T = dolfinx.fem.Constant(domain, np.array([0.0, 0.0]))
        
        rho = DolfinxFunction(point_scalar_func_space, name='rho')
        
        # ------ init rho ------
        rho.x.array[:] = 1e-2  # type: ignore
        with stop_annotating():
            # NOTE: please don't annotate here
            init_rho = dolfinx.fem.Function(point_scalar_func_space, name='fluid_zone')
            init_rho.interpolate(DolfinUtils.create_expression(fluid_zone, point_scalar_func_space))  # type: ignore
            rho.x.array[np.where(init_rho.x.array < 1e-6)[0]] = 0.5  # type: ignore
            recorder.update_point_data('rho', rho.x.array)  # type: ignore
        # rho.x.array[:] = 0.5  # type: ignore
        
        a_form = ufl.inner(alpha_fun(rho) * sigma(strain), epsilon(strain_v)) * ufl.dx  # type: ignore
        L_form = ufl.dot(surface_f, strain_v) * ufl.dx + ufl.dot(T, strain_v) * ufl.ds

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

        strain = equation_solve(
            domain=domain,
            F_form=a_form - L_form,
            state=strain,  # type: ignore
            adjoint=strain_v,  # type: ignore
            bcs=bcs,  # type: ignore
            is_linear=True,
            tape=tape,
            with_block=False,
        )
        
        def record_strain():
            with stop_annotating():
                strain_np = strain.x.array.reshape((-1, mesh.tdim))  # type: ignore
                if strain_np.shape[1] == 2:
                    strain_np = np.concatenate([strain_np, np.zeros((strain_np.shape[0], 1))], axis=1)
                recorder.update_point_data('strain', strain_np)
        record_strain()
        
        energy_weight = 1e1
        object_form = energy_weight * ufl.inner(alpha_fun(rho) * sigma(strain), epsilon(strain)) * ufl.dx  # type: ignore
        object_cost = assemble_scalar(object_form, domain=domain, tape=tape)

        with stop_annotating():
            volume_space = (
                AssembleUtils.assemble_scalar(dolfinx.fem.Constant(domain, 1.0) * ufl.dx)  # type: ignore
                - AssembleUtils.assemble_scalar(fluid_zone * ufl.dx)
            )
            target_volume = volume_space * 0.25
        
        volume_form = rho * ufl.dx  # type: ignore
        volume_rho = assemble_scalar(volume_form, domain=domain, tape=tape)
        
        volume_weight = 1.0
        volume_loss = volume_weight * 0.5 * (volume_rho - target_volume) ** 2
        
        obstacle_weight = 1e4
        # obstacle_form = obstacle_weight * ufl.inner(obstacle, rho) * ufl.dx  # type: ignore
        obstacle_form = obstacle_weight * ufl.inner(
            ufl.conditional(ufl.ge(fluid_zone, 0.75), fluid_zone, 0.0), 
            rho
        ) ** 2 * ufl.dx  # type: ignore
        obstacle_loss = assemble_scalar(obstacle_form, domain=domain, tape=tape)
        
        Jhat = object_cost + volume_loss + obstacle_loss
        
    def eval():
        return {
            'sobject': AssembleUtils.assemble_scalar(object_form),
            'svolume': volume_weight * 0.5 * (AssembleUtils.assemble_scalar(volume_form) - target_volume) ** 2,
            'obstacle': AssembleUtils.assemble_scalar(obstacle_form),
        }
    print('[solid] eval_fun: ', eval())
        
    return {
        'Jhat': Jhat,
        'solid_strain': strain,
        'solid_rho': rho,
        'eval_fun': eval,
        'record_strain': record_strain,
        'record_fluid_zone': record_fluid_zone,
        'record_surface_force': record_surface_force,
    }


fluid_result = solve_fluid_problem()
solid_result = solve_topo_solid(fluid_result)
recorder.save(0)

with start_annotation():
    Jhat = fluid_result['Jhat'] + solid_result['Jhat']

control_rho = Control(solid_result['solid_rho'])  # type: ignore
control_displacement = Control(fluid_result['displacement'])  # type: ignore
opt_problem = ReducedFunctional(Jhat, [control_rho, control_displacement])
opt_problem.derivative(adj_input=1.0, apply_riesz=False)

# ------ optimization loop ------
topo_rho = DolfinxFunction(solid_result['solid_rho'].function_space, name=f'topo_rho')  # type: ignore
topo_rho.x.array[:] = solid_result['solid_rho'].x.array.copy()  # type: ignore

with_filter=True
with_norm=True
weight_filter = DistanceBandFilter(mesh.grid, 0.05, False, True)
run_iter = 0

loss_current = opt_problem(values=[topo_rho, fluid_result['displacement']])  # type: ignore
print(f'[step/{run_iter}] loss_current: {loss_current}')

while True:
    run_iter += 1
    
    dJ_drho, dJ_dmesh = opt_problem.derivative(adj_input=1.0, apply_riesz=False)
    
    # ------ compute gradient of fluid
    dJ_dmesh = fluid_result['reg_solver'].solve(dJ_dmesh)
    dJ_dmesh = dJ_dmesh.reshape((-1, 2))
    dJ_dmesh = dJ_dmesh / np.max(np.linalg.norm(dJ_dmesh, ord=2, axis=1))
    dJ_dmesh = dJ_dmesh * -1.0
    
    dJ_dmesh_fun = dolfinx.fem.Function(deformation_space, name='dJ_dmesh')
    dJ_dmesh_fun.x.array[:] = dJ_dmesh.reshape(-1)  # type: ignore
    
    hjb_solver = HamiltonJacobiSolver(
        domain=domain,
        velocity=dJ_dmesh_fun,  # type: ignore
        phi=fluid_result['fluid_phi'],  # type: ignore
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
        diff_max = np.max(np.abs(phi_new - fluid_result['fluid_phi'].x.array))  # type: ignore
        if diff_max >= 0.1:
            break
    
    fluid_result['update_phi'](phi_new)
    recorder.update_point_data('phi', fluid_result['fluid_phi'].x.array)  # type: ignore
    
    if run_iter % 1 == 0:
        phi_new = ReinitializationSolver.solve(domain, fluid_result['fluid_phi'])  # type: ignore
        fluid_result['update_phi'](phi_new)
        recorder.update_point_data('phi_reinitialized', fluid_result['fluid_phi'].x.array)  # type: ignore

    # ------ compute gradient of solid
    if with_filter:
        dJ_drho = dJ_drho.reshape((-1, 1))
        dJ_drho = weight_filter.run(dJ_drho)
        dJ_drho = np.asarray(dJ_drho).reshape((-1))
    if with_norm:
        norm = np.max(np.abs(dJ_drho))
        dJ_drho = dJ_drho / norm            
    direction_np = dJ_drho

    new_topo_rho = topo_rho.x.array.copy() - direction_np * 0.1  # type: ignore
    new_topo_rho = np.maximum(np.minimum(new_topo_rho, 0.999), 0.001)
    topo_rho.x.array[:] = new_topo_rho  # type: ignore
    
    # ------ run eval
    loss_current = opt_problem(values=[topo_rho, fluid_result['displacement']])  # type: ignore
    
    fluid_result['record_Upw']()
    solid_result['record_strain']()
    solid_result['record_fluid_zone']()
    solid_result['record_surface_force']()
    recorder.update_point_data('rho', topo_rho.x.array)  # type: ignore
    recorder.save(run_iter)
    
    loss_dict = {}
    loss_dict.update(solid_result['eval_fun']())
    loss_dict.update(fluid_result['eval_fun']())
    log = f"[step/{run_iter}] loss_current: {loss_current}"
    for name, cost_value in loss_dict.items():
        log += f" {name}:{cost_value:.5f}"
    print(log)
    
    if run_iter > 200:
        break
