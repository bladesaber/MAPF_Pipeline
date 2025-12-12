import dolfinx
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


mesh_file = '/home/admin123456/Desktop/work/gmsh_test/t17/mix_opt_03.msh'
xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t17/mix_opt_03.xdmf'
# MeshUtils.msh_to_XDMF(mesh_file, xdmf_file, dim=2)

mesh = MeshObj(xdmf_file=xdmf_file, mesh_name="model")
grid = mesh.grid
domain = DolfinxMesh(mesh.domain)
mesh.domain = domain

inlet_marker = 10
outlet_marker = 11
wall_fix_marker = 12
wall_free_marker = 13

inlet_face_center = np.array([0.0, 2.5])
inlet_face_norm_vector = np.array([1.0, 0.0])
inlet_face_radius = 0.25

point_scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
point_vector_func_space = DolfinUtils.point_vector_function_space(mesh.domain)

tape = Tape()
set_working_tape(tape)


def solve_fluid_problem():
    velocity_element = element("Lagrange", mesh.cell_types(), degree=2, shape=(mesh.tdim,))
    pressure_element = element("Lagrange", mesh.cell_types(), degree=1)
    mix_element = mixed_element([velocity_element, pressure_element])

    W = dolfinx.fem.functionspace(mesh.domain, mix_element)
    W0, W1 = W.sub(0), W.sub(1)
    V, V_to_W_dofs = W0.collapse()
    Q, Q_to_W_dofs = W1.collapse()

    bcs: list[dolfinx.fem.DirichletBC] = []
        
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
        BoundaryUtils.define_dirichlet_bc(
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
        BoundaryUtils.define_dirichlet_bc(
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
        BoundaryUtils.define_dirichlet_bc(
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
        BoundaryUtils.define_dirichlet_bc(
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

    def init_levelset_function(pos: np.ndarray):
        init_path = np.array([
            np.array([0.0, 2.5]),
            np.array([5.0, 2.5]),
        ])
        detail_path = []
        for i in range(init_path.shape[0] - 1):
            step = int(np.linalg.norm(init_path[i + 1] - init_path[i]) / 0.1)
            detail_path.append(np.linspace(init_path[i], init_path[i + 1], step))
        detail_path = np.concatenate(detail_path, axis=0)
                    
        tree = KDTree(detail_path)
        dists_list, _ = tree.query(pos.T[:, :mesh.tdim], k=1, return_distance=True)
        dists_np = np.concatenate(dists_list, axis=-1)
        return dists_np - inlet_face_radius
        
    phi = DolfinxFunction(point_scalar_func_space, name='levelset_phi')
    phi.interpolate(init_levelset_function)  # type: ignore

    uh = ufl.TrialFunction(W)
    u, p = ufl.split(uh)  # type: ignore
    vq = ufl.TestFunction(W)
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
    
    uh = DolfinxFunction(W, name='state')
    res_dict = LinearProblemSolver.solve_fea_petsc(
        comm=domain.comm,
        uh=uh,  # type: ignore
        a_form=ufl.lhs(F_form),  # type: ignore
        l_form=ufl.rhs(F_form),
        bcs=bcs,
        ksp_option={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1,
        },
        mat_assemble_method='lift',
        with_residual=False,
    )
    if not res_dict['is_solved']:
        raise ValueError(f"Failed to solve fluid problem.")

    # ------ debug visualize ------
    vel_ref = dolfinx.fem.Function(point_vector_func_space, name='vel_ref')
    vel_ref.interpolate(uh.sub(0).collapse())  # type: ignore
    p_ref = uh.sub(1).collapse()  # type: ignore

    vel_ref_np = vel_ref.x.array.copy().reshape((-1, mesh.tdim))  # type: ignore
    if vel_ref_np.shape[1] == 2:
        vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)

    mesh.grid.point_data['U'] = vel_ref_np  # type: ignore
    mesh.grid.point_data['p'] = p_ref.x.array.copy()  # type: ignore
    mesh.grid.point_data['phi'] = phi.x.array  # type: ignore
    mesh.grid.save('/home/admin123456/Desktop/work/gmsh_test/t17/fluid_result.vtu')
    # ------------------- debug visualize -------------------
    
    
# solve_fluid_problem()


def extract_surface_force():
    fluid_grid = pyvista.read('/home/admin123456/Desktop/work/gmsh_test/t17/fluid_result.vtu')

    p = dolfinx.fem.Function(point_scalar_func_space, name='p')
    p.x.array[:] = fluid_grid.point_data['p']  # type: ignore
    
    phi = dolfinx.fem.Function(point_scalar_func_space, name='phi')
    phi.x.array[:] = fluid_grid.point_data['phi']  # type: ignore
    
    # using safe distance for avoiding overlap with fluid zone
    safe_dist = 0.075
    surface_phi = dirac_delta(phi - safe_dist, exp_operator=ufl.exp, beta=200.0)  # type: ignore
    surface_nvec = ufl.grad(phi)
    
    # NOTE: very small here
    surface_force = surface_phi * surface_nvec * (p + 0.1)  # type: ignore
    
    # ------ debug visualize ------
    v = ufl.TestFunction(point_vector_func_space)
    surface_f_form = ufl.inner(surface_force, v) * ufl.dx
    surface_f_np = AssembleUtils.assemble_vec(surface_f_form).array
    surface_f_np = surface_f_np.reshape((-1, mesh.tdim))    
    if surface_f_np.shape[1] == 2:
        surface_f_np = np.concatenate([surface_f_np, np.zeros((surface_f_np.shape[0], 1))], axis=1)
    fluid_grid.point_data['surface_force'] = surface_f_np  # type: ignore
    
    fluid_zone_form = heaviside_step_fun(-phi, rho_max=1.0, rho_min=0.0, exp_operator=ufl.exp)  # type: ignore
    fluid_zone = dolfinx.fem.Function(point_scalar_func_space, name='fluid_zone')
    fluid_zone.interpolate(DolfinUtils.create_expression(fluid_zone_form, point_scalar_func_space))  # type: ignore
    fluid_grid.point_data['fluid_zone'] = fluid_zone.x.array  # type: ignore
    
    fluid_grid.save('/home/admin123456/Desktop/work/gmsh_test/t17/surface_result.vtu')


# extract_surface_force()


def solve_topo_solid():
    E, nu = 1.0, 0.3
    mu: float = E / (2 * (1 + nu))
    lmbda: float = E * nu / ((1 + nu) * (1 - 2 * nu))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(2)  # type: ignore

    def alpha_fun(rho: np.ndarray|dolfinx.fem.Function, alpha_max=1.0, alpha_min=1e-3, p=5.0, pow_operator=ufl.elem_pow):
        return alpha_min + pow_operator(rho, p) * (alpha_max - alpha_min)  # type: ignore

    surface_grid = pyvista.read('/home/admin123456/Desktop/work/gmsh_test/t17/surface_result.vtu')
    obstacle = DolfinxFunction(point_scalar_func_space, name='obstacle')
    obstacle.x.array[:] = surface_grid.point_data['fluid_zone']  # type: ignore
    
    f = DolfinxFunction(point_vector_func_space, name='f')
    f.x.array[:] = surface_grid.point_data['surface_force'][:, :mesh.tdim].reshape(-1)  # type: ignore
    
    for key in surface_grid.point_data.keys():
        mesh.grid.point_data[key] = surface_grid.point_data[key]  # type: ignore

    with start_annotation():
        strain = DolfinxFunction(point_vector_func_space, name='strain')
        strain_v = DolfinxFunction(point_vector_func_space, name='strain_v')
        T = dolfinx.fem.Constant(domain, np.array([0.0, 0.0]))
        
        rho = DolfinxFunction(point_scalar_func_space, name='rho')
        rho.x.array[:] = 1e-2  # type: ignore
        rho.x.array[np.where(obstacle.x.array < 1e-6)[0]] = 0.5  # type: ignore
        mesh.grid.point_data['rho'] = rho.x.array  # type: ignore

        a_form = ufl.inner(alpha_fun(rho) * sigma(strain), epsilon(strain_v)) * ufl.dx  # type: ignore
        L_form = ufl.dot(f, strain_v) * ufl.dx + ufl.dot(T, strain_v) * ufl.ds

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
        
        energy_weight = 5e8
        object_form = energy_weight * ufl.inner(alpha_fun(rho) * sigma(strain), epsilon(strain)) * ufl.dx  # type: ignore
        object_cost = assemble_scalar(object_form, domain=domain, tape=tape)

        volume_space = (
            AssembleUtils.assemble_scalar(dolfinx.fem.Constant(domain, 1.0) * ufl.dx)  # type: ignore
            - AssembleUtils.assemble_scalar(obstacle * ufl.dx)
        )
        target_volume = volume_space * 0.25
        
        volume_form = rho * ufl.dx  # type: ignore
        volume_rho = assemble_scalar(volume_form, domain=domain, tape=tape)
        
        volume_weight = 1.0
        volume_loss = volume_weight * 0.5 * (volume_rho - target_volume) ** 2
        
        obstacle_weight = 1e4
        # obstacle_form = obstacle_weight * ufl.inner(obstacle, rho) * ufl.dx  # type: ignore
        obstacle_form = obstacle_weight * ufl.inner(
            ufl.conditional(ufl.ge(obstacle, 0.75), obstacle, 0.0), 
            ufl.conditional(ufl.ge(rho, 0.75), rho, 0.0)
        ) ** 2 * ufl.dx  # type: ignore
        obstacle_loss = assemble_scalar(obstacle_form, domain=domain, tape=tape)

        Jhat = object_cost + volume_loss + obstacle_loss

        control_rho = Control(rho)  # type: ignore
        opt_problem = ReducedFunctional(Jhat, [control_rho])
        
        strain_np = strain.x.array.reshape((-1, mesh.tdim))  # type: ignore
        if strain_np.shape[1] == 2:
            strain_np = np.concatenate([strain_np, np.zeros((strain_np.shape[0], 1))], axis=1)
        mesh.grid.point_data['strain'] = strain_np  # type: ignore
        mesh.grid.save('/home/admin123456/Desktop/work/gmsh_test/t17/strain_result.vtu')
    
    def eval():
        return {
            'object_cost': AssembleUtils.assemble_scalar(object_form),
            'volume_cost': volume_weight * 0.5 * (AssembleUtils.assemble_scalar(volume_form) - target_volume) ** 2,
            'obstacle_cost': AssembleUtils.assemble_scalar(obstacle_form),
        }
    
    # ------------------------------
    recorder = VtuRecorder(mesh, save_dir='/home/admin123456/Desktop/work/gmsh_test/t17/cache', name='mix')
    line_search_steps = np.array([0.1, 0.05, 0.01, 0.005, 0.001])
    max_iteration=500

    is_solved, step = False, 0
    loss_current, loss_last = 0.0, 0.0
    step_length_last = line_search_steps[0]
    with_filter=True
    with_norm=True
    weight_filter = DistanceBandFilter(mesh.grid, 0.2, False, True)

    topo_rho = DolfinxFunction(rho.function_space, name=f'topo_rho')  # type: ignore
    topo_rho.x.array[:] = rho.x.array.copy()  # type: ignore
            
    loss_dict = eval()
    loss_current = np.sum(list(loss_dict.values()))
    
    while True:
        step += 1
                    
        if step > 1:
            loss_dict = eval()
            log = f"[step/{step} loss:{loss_current} last_step_length:{step_length_last:.3f}]"
            for name, cost_value in loss_dict.items():
                log += f" {name}:{cost_value:.5f}"
            print(log)
                    
            loss_residual = loss_current - loss_last
            if np.abs(loss_residual) < 1e-8:
                is_solved = True
                print('solve success')
                break
            
        # ------ compute gradient
        adj_input = np.minimum(np.maximum(loss_current * 0.1, 0.01), 1e2)
        dJ_drho = opt_problem.derivative(adj_input=float(adj_input))[0]
        if with_filter:
            dJ_drho = dJ_drho.reshape((-1, 1))
            dJ_drho = weight_filter.run(dJ_drho)
            dJ_drho = np.asarray(dJ_drho).reshape((-1))
        if with_norm:
            norm = np.max(np.abs(dJ_drho))
            dJ_drho = dJ_drho / norm            
        direction_np = dJ_drho
                        
        find_best_step = False
        topo_rho_origin: np.array = topo_rho.x.array.copy()  # type: ignore
                
        last_step_index = np.where(line_search_steps == step_length_last)[0][0]
        line_search_start_index = np.maximum(last_step_index - 1, 0)
        for i in range(line_search_start_index, line_search_steps.shape[0]):
            step_length = line_search_steps[i]
                
            new_topo_rho = topo_rho_origin - direction_np * step_length
            new_topo_rho = np.maximum(np.minimum(new_topo_rho, 0.999), 0.001)
            topo_rho.x.array[:] = new_topo_rho  # type: ignore
                    
            loss_new = opt_problem(values=[topo_rho])  # type: ignore
            if loss_new < loss_current:
                find_best_step = True
                loss_last = loss_current
                loss_current = loss_new
                step_length_last = step_length
                            
                # ------ record Upw ------
                recorder.update_point_data('rho', topo_rho.x.array)  # type: ignore
                strain_np = strain.x.array.copy().reshape((-1, mesh.tdim))  # type: ignore
                if strain_np.shape[1] == 2:
                    strain_np = np.concatenate([strain_np, np.zeros((strain_np.shape[0], 1))], axis=1)
                recorder.update_point_data('strain', strain_np)  # type: ignore
                recorder.save(step)
                
                break
                    
        if not find_best_step:
            print(f"[Warning]: line search fail.")
            is_solved = False
            break
                    
        if step > max_iteration:
            print(f"[Warning]: max iteration reached.")
            is_solved = False
            break
    
        
solve_topo_solid()

