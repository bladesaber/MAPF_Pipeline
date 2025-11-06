import dolfinx
import ufl

import numpy as np
import pyvista
from sklearn.neighbors import KDTree

from thirdparty.pyadjoint.pyadjoint import Tape, ReducedFunctional, Control
from thirdparty.pyadjoint.pyadjoint.tape import set_working_tape

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import BoundaryUtils, DolfinUtils, AssembleUtils
from scripts_py.vtu_recorder import VtuRecorder
from scripts_py.simulator.dolfin_simulator import FluidBoundary
from scripts_py.eqn_solver import LinearProblemSolver
from scripts_py.dolfinx_adjoint.topology_utils.topology_filter import DistanceBandFilter
from scripts_py.simulator.dolfin_simulator import StokeSimulator


from scripts_py.auto_dolfinx.types.type_mesh import DolfinxMesh
from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.types.type_dirichletBC import DolfinxDirichletBC
from scripts_py.auto_dolfinx.blocks.block_solve import equation_solve
from scripts_py.auto_dolfinx.blocks.block_assemble import assemble_scalar
from scripts_py.auto_dolfinx.blocks.block_deform import deform
from scripts_py.auto_dolfinx.tape_utils import start_annotation


def solve_fluid_problem():
    mesh_file = '/home/admin123456/Desktop/work/gmsh_test/t19/mix_opt_04_fluid.msh'
    xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t19/mix_opt_04_fluid.xdmf'
    # MeshUtils.msh_to_XDMF(mesh_file, xdmf_file, dim=2)

    inlet_marker = 6
    outlet_marker = 7
    wall_marker = 8

    mesh = MeshObj(xdmf_file=xdmf_file)
    simulator = StokeSimulator('stoke', mesh)
    simulator.define_equation(nu_value=1e-5)
    
    inlet_bry = FluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[100.0, 0.],
        marker=inlet_marker,
        face_center=[0.0, 2.5],
        face_norm_vector=[1.0, 0.0],
        face_radius=0.25
    )
    outlet_bry = FluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=outlet_marker
    )
    wall_bry = FluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=wall_marker
    )
    
    simulator.add_boundary(inlet_bry)
    simulator.add_boundary(outlet_bry)
    simulator.add_boundary(wall_bry)
    
    simulator.define_boundary()
    simulator.run(with_residual=True)
    simulator.save('/home/admin123456/Desktop/work/gmsh_test/t19/fluid_res.vtu')

solve_fluid_problem()


def get_fluid_boundary_force():
    fluid_xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t19/mix_opt_04_fluid.xdmf'
    mesh = MeshObj(xdmf_file=fluid_xdmf_file, mesh_name="model")
    ds = MeshUtils.define_ds(mesh.domain, facets_tag=mesh.facet_tags)
    
    point_scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
    point_vector_func_space = DolfinUtils.point_vector_function_space(mesh.domain)
    
    wall_marker = 8
    
    fluid_res = pyvista.read('/home/admin123456/Desktop/work/gmsh_test/t19/fluid_res.vtu')    
    p = dolfinx.fem.Function(point_scalar_func_space, name='p')
    p.x.array[:] = fluid_res.point_data['pressure']  # type: ignore
    
    # safe pressure interpolation    
    p.x.array[:] = np.maximum(p.x.array, np.max(p.x.array) * 0.2)  # type: ignore
        
    v = ufl.TestFunction(point_vector_func_space)
    n_vec = MeshUtils.define_facet_norm(mesh.domain)
    f_form = ufl.inner(p * n_vec, v) * ds(wall_marker)  # type: ignore
    surface_force =AssembleUtils.assemble_vec(f_form).array
    surface_force = surface_force.reshape((-1, mesh.tdim))
    
    # NOTE: multiply 10.0 just for testing
    surface_force = surface_force / np.max(np.linalg.norm(surface_force, axis=1)) * 10.0
    
    surface_indices = mesh.extract_surface_indices(wall_marker)
    
    return {
        'surface_points': mesh.grid.points[surface_indices],
        'surface_force': surface_force[surface_indices],
        'fluid_points': mesh.grid.points
    }
    

fluid_dict = get_fluid_boundary_force()

# -------------------------------------------------------------------
tape = Tape()
set_working_tape(tape)

mesh_file = "/home/admin123456/Desktop/work/gmsh_test/t19/mix_opt_04_solid.msh"
xdmf_file = "/home/admin123456/Desktop/work/gmsh_test/t19/mix_opt_04_solid.xdmf"
MeshUtils.msh_to_XDMF(mesh_file, xdmf_file, dim=2)

mesh = MeshObj(xdmf_file=xdmf_file, mesh_name="model")
grid = mesh.grid
domain = DolfinxMesh(mesh.domain)
mesh.domain = domain

wall_fix_marker = 6
wall_free_marker = 7

point_scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
point_vector_func_space = DolfinUtils.point_vector_function_space(mesh.domain)

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
    
    # ------ define obstacle function ------
    tree = KDTree(mesh.grid.points)
    obstacle = DolfinxFunction(point_scalar_func_space, name='obstacle')
    obstacle.x.array[:] = 0.0  # type: ignore
    
    idxs_list = tree.query_radius(fluid_dict['fluid_points'], r=0.075, return_distance=False)
    for mesh_idxs in idxs_list:
        obstacle.x.array[np.array(mesh_idxs)] = 1.0  # type: ignore
    mesh.grid.point_data['obstacle'] = obstacle.x.array  # type: ignore
    # --------------------------------------
    
    rho = DolfinxFunction(point_scalar_func_space, name='rho')
    rho.x.array[:] = 1e-2  # type: ignore
    rho.x.array[np.where(obstacle.x.array == 0.0)[0]] = 0.5  # type: ignore
    mesh.grid.point_data['rho'] = rho.x.array  # type: ignore
    
    # ------ define force function ------
    inner_idxs = np.where(obstacle.x.array == 0.0)[0]  # type: ignore
    inner_points = mesh.grid.points[inner_idxs]
    
    tree = KDTree(inner_points)
    idxs_list = tree.query(fluid_dict['surface_points'], k=1, return_distance=False)
    idxs_list = np.concatenate(idxs_list, axis=-1)
    inner_surface_indices = inner_idxs[idxs_list]
    
    force_np = np.zeros((mesh.grid.points.shape[0], mesh.tdim))
    force_np[inner_surface_indices] = fluid_dict['surface_force']
    
    f = DolfinxFunction(point_vector_func_space, name='f')
    f.x.array[:] = force_np.reshape(-1)  # type: ignore
    
    # debug: visualize force
    f_np = f.x.array.reshape((-1, mesh.tdim))  # type: ignore
    if f_np.shape[1] == 2:
        f_np = np.concatenate([f_np, np.zeros((f_np.shape[0], 1))], axis=1)
    mesh.grid.point_data['f'] = f_np
    # ---------------------------------------

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
    
    energy_weight = 1.0
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


def eval():
    return {
        'object_cost': AssembleUtils.assemble_scalar(object_form),
        'volume_cost': volume_weight * 0.5 * (AssembleUtils.assemble_scalar(volume_form) - target_volume) ** 2,
        'obstacle_cost': AssembleUtils.assemble_scalar(obstacle_form),
    }

# ------------------------------
recorder = VtuRecorder(mesh, save_dir='/home/admin123456/Desktop/work/gmsh_test/t19/cache', name='mix')

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
    line_search_start_index = np.max(last_step_index - 1, 0)
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