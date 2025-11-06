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
from scripts_py.dolfinx_adjoint.topology_utils.topology_filter import DistanceBandFilter

from scripts_py.auto_dolfinx.types.type_mesh import DolfinxMesh
from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.types.type_dirichletBC import DolfinxDirichletBC
from scripts_py.auto_dolfinx.blocks.block_solve import equation_solve
from scripts_py.auto_dolfinx.blocks.block_assemble import assemble_scalar
from scripts_py.auto_dolfinx.blocks.block_deform import deform
from scripts_py.auto_dolfinx.tape_utils import start_annotation


tape = Tape()
set_working_tape(tape)

xdmf_file = "/home/admin123456/Desktop/work/gmsh_test/t18/levelset_01.xdmf"

mesh = MeshObj(xdmf_file=xdmf_file, mesh_name="model")
grid = mesh.grid
domain = DolfinxMesh(mesh.domain)
mesh.domain = domain

wall_fix_marker = 10
wall_free_marker = 11
wall_force_marker = 12

point_scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
point_vector_func_space = DolfinUtils.point_vector_function_space(mesh.domain)


E, nu = 1.0, 0.3
mu: float = E / (2 * (1 + nu))
lmbda: float = E * nu / ((1 + nu) * (1 - 2 * nu))

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(2)  # type: ignore

def alpha_fun(rho: np.ndarray|dolfinx.fem.Function, alpha_max=1.0, alpha_min=1e-3, p=4.0, pow_operator=ufl.elem_pow):
    return alpha_min + pow_operator(rho, p) * (alpha_max - alpha_min)  # type: ignore


with start_annotation():
    rho = DolfinxFunction(point_scalar_func_space, name='rho')
    rho.x.array[:] = 0.0  # type: ignore
    
    strain = DolfinxFunction(point_vector_func_space, name='strain')
    strain_v = DolfinxFunction(point_vector_func_space, name='strain_v')
    
    f = dolfinx.fem.Constant(domain, np.array([0.0, 0.0]))
    T = dolfinx.fem.Constant(domain, np.array([0.0, -75.0]))
    ds = MeshUtils.define_ds(domain, facets_tag=mesh.facet_tags)

    a_form = ufl.inner(alpha_fun(rho) * sigma(strain), epsilon(strain_v)) * ufl.dx  # type: ignore
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
    
    energy_weight = 1e-6
    object_form = energy_weight * ufl.inner(alpha_fun(rho) * sigma(strain), epsilon(strain)) * ufl.dx  # type: ignore
    object_cost = assemble_scalar(object_form, domain=domain, tape=tape)

    volume_space = AssembleUtils.assemble_scalar(dolfinx.fem.Constant(domain, 1.0) * ufl.dx)  # type: ignore
    target_volume = volume_space * 0.3
    
    volume_form = rho * ufl.dx  # type: ignore
    volume_rho = assemble_scalar(volume_form, domain=domain, tape=tape)
    
    volume_weight = 1.0
    volume_loss = volume_weight * 0.5 * (volume_rho - target_volume) ** 2

    Jhat = object_cost + volume_loss

    control_rho = Control(rho)  # type: ignore
    opt_problem = ReducedFunctional(Jhat, [control_rho])


def eval():
    return {
        'object_cost': AssembleUtils.assemble_scalar(object_form),
        'volume_cost': volume_weight * 0.5 * (AssembleUtils.assemble_scalar(volume_form) - target_volume) ** 2,
    }


# ------------------------------
recorder = VtuRecorder(mesh, save_dir='/home/admin123456/Desktop/work/gmsh_test/t18/cache', name='mix')

line_search_steps = np.array([0.1, 0.05, 0.01, 0.005, 0.001, 0.00025])
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