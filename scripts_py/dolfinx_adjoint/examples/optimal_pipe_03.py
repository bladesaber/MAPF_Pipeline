"""
reference: https://github.com/dolfin-adjoint/dolfin-adjoint/tree/main/examples/shape-optimization-pipe
"""

import dolfinx
import ufl
from ufl import inner, grad, div
from mpi4py import MPI
from basix.ufl import element, mixed_element

import numpy as np

from scripts_py.dolfinx_adjoint.base_type import (
    StateProblemDataBase,
    ShapeProblemDataBase,
    BoundaryCell,
)
from scripts_py.dolfinx_adjoint.problem_state import StateProblem
from scripts_py.dolfinx_adjoint.cost_functions import IntegralFunctional
from scripts_py.dolfinx_adjoint.solver_optimize import ShapeOptimizer
from scripts_py.dolfinx_adjoint.regularization_shape import ShapeRegularizationPipe, VolumeRegularization
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils
from scripts_py.dolfinx_adjoint.vtu_recorder import VtuRecorder


msh_file = "/home/admin123456/Desktop/work/gmsh_test/test/t3/optimal_pipe_03.msh"
xdmf_file = "/home/admin123456/Desktop/work/gmsh_test/test/t3/optimal_pipe_03.xdmf"
# MeshUtils.msh_to_XDMF(name='model', msh_file=msh_file, xdmf_file=xdmf_file, dim=2)

mesh = MeshObj(xdmf_file=xdmf_file, mesh_name="model")
inlet_marker = 14
outlet_marker = 15
boundary_fixed_markers = 16
boundary_free_markers = 17

# --- 1.1 define function space ---
velocity_element = element("Lagrange", mesh.cell_types(), degree=2, shape=(mesh.tdim,))
pressure_element = element("Lagrange", mesh.cell_types(), degree=1)
mix_element = mixed_element([velocity_element, pressure_element])

W = dolfinx.fem.functionspace(mesh.domain, mix_element)
W0, W1 = W.sub(0), W.sub(1)
V, V_to_W_dofs = W0.collapse()
Q, Q_to_W_dofs = W1.collapse()

vector_element = element("CG", mesh.cell_types(), degree=1, shape=(mesh.tdim,))
scalar_element = element("CG", mesh.cell_types(), degree=1)

up = dolfinx.fem.Function(W, name="state_1")
u, p = ufl.split(up)  # type: ignore please don't use up.split()
vq = dolfinx.fem.Function(W, name="adjoint_1")
v, q = ufl.split(vq)  # type: ignore please don't use vq.split()
f = dolfinx.fem.Constant(mesh.domain, np.zeros(mesh.tdim))

nu = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh.domain, scalar_element))
nu.x.array[:] = 1.0 / 400.0  # type: ignore

F_form = (
    nu * inner(grad(u), grad(v)) * ufl.dx  # type: ignore
    - p * div(v) * ufl.dx
    + q * div(u) * ufl.dx
    - inner(f, v) * ufl.dx
)

state_database_1 = StateProblemDataBase(
    name="state_problem_1",
    F_form=F_form,
    state=up,
    adjoint=vq,
    is_linear=True,
    state_ksp_option={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
    },
    adjoint_ksp_option={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
    },
)

# --- 2.1 define boundary condition ---
bc_fixed_state = BoundaryCell.define_bry_cell(
    name="bc_fixed_state",
    bc_func_space=W0,
    value_func_space=V,
    dofs=MeshUtils.extract_entity_dofs(
        (W0, V),
        mesh.fdim,
        MeshUtils.extract_entity_indices(
            mesh.domain,
            style="facet",
            marker=boundary_fixed_markers,
            mesh_tags=mesh.facet_tags,
        ),
    ),
    bc_value=dolfinx.fem.Function(V, name="bc_fixed_state"),
    is_mixed_element=True,
)

bc_free_state = BoundaryCell.define_bry_cell(
    name="bc_free_state",
    bc_func_space=W0,
    value_func_space=V,
    dofs=MeshUtils.extract_entity_dofs(
        (W0, V),
        mesh.fdim,
        MeshUtils.extract_entity_indices(
            mesh.domain,
            style="facet",
            marker=boundary_free_markers,
            mesh_tags=mesh.facet_tags,
        ),
    ),
    bc_value=dolfinx.fem.Function(V, name="bc_free_state"),
    is_mixed_element=True,
)

def inflow_velocity_exp(x):
    num = x.shape[1]
    values = np.zeros((mesh.tdim, num))
    values[0] = 6 * (1 - x[1]) * x[1]
    return values

bc_inlet_state = BoundaryCell.define_bry_cell(
    name="bc_inlet_state",
    bc_func_space=W0,
    value_func_space=V,
    dofs=MeshUtils.extract_entity_dofs(
        (W0, V),
        mesh.fdim,
        MeshUtils.extract_entity_indices(
            mesh.domain, style="facet", marker=inlet_marker, mesh_tags=mesh.facet_tags
        ),
    ),
    bc_value=inflow_velocity_exp,
    is_mixed_element=True,
)

bc_outlet_state = BoundaryCell.define_bry_cell(
    name="bc_outlet_state",
    bc_func_space=W1,
    value_func_space=Q,
    dofs=MeshUtils.extract_entity_dofs(
        (W1, Q),
        mesh.fdim,
        MeshUtils.extract_entity_indices(
            mesh.domain, style="facet", marker=outlet_marker, mesh_tags=mesh.facet_tags
        ),
    ),
    bc_value=dolfinx.fem.Function(Q, name='bc_outlet_state'),
    is_mixed_element=True,
)

state_database_1.add_bc(bc_fixed_state)
state_database_1.add_bc(bc_free_state)
state_database_1.add_bc(bc_inlet_state)
state_database_1.add_bc(bc_outlet_state)

# --- 3.1 define control problem ---
control_database_1 = ShapeProblemDataBase(
    domain=mesh.domain,
    gradient_ksp_option={
        'ksp_type': 'preonly', 
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps", 
        "mat_mumps_icntl_24": 1
    }
)

point_vector_func_space = DolfinUtils.point_vector_function_space(mesh.domain)

bc_fixed_control = BoundaryCell.define_bry_cell(
    name="bc_fixed_control",
    bc_func_space=point_vector_func_space,
    value_func_space=point_vector_func_space,
    dofs=MeshUtils.extract_entity_dofs(
        point_vector_func_space,
        mesh.fdim,
        MeshUtils.extract_entity_indices(
            mesh.domain,
            style="facet",
            marker=boundary_fixed_markers,
            mesh_tags=mesh.facet_tags,
        ),
    ),
    bc_value=dolfinx.fem.Function(point_vector_func_space, name="bc_fixed_control"),
    is_mixed_element=False,
)

bc_inlet_control = BoundaryCell.define_bry_cell(
    name="bc_inlet_control",
    bc_func_space=point_vector_func_space,
    value_func_space=point_vector_func_space,
    dofs=MeshUtils.extract_entity_dofs(
        point_vector_func_space,
        mesh.fdim,
        MeshUtils.extract_entity_indices(
            mesh.domain,
            style="facet",
            marker=inlet_marker,
            mesh_tags=mesh.facet_tags,
        ),
    ),
    bc_value=dolfinx.fem.Function(point_vector_func_space, name="bc_inlet_control"),
    is_mixed_element=False,
)

bc_outlet_control = BoundaryCell.define_bry_cell(
    name="bc_outlet_control",
    bc_func_space=point_vector_func_space,
    value_func_space=point_vector_func_space,
    dofs=MeshUtils.extract_entity_dofs(
        point_vector_func_space,
        mesh.fdim,
        MeshUtils.extract_entity_indices(
            mesh.domain,
            style="facet",
            marker=outlet_marker,
            mesh_tags=mesh.facet_tags,
        ),
    ),
    bc_value=dolfinx.fem.Function(point_vector_func_space, name="bc_outlet_control"),
    is_mixed_element=False,
)

control_database_1.add_bc(bc_fixed_control)
control_database_1.add_bc(bc_inlet_control)
control_database_1.add_bc(bc_outlet_control)

# --- 3.2 define cost function ---
cost1_fun = IntegralFunctional(
    name='cost1', 
    domain=mesh.domain, 
    cost_form=nu * inner(grad(u), grad(u)) * ufl.dx,  # type: ignore
    weight=1.0
)

shape_regularization_pipeline = ShapeRegularizationPipe([
    VolumeRegularization(
        name='volume_reg',
        shape_database=control_database_1,
        weight=0.2,
        target_volume_rho=1.0,
    )
])

opt_problem = ShapeOptimizer(
    state_problem=StateProblem([state_database_1]),
    shape_database=control_database_1,
    cost_functional_list=[cost1_fun],
    shape_regularization_list=shape_regularization_pipeline,
    scalar_product_kwargs={"method": "default"}
)

recorder = VtuRecorder(mesh, save_dir='/home/admin123456/Desktop/work/gmsh_test/test/t3/res')

def record_Up(step):
    vel_ref = dolfinx.fem.Function(point_vector_func_space, name='vel_ref')
    vel_ref.interpolate(up.sub(0).collapse())  # type: ignore
    p_ref = up.sub(1).collapse()  # type: ignore

    vel_ref_np = vel_ref.x.array.copy().reshape((-1, mesh.tdim))  # type: ignore
    if vel_ref_np.shape[1] == 2:
        vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)

    recorder.update_point_data('U', vel_ref_np)
    recorder.update_point_data('p', p_ref.x.array.copy())
    recorder.save(step)

record_Up(0)
loss_current = opt_problem.evaluate_cost_functional(mesh.domain.comm, update_state=True)
print(f"step/{0} loss:{loss_current}")

line_search_step = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
is_solved, step = False, 0
while True:
    step += 1
    if step > 1:
        loss = opt_problem.evaluate_cost_functional(mesh.domain.comm, update_state=True)
        loss_residual = loss_current - loss
        if np.abs(loss_residual) < 1e-6:
            print(f"step/{step} loss:{loss_current} residual:{loss_residual:.8f}.")
            is_solved = True
            break
        else:
            loss_current = loss
            print(f"step/{step} loss:{loss_current} residual:{loss_residual:.8f}.")

    report_dict = opt_problem.compute_gradient(
        mesh.domain.comm,
        state_kwargs={
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        adjoint_kwargs={
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        gradient_kwargs={
            'mat_assemble_method': 'identity_row',
            'with_residual': False
        },
        scalar_product_kwargs={
            'with_residual': False
        },
        update_state=True,
        update_adjoint=True
    )
    is_success = report_dict['is_success']
    if not is_success:
        print(f'[Error]: {report_dict["log"]}')
        is_solved = False
        break

    dire_np: np.ndarray = report_dict['shape_grad'].x.array.copy()
    dire_np = dire_np.reshape((-1, mesh.tdim))    
    dire_np = dire_np / np.max(np.abs(np.linalg.norm(dire_np, ord=2, axis=1)))
    
    find_best_step = False
    for step_length in line_search_step:
        displacement_np = dire_np * step_length * -1.0
        mesh.deform(displacement_np)
        loss_new = opt_problem.evaluate_cost_functional(mesh.domain.comm, update_state=True)
        if loss_new < loss_current:
            find_best_step = True
            record_Up(step)
            break
        else:
            mesh.deform(displacement_np * -1.0)  # recover mesh
        
    if not find_best_step:
        print(f"[Warning]: line search fail.")
        is_solved = False
        break

