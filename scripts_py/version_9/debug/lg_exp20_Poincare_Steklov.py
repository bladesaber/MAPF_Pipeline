"""
Ref: https://github.com/sblauth/cashocs/tree/main/demos/documented/shape_optimization/shape_poisson
"""

import numpy as np
import os
import dolfinx
import ufl
import ctypes

from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_state_problem, create_shape_problem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.remesh_helper import MeshDeformationRunner
from scripts_py.version_9.dolfinx_Grad.optimizer_utils import CostConvergeHandler

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/possion_shape_2'
model_xdmf_file = os.path.join(proj_dir, 'model.xdmf')

# ------ create xdmf
MeshUtils.msh_to_XDMF(
    name='model', msh_file=os.path.join(proj_dir, 'model.msh'), output_file=model_xdmf_file, dim=2
)
# ------

domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=model_xdmf_file, mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)
grid = VisUtils.convert_to_grid(domain)
tdim = domain.topology.dim
fdim = tdim - 1

state_problems = []

# ------ Define State Problem 1
V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
u1 = dolfinx.fem.Function(V, name='state_1')
v1 = dolfinx.fem.Function(V, name='adjoint_1')
bry_free_markers = [3]
bry_fixed_markers = []

"""
Must Use Coordinate, The x of interpolate function of class(dolfinx.fem.Function) means the coordinate of Mesh
"""
coodr = MeshUtils.define_coordinate(domain)
f_exp = 2.5 * np.power(coodr[0] + 0.4 - np.power(coodr[1], 2), 2) + np.power(coodr[0], 2) + np.power(coodr[1], 2) - 1

# vis_f = dolfinx.fem.Function(V, name='vis_f')
# vis_f.interpolate(UFLUtils.create_expression(f_exp, V))
# VisUtils.show_scalar_res_vtk(grid, 'vis_f', vis_f)

F1_form = ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx - f_exp * v1 * ufl.dx

bcs_info = []
for marker in bry_free_markers:
    bc_dofs = MeshUtils.extract_entity_dofs(V, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker))
    bc: dolfinx.fem.DirichletBC = dolfinx.fem.dirichletbc(0.0, bc_dofs, V)
    bcs_info.append((bc, V, bc_dofs, 0.0))

state_problem_1 = create_state_problem(
    name='state_1', F_form=F1_form, state=u1, adjoint=v1, is_linear=True,
    bcs_info=bcs_info,
    state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
)
state_problems.append(state_problem_1)

state_system = StateProblem(state_problems)

# ------ Define Control problem
control_problem = create_shape_problem(
    domain=domain,
    bcs_info=[],
    lambda_lame=0.0,
    damping_factor=0.0,
    gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'ksp'}
)

# ------ Define Cost Function
cost1_form = u1 * ufl.dx
cost1_fun = IntegralFunction(cost1_form)

# ------ Define Optimal Problem
opt_problem = OptimalShapeProblem(
    state_system=state_system,
    shape_problem=control_problem,
    shape_regulariztions=ShapeRegularization(regularization_list=[]),
    cost_functional_list=[cost1_fun],
    scalar_product=None,
    scalar_product_method={
        'method': 'Poincare-Steklov operator',
        # 'lambda_lame': 1.0,
        # 'damping_factor': 1.0,
        'cell_tags': cell_tags,
        'facet_tags': facet_tags,
        'bry_free_markers': bry_free_markers,
        'bry_fixed_markers': bry_fixed_markers,
        'use_inhomogeneous': False,
        'inhomogeneous_exponent': 1.0,
        'update_inhomogeneous': False
    }
)

# opt_problem.compute_state_problem(domain.comm)
# VisUtils.show_scalar_res_vtk(grid, 'u1', u1)

deformation_handler = MeshDeformationRunner(
    domain=domain,
    volume_change=0.1,
    quality_measures={
        'max_angle': {
            'measure_type': 'max',
            'tol_upper': 150.0,
            'tol_lower': 0.0
        },
        'min_angle': {
            'measure_type': 'min',
            'tol_upper': 180.0,
            'tol_lower': 30.0
        }
    }
)

orig_loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
cur_loss_ctype = ctypes.c_double(orig_loss)

cost_converger = CostConvergeHandler(stat_num=25, warm_up_num=25, tol=1e-2, scale=1.0 / orig_loss)


def detect_cost_valid_func(tol_rho=0.0):
    loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
    is_valid = loss < cur_loss_ctype.value + np.abs(cur_loss_ctype.value) * tol_rho
    return is_valid


step = 0
while True:
    step += 1

    shape_grad: dolfinx.fem.Function = opt_problem.compute_gradient(domain.comm)
    # VisUtils.show_scalar_res_vtk(grid, 'u_opt', u1)

    shape_grad_np = shape_grad.x.array
    # shape_grad_np = shape_grad_np / np.linalg.norm(shape_grad_np, ord=2)
    shape_grad_np = shape_grad_np * -1.0

    displacement_np = np.zeros(domain.geometry.x.shape)
    displacement_np[:, :tdim] = shape_grad_np.reshape((-1, tdim))

    if step % 50 == 0:
        # dJ = dolfinx.fem.Function(shape_grad.function_space)
        # dJ.x.array[:] = shape_grad_np.reshape(-1)
        # VisUtils.show_vector_res_vtk(grid, dJ, dim=2, with_wrap=True)
        # VisUtils.show_scalar_res_vtk(grid, 'u1', u1)
        pass

    # MeshUtils.move(domain, displacement_np * 0.2)
    # success_flag, info = deformation_handler.move_mesh(displacement_np * 0.2)

    # print(f"[Step {step}]")
    success_flag, step_size = deformation_handler.move_mesh_by_line_search(
        displacement_np, max_iter=10, init_stepSize=1.0, stepSize_lower=1e-3,
        detect_cost_valid_func=detect_cost_valid_func
    )
    # print()

    if success_flag:
        opt_problem.update_update_scalar_product(with_debug=False)
        loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=False)
        cur_loss_ctype.value = loss

        print(f"[Step {step}] loss:{loss}")

        is_converge = cost_converger.is_converge(loss)
        if is_converge:
            break

        if step > 200:
            break

    else:
        break

VisUtils.show_scalar_res_vtk(grid, 'u_opt', u1)