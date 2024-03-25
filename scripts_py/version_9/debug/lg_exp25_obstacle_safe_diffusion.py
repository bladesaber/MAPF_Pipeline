import ctypes
import os
import shutil
from copy import deepcopy
import dolfinx
import numpy as np
import pyvista
import ufl
from mpi4py import MPI

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, UFLUtils
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization, \
    VolumeRegularization
from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_state_problem, create_shape_problem
from scripts_py.version_9.dolfinx_Grad.optimizer_utils import CostConvergeHandler
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder, XDMFRecorder
from scripts_py.version_9.dolfinx_Grad.remesh_helper import MeshDeformationRunner
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/safe_diffusion_01'


def get_background_map(coord):
    return (np.power(coord[0] - 0.5, 2) + np.power(coord[1] + 1.5, 2) - 1) * 3.0


def get_obstacle_map(coord):
    return np.power(coord[0] - 0.0, 2) + np.power(coord[1] + 0.5, 2) - 0.09


def compute_sigmoid(func):
    return 5.0 / (1.0 + ufl.exp(20.0 * func))


# ------ Create Background Map and Obstacle Map
background_domain = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, points=[np.array([-2, -2]), np.array([2, 2])], n=[60, 60]
)
V = dolfinx.fem.FunctionSpace(background_domain, ("Lagrange", 1))

f_exp = get_background_map(MeshUtils.define_coordinate(background_domain))
background_fun = dolfinx.fem.Function(V, name='vis_f')
background_fun.interpolate(UFLUtils.create_expression(f_exp, V))
background_recorder = XDMFRecorder(os.path.join(proj_dir, 'background.xdmf'))
background_recorder.write_mesh(background_domain)
background_recorder.write_function(background_fun, 0)

shape_map = get_obstacle_map(MeshUtils.define_coordinate(background_domain))
shape_map = compute_sigmoid(shape_map)
shape_func = dolfinx.fem.Function(V, name='shape_func')
shape_func.interpolate(UFLUtils.create_expression(shape_map, V))
shape_map_recorder = XDMFRecorder(os.path.join(proj_dir, 'shape_map.xdmf'))
shape_map_recorder.write_mesh(background_domain)
shape_map_recorder.write_function(shape_func, step=0)
# --------------------------------------------------------

model_xdmf_file = os.path.join(proj_dir, 'model.xdmf')
MeshUtils.msh_to_XDMF(name='model', msh_file=os.path.join(proj_dir, 'model.msh'), output_file=model_xdmf_file, dim=2)
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=model_xdmf_file, mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)
grid = VisUtils.convert_to_grid(domain)
# grid_copy = deepcopy(grid)

bry_markers = [6]
bry_fixed_markers = []
bry_free_markers = [6]
tdim = domain.topology.dim
fdim = tdim - 1

V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
u = dolfinx.fem.Function(V, name='state')
v = dolfinx.fem.Function(V, name='adjoint')

bcs_info = []
for marker in bry_markers:
    bc_value = 1.0
    bc_dof = MeshUtils.extract_entity_dofs(V, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker))
    bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, V)
    bcs_info.append((bc, V, bc_dof, bc_value))

f_exp = get_background_map(MeshUtils.define_coordinate(domain))
# vis_f = dolfinx.fem.Function(V, name='vis_f')
# vis_f.interpolate(UFLUtils.create_expression(f_exp, V))
# VisUtils.show_scalar_res_vtk(grid, 'vis_f', vis_f)

F1_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f_exp * v * ufl.dx

state_problems = []
state_problem_1 = create_state_problem(
    name='state_1', F_form=F1_form, state=u, adjoint=v, is_linear=True, bcs_info=bcs_info,
    state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
)
state_problems.append(state_problem_1)
state_system = StateProblem(state_problems)

coordinate_space = domain.ufl_domain().ufl_coordinate_element()
V_S = dolfinx.fem.FunctionSpace(domain, coordinate_space)

bcs_info = []
for marker in bry_fixed_markers:
    bc_value = dolfinx.fem.Function(V_S, name=f"fix_bry_shape_{marker}")
    bc_dofs = MeshUtils.extract_entity_dofs(V_S, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker))
    bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, None)
    bcs_info.append((bc, V_S, bc_dofs, bc_value))

control_problem = create_shape_problem(
    domain=domain, bcs_info=bcs_info,
    gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
)

obstacle_cost = compute_sigmoid(get_obstacle_map(MeshUtils.define_coordinate(domain)))
cost_form = (
        u * ufl.dx
        + obstacle_cost * ufl.dx
)
cost_fun = IntegralFunction(domain=domain, form=cost_form)

opt_problem = OptimalShapeProblem(
    state_system=state_system, shape_problem=control_problem,
    shape_regulariztions=ShapeRegularization(regularization_list=[
        VolumeRegularization(control_problem, mu=5.0, target_volume_rho=1.0, method='percentage_div')
    ]),
    cost_functional_list=[cost_fun],
    scalar_product=None,
    scalar_product_method={
        'method': 'Poincare-Steklov operator',
        'lambda_lame': 1.0,
        'damping_factor': 0.2,
        'cell_tags': cell_tags,
        'facet_tags': facet_tags,
        'bry_free_markers': bry_free_markers,
        'bry_fixed_markers': bry_fixed_markers,
        'mu_fix': 1.0,
        'mu_free': 1.0,
        'use_inhomogeneous': False,
        'inhomogeneous_exponent': 1.0,
        'update_inhomogeneous': False
    }
)

deformation_handler = MeshDeformationRunner(
    domain,
    volume_change=0.15,
    quality_measures={
        'max_angle': {
            'measure_type': 'max',
            'tol_upper': 165.0,
            'tol_lower': 0.0
        },
        'min_angle': {
            'measure_type': 'min',
            'tol_upper': 180.0,
            'tol_lower': 15.0
        }
    }
)

# opt_problem.compute_state(domain.comm)
# VisUtils.show_scalar_res_vtk(grid, 'state', u)

simulate_dir = os.path.join(proj_dir, 'simulate')
if os.path.exists(simulate_dir):
    shutil.rmtree(simulate_dir)
os.mkdir(simulate_dir)
vtk_recorder = VTKRecorder(os.path.join(simulate_dir, 'simulate_u.pvd'))

init_loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
loss_storge_ctype = ctypes.c_double(init_loss)
cost_converger = CostConvergeHandler(stat_num=25, warm_up_num=25, tol=1e-5, scale=1.0 / init_loss)
vtk_recorder.write_function(u, step=0)


def detect_cost_valid_func(tol_rho=0.1):
    loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
    is_valid = loss < loss_storge_ctype.value + np.abs(loss_storge_ctype.value) * tol_rho
    return is_valid


step = 0
while True:
    step += 1

    shape_grad: dolfinx.fem.Function = opt_problem.compute_gradient(
        domain.comm,
        state_kwargs={'with_debug': False},
        adjoint_kwargs={'with_debug': False},
        gradient_kwargs={'with_debug': False, 'A_assemble_method': 'Identity_row'},
    )

    shape_grad_np = shape_grad.x.array
    shape_grad_np = shape_grad_np * -1.0

    direction_np = np.zeros(domain.geometry.x.shape)
    direction_np[:, :tdim] = shape_grad_np.reshape((-1, tdim))

    # success_flag, info = deformation_handler.move_mesh(direction_np * 0.1)
    success_flag, stepSize = deformation_handler.move_mesh_by_line_search(
        direction_np, max_iter=10, init_stepSize=0.3, stepSize_lower=1e-3,
        detect_cost_valid_func=detect_cost_valid_func
    )

    if success_flag:
        loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
        loss_storge_ctype.value = loss

        print(f"[###Info {step}] loss:{loss:.8f} stepSize:{stepSize}")
        vtk_recorder.write_function(u, step=step)

        is_converge = cost_converger.is_converge(loss)
        if is_converge:
            break

        if step > 250:
            break

    else:
        break

# plt = pyvista.Plotter()
# grid.point_data['u'] = u.x.array
# grid.set_active_scalars('u')
# plt.add_mesh(grid, show_edges=False)
# plt.add_mesh(grid_copy, style='wireframe')
# plt.show()
