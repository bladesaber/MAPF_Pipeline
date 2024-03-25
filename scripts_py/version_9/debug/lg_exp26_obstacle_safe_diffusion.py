import numpy as np
import os
import shutil
import dolfinx
import ufl
from mpi4py import MPI
import ctypes

from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_state_problem, create_shape_problem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, UFLUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder, XDMFRecorder
from scripts_py.version_9.dolfinx_Grad.remesh_helper import MeshDeformationRunner
from scripts_py.version_9.dolfinx_Grad.optimizer_utils import CostConvergeHandler

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/safe_diffusion_02'
model_xdmf_file = os.path.join(proj_dir, 'model.xdmf')


def get_background_map(coord):
    return 2.5 * np.power(coord[0] + 0.4 - np.power(coord[1], 2), 2) + \
        np.power(coord[0], 2) + np.power(coord[1], 2) - 1


def get_obstacle1_map(coord):
    return np.power(coord[0] + 0.3, 2) + np.power(coord[1] - 1.8, 2) - 0.04


def get_obstacle2_map(coord):
    return np.power(coord[0] - 1.1, 2) + np.power(coord[1] - 1.8, 2) - 0.04


def compute_sigmoid(func):
    return 300.0 / (1.0 + ufl.exp(50.0 * func))


# ------ Create Background Map and Obstacle Map
background_domain = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, points=[np.array([-1, -2]), np.array([2, 4])], n=[180, 180]
)
V = dolfinx.fem.FunctionSpace(background_domain, ("Lagrange", 1))

f_exp = get_background_map(MeshUtils.define_coordinate(background_domain))
background_fun = dolfinx.fem.Function(V, name='background')
background_fun.interpolate(UFLUtils.create_expression(f_exp, V))
background_recorder = XDMFRecorder(os.path.join(proj_dir, 'background.xdmf'))
background_recorder.write_mesh(background_domain)
background_recorder.write_function(background_fun, 0)

obs1_map = compute_sigmoid(get_obstacle1_map(MeshUtils.define_coordinate(background_domain)))
obs1_func = dolfinx.fem.Function(V, name='obs1_func')
obs1_func.interpolate(UFLUtils.create_expression(obs1_map, V))
obs1_map_recorder = XDMFRecorder(os.path.join(proj_dir, 'obstacle1_map.xdmf'))
obs1_map_recorder.write_mesh(background_domain)
obs1_map_recorder.write_function(obs1_func, step=0)

obs2_map = compute_sigmoid(get_obstacle2_map(MeshUtils.define_coordinate(background_domain)))
obs2_func = dolfinx.fem.Function(V, name='obs1_func')
obs2_func.interpolate(UFLUtils.create_expression(obs2_map, V))
obs2_map_recorder = XDMFRecorder(os.path.join(proj_dir, 'obstacle2_map.xdmf'))
obs2_map_recorder.write_mesh(background_domain)
obs2_map_recorder.write_function(obs2_func, step=0)

# -----------------------------------------
MeshUtils.msh_to_XDMF(name='model', msh_file=os.path.join(proj_dir, 'model.msh'), output_file=model_xdmf_file, dim=2)
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=model_xdmf_file, mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)

bry_markers = [3]
bry_fixed_markers = []
bry_free_markers = [3]

tdim = domain.topology.dim
fdim = tdim - 1

V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
u = dolfinx.fem.Function(V, name='state_1')
v = dolfinx.fem.Function(V, name='adjoint_1')
boundary_marker = 3

f_exp = get_background_map(MeshUtils.define_coordinate(domain))
F1_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f_exp * v * ufl.dx

bcs_info = []
for marker in bry_markers:
    bc_value = 0.0
    facets = MeshUtils.extract_facet_entities(domain, facet_tags, marker)
    bc1_dofs = MeshUtils.extract_entity_dofs(V, fdim, facets)
    bc1 = dolfinx.fem.dirichletbc(bc_value, bc1_dofs, V)
    bcs_info.append((bc1, V, bc1_dofs, bc_value))

state_problem = create_state_problem(
    name='state_1', F_form=F1_form, state=u, adjoint=v, is_linear=True, bcs_info=bcs_info,
    state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
)
state_system = StateProblem([state_problem])

control_problem = create_shape_problem(
    domain=domain, bcs_info=[],
    gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
)

obs1_cost = compute_sigmoid(get_obstacle1_map(MeshUtils.define_coordinate(domain)))
obs2_cost = compute_sigmoid(get_obstacle2_map(MeshUtils.define_coordinate(domain)))
cost_form = (
        u * ufl.dx
        + obs1_cost * ufl.dx
        + obs2_cost * ufl.dx
)
cost_fun = IntegralFunction(domain=domain, form=cost_form)

opt_problem = OptimalShapeProblem(
    state_system=state_system,
    shape_problem=control_problem,
    shape_regulariztions=ShapeRegularization(regularization_list=[]),
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
    # quality_measures={
    #     'max_angle': {
    #         'measure_type': 'max',
    #         'tol_upper': 165.0,
    #         'tol_lower': 0.0
    #     },
    #     'min_angle': {
    #         'measure_type': 'min',
    #         'tol_upper': 180.0,
    #         'tol_lower': 15.0
    #     }
    # }
)

simulate_dir = os.path.join(proj_dir, 'simulate')
if os.path.exists(simulate_dir):
    shutil.rmtree(simulate_dir)
os.mkdir(simulate_dir)
vtk_recorder = VTKRecorder(os.path.join(simulate_dir, 'simulate_u.pvd'))
vtk_recorder.write_function(u, step=0)

orig_loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
cur_loss_ctype = ctypes.c_double(orig_loss)
cost_converger = CostConvergeHandler(stat_num=25, warm_up_num=25, tol=1e-4, scale=1.0 / orig_loss)


def detect_cost_valid_func(tol_rho=0.1):
    loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
    is_valid = loss < cur_loss_ctype.value + np.abs(cur_loss_ctype.value) * tol_rho
    return is_valid


step = 0
while True:
    step += 1

    shape_grad: dolfinx.fem.Function = opt_problem.compute_gradient(domain.comm)
    shape_grad_np = shape_grad.x.array
    shape_grad_np = shape_grad_np * -1.0

    direction_np = np.zeros(domain.geometry.x.shape)
    direction_np[:, :tdim] = shape_grad_np.reshape((-1, tdim))

    success_flag, step_size = deformation_handler.move_mesh_by_line_search(
        direction_np, max_iter=10, init_stepSize=1.0, stepSize_lower=1e-3,
        detect_cost_valid_func=detect_cost_valid_func, max_step_limit=0.1
    )

    if success_flag:
        loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=False)
        cur_loss_ctype.value = loss

        print(f"[###Info {step}] loss:{loss:.8f} stepSize:{step_size}")
        vtk_recorder.write_function(u, step=step)

        is_converge = cost_converger.is_converge(loss)
        if is_converge:
            break

        if step > 300:
            break

    else:
        break

# orig_recorder = XDMFRecorder(os.path.join(proj_dir, 'orig_opt.xdmf'))
# orig_recorder.write_mesh(domain)
# orig_recorder.write_function(u, step=0)
