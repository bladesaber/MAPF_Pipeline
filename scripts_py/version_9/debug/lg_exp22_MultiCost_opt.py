import shutil
import numpy as np
import ufl
import dolfinx
import os
import ctypes
from basix.ufl import element
from ufl import div, inner, grad
from functools import partial
import pyvista

from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_shape_problem, create_state_problem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.problem_state import StateProblem

from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import ScalarTrackingFunctional, \
    IntegralFunction, MinMaxFunctional
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization, \
    VolumeRegularization
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder, TensorBoardRecorder, RefDataRecorder
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, AssembleUtils
from scripts_py.version_9.dolfinx_Grad.remesh_helper import MeshDeformationRunner
from scripts_py.version_9.dolfinx_Grad.optimizer_utils import CostConvergeHandler

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/fluid_shape6'
MeshUtils.msh_to_XDMF(
    name='model', dim=2,
    msh_file=os.path.join(proj_dir, 'model.msh'), output_file=os.path.join(proj_dir, 'model.xdmf'),
)
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=os.path.join(proj_dir, 'model.xdmf'),
    mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)

# ----------------------------------------
input_marker = 1
output_markers = [5, 6, 7]
bry_markers = [2, 3, 4]

bry_fixed_markers = [1, 4, 5, 6, 7]
bry_free_markers = [2, 3]


def inflow_velocity_exp(x, tdim):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[0] = 12.0 * (0.0 - x[1]) * (x[1] + 1.0)
    return values


# ------
grid = VisUtils.convert_to_grid(domain)
tdim = domain.topology.dim
fdim = tdim - 1

W = dolfinx.fem.FunctionSpace(
    domain, ufl.MixedElement([
        ufl.VectorElement("Lagrange", domain.ufl_cell(), 2), ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    ])
)
W0, W1 = W.sub(0), W.sub(1)
V, V_to_W = W0.collapse()
Q, Q_to_W = W1.collapse()
V_mapping_space = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
Q_mapping_space = dolfinx.fem.FunctionSpace(domain, ("CG", 1))

# ---------------------
bcs_info = []
for marker in bry_markers:
    bc_value = dolfinx.fem.Function(V, name=f"bry_u{marker}")
    bc_dofs = MeshUtils.extract_entity_dofs(
        (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
    )
    bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, W0)
    bcs_info.append((bc, W0, bc_dofs, bc_value))

bc_in_value = dolfinx.fem.Function(V, name='inflow_u')
bc_in_value.interpolate(partial(inflow_velocity_exp, tdim=tdim))
bc_in_dofs = MeshUtils.extract_entity_dofs(
    (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
)

bc_in1 = dolfinx.fem.dirichletbc(bc_in_value, bc_in_dofs, W0)
bcs_info.append((bc_in1, W0, bc_in_dofs, bc_in_value))

for marker in output_markers:
    bc_out_value = dolfinx.fem.Function(Q, name=f"outflow_p_{marker}")
    bc_out_dofs = MeshUtils.extract_entity_dofs(
        (W1, Q), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
    )
    bc_out = dolfinx.fem.dirichletbc(bc_out_value, bc_out_dofs, W1)
    bcs_info.append((bc_out, W1, bc_out_dofs, bc_out_value))

# ---------------------
up = dolfinx.fem.Function(W, name='coarse_state')
u, p = ufl.split(up)
vq = dolfinx.fem.Function(W, name='coarse_adjoint')
v, q = ufl.split(vq)
f = dolfinx.fem.Constant(domain, np.zeros(tdim))

F_form = inner(grad(u), grad(v)) * ufl.dx - p * div(v) * ufl.dx - q * div(u) * ufl.dx - inner(f, v) * ufl.dx

state_problem = create_state_problem(
    name='coarse_1', F_form=F_form, state=up, adjoint=vq, is_linear=True,
    bcs_info=bcs_info,
    state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
)
state_system = StateProblem([state_problem])

# --------------------------
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
    gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
)

# ------------------------------------------------
ds = MeshUtils.define_ds(domain, facet_tags)
n_vec = MeshUtils.define_facet_norm(domain)
tracking_goal = -1.0 * AssembleUtils.assemble_scalar(dolfinx.fem.form(
    ufl.dot(bc_in_value, n_vec) * ds(input_marker)
)) / 3.0

target_goal_dict = {}
for marker in output_markers:
    target_goal_dict[f"marker_{marker}"] = tracking_goal

cost_functional_list = []
for marker in output_markers:
    integrand_form = ufl.dot(u, n_vec) * ds(marker)
    cost_functional_list.append(ScalarTrackingFunctional(
        domain, integrand_form, target_goal_dict[f"marker_{marker}"], name=f"track_{marker}"
    ))

energy_loss_form = inner(grad(u), grad(u)) * ufl.dx
energy_loss_fun = IntegralFunction(domain=domain, form=energy_loss_form, name=f"energy_loss")
cost_functional_list.append(energy_loss_fun)

# ------------------------------------------------
opt_problem = OptimalShapeProblem(
    state_system=state_system,
    shape_problem=control_problem,
    shape_regulariztions=ShapeRegularization([
        VolumeRegularization(control_problem, mu=10.0, target_volume_rho=1.0, method='percentage_div')
    ]),
    cost_functional_list=cost_functional_list,
    scalar_product=None,
    scalar_product_method={
        'method': 'Poincare-Steklov operator',
        'lambda_lame': 0.0,
        'damping_factor': 0.0,
        'cell_tags': cell_tags,
        'facet_tags': facet_tags,
        'bry_free_markers': bry_free_markers,
        'bry_fixed_markers': bry_fixed_markers,
        'mu_fix': 1.0,
        'mu_free': 1.0,
        'use_inhomogeneous': True,
        'inhomogeneous_exponent': 1.0,
        'update_inhomogeneous': False
    }
)

# ------ Single Step Debug
# opt_problem.state_system.solve(domain.comm, with_debug=True)
# for cost_func in cost_functional_list:
#     print(cost_func.name, cost_func.evaluate())
# opt_problem.update_cost_funcs()
# opt_problem.adjoint_system.solve(domain.comm, with_debug=True)
# opt_problem.gradient_system.solve(domain.comm, with_debug=True, A_assemble_method='Identity_row')

# --------------------------------------------------
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
# res = deformation_handler.compute_mesh_quality(domain)

# ----------------------------------------------------------------------------------
simulate_dir = os.path.join(proj_dir, 'simulate')
if os.path.exists(simulate_dir):
    shutil.rmtree(simulate_dir)
os.mkdir(simulate_dir)
vtk_recorder = VTKRecorder(os.path.join(simulate_dir, 'simulate_u.pvd'))

tensorBoard_dir = os.path.join(proj_dir, 'log')
if os.path.exists(tensorBoard_dir):
    shutil.rmtree(tensorBoard_dir)
os.mkdir(tensorBoard_dir)
log_recorder = TensorBoardRecorder(tensorBoard_dir)


class DataCell(object):
    def __init__(self):
        self.outflow_datas = {}
        for marker in output_markers:
            self.outflow_datas[f"marker_{marker}"] = {
                'form': dolfinx.fem.form(ufl.dot(u, n_vec) * ds(marker))
            }

        self.energy_form = dolfinx.fem.form(ufl.inner(u, u) * ufl.dx)

    def evaluate_outflow(self):
        data = {}
        for key in self.outflow_datas.keys():
            data[key] = AssembleUtils.assemble_scalar(self.outflow_datas[key]['form'])
        return data

    def evaluate_energy(self):
        return AssembleUtils.assemble_scalar(self.energy_form)

    def evaluate_cost(self):
        data = {}
        for cost_fun in cost_functional_list:
            data[cost_fun.name] = cost_fun.evaluate()
        return data


# ---------------------------------------------------------------------------------
opt_problem.state_system.solve(domain.comm, with_debug=True)
for cost_func in cost_functional_list:
    cost = cost_func.evaluate()
    print(f"[DEBUG Weight init]: {cost_func.name}:{cost}")
    if cost_func.name == 'energy_loss':
        weight = 2.0 / cost
    elif cost_func.name == 'MinMax_volume':
        pass
    else:
        weight = (1.0 / 3.0) / cost
    cost_func.update_scale(weight)

data_cell = DataCell()
outflow_cells = data_cell.evaluate_outflow()
log_recorder.write_scalars('outflow', outflow_cells, step=0)

energy_value = data_cell.evaluate_energy()
log_recorder.write_scalar('energy', scalar_value=energy_value, step=0)

# ----------------------------------------------------------------------------------
init_loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
loss_storge_ctype = ctypes.c_double(init_loss)
cost_converger = CostConvergeHandler(stat_num=25, warm_up_num=25, tol=5e-3, scale=1.0 / init_loss)


def detect_cost_valid_func(tol_rho=0.05):
    loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
    is_valid = loss < loss_storge_ctype.value + np.abs(loss_storge_ctype.value) * tol_rho
    return is_valid


step = 0
while True:
    step += 1

    print(f"Step {step}:")
    shape_grad: dolfinx.fem.Function = opt_problem.compute_gradient(
        domain.comm,
        state_kwargs={'with_debug': False},
        adjoint_kwargs={'with_debug': False},
        gradient_kwargs={'with_debug': False, 'A_assemble_method': 'Identity_row'},
    )

    shape_grad_np = shape_grad.x.array
    # shapr_grade_scale = np.linalg.norm(shape_grad_np, ord=2)
    shape_grad_np = shape_grad_np * -1.0

    displacement_np = np.zeros(domain.geometry.x.shape)
    displacement_np[:, :tdim] = shape_grad_np.reshape((-1, tdim))

    # grid['grad_test'] = displacement_np
    # VisUtils.show_arrow_from_grid(grid, 'grad_test', scale=30.0).show()

    # success_flag, info = deformation_handler.move_mesh(displacement_np)
    success_flag, stepSize = deformation_handler.move_mesh_by_line_search(
        displacement_np, max_iter=10, init_stepSize=2.0, stepSize_lower=1e-3,
        detect_cost_valid_func=detect_cost_valid_func
    )

    if success_flag:
        loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
        loss_storge_ctype.value = loss

        is_converge = cost_converger.is_converge(loss)

        # ------ record
        scale_loss = cost_converger.compute_scale_loss(loss)
        log_recorder.write_scalar('scale_loss', scale_loss, step=step)
        log_recorder.write_scalar('scale_loss_var', cost_converger.scale_cost_variation, step=step)

        outflow_cells = data_cell.evaluate_outflow()
        log_recorder.write_scalars('outflow', outflow_cells, step=step)

        energy_value = data_cell.evaluate_energy()
        log_recorder.write_scalar('energy', scalar_value=energy_value, step=step)

        cost_cells = data_cell.evaluate_cost()
        log_recorder.write_scalars('cost_cells', cost_cells, step=step)

        vtk_recorder.write_function(up.sub(0).collapse(), step=step)

        # ------ debug output
        debug_log = f"[###Info] "
        for key in outflow_cells.keys():
            target_flow = target_goal_dict[key]
            out_flow = outflow_cells[key]
            ratio = out_flow / target_flow
            debug_log += f"[{key}: {ratio:.2f}| {out_flow:.3f}/{target_flow: .3f}] "
        debug_log += f"loss:{loss:.8f}, energy:{energy_value:.4f}, stepSize:{stepSize}"
        print(debug_log)
        # ------

        if is_converge:
            break

        if step > 150:
            break

    else:
        break

    # print("\n")

u_res = up.sub(0).collapse()
u_res_map = dolfinx.fem.Function(V_mapping_space)
u_res_map.interpolate(u_res)
u_res_map_np = np.zeros(grid.points.shape)
u_res_map_np[:, :tdim] = u_res_map.x.array.reshape((-1, tdim))
grid['u_res'] = u_res_map_np
VisUtils.show_arrow_from_grid(grid, 'u_res', scale=0.25).show()

MeshUtils.save_XDMF(os.path.join(proj_dir, 'last_model.xdmf'), domain, cell_tags, facet_tags)
