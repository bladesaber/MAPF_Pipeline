"""
关于下一步如何处理NS流的问题与后续工作：
1.切换使用其他成熟软件来做仿真
2.使用AutoGrad方法
3.使用成熟模型RANS的dolfin表达
4.shape optimization会产生网格冲突，需要quality工具
5.先复现space mapping结果
"""

import numpy as np
import ufl
import dolfinx
import shutil
import os
from ufl import div, inner, grad

from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_shape_problem, create_state_problem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import GovDataBase
from scripts_py.version_9.dolfinx_Grad.equation_solver import LinearProblemSolver, NonLinearProblemSolver

from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import ScalarTrackingFunctional
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder, TensorBoardRecorder
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, AssembleUtils
from scripts_py.version_9.dolfinx_Grad.remesh_helper import ReMesher

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/fluid_shape4'
model_xdmf = os.path.join(proj_dir, 'model.xdmf')
msh_file = os.path.join(proj_dir, 'model.msh')

# ------ create xdmf
MeshUtils.msh_to_XDMF(
    name='model',
    msh_file=os.path.join(proj_dir, 'model.msh'),
    output_file=model_xdmf,
    dim=2
)
# -------------------

domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=model_xdmf,
    mesh_name='model',
    cellTag_name='model_cells',
    facetTag_name='model_facets'
)
tdim = domain.topology.dim
fdim = tdim - 1
grid = VisUtils.convert_to_grid(domain)

vertex_indices = ReMesher.reconstruct_vertex_indices(
    orig_msh_file=msh_file,
    domain=domain
)

# ------ Define State Problem 1
input_marker = 1
output_markers = [5, 6, 7]
bry_markers = [2, 3, 4]

P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
TH = ufl.MixedElement([P2, P1])
W = dolfinx.fem.FunctionSpace(domain, TH)
W0, W1 = W.sub(0), W.sub(1)
V, V_to_W = W0.collapse()
Q, Q_to_W = W1.collapse()

# ------ define state problem 1 boundary
bcs_info = []

for marker in bry_markers:
    bc_value = dolfinx.fem.Function(V, name=f"bry_u{marker}")
    bc_dofs = MeshUtils.extract_entity_dofs(
        (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
    )
    bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, W0)
    bcs_info.append((bc, W0, bc_dofs, bc_value))


def inflow_velocity_exp(x):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    # values[0] = 12.0 * (0.0 - x[1]) * (x[1] + 1.0)
    values[0] = 2.0
    return values


bc_in1_value = dolfinx.fem.Function(V, name='inflow_u')
bc_in1_value.interpolate(inflow_velocity_exp)
bc_in1_dofs = MeshUtils.extract_entity_dofs(
    (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
)
bc_in1 = dolfinx.fem.dirichletbc(bc_in1_value, bc_in1_dofs, W0)
bcs_info.append((bc_in1, W0, bc_in1_dofs, bc_in1_value))

for marker in [5, 6, 7]:
    bc_out_value = dolfinx.fem.Function(Q, name=f"outflow_p_{marker}")
    bc_out_dofs = MeshUtils.extract_entity_dofs(
        (W1, Q), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
    )
    bc_out = dolfinx.fem.dirichletbc(bc_out_value, bc_out_dofs, W1)
    bcs_info.append((bc_out, W1, bc_out_dofs, bc_out_value))

# ------ define variation problems
Re = 120
nuValue = 1. / Re
nu = dolfinx.fem.Constant(domain, nuValue)

up = dolfinx.fem.Function(W, name='state_1')
u, p = ufl.split(up)  # please don't use up.split()
vq1 = dolfinx.fem.Function(W, name='adjoint_1')
v1, q1 = ufl.split(vq1)  # please don't use vq.split()
f1 = dolfinx.fem.Constant(domain, np.zeros(tdim))

stoke_form = nu * inner(grad(u), grad(v1)) * ufl.dx - \
             p * div(v1) * ufl.dx - \
             div(u) * q1 * ufl.dx - \
             inner(f1, v1) * ufl.dx
stoke_problem = create_state_problem(
    name='stoke', F_form=stoke_form, state=up, adjoint=vq1, is_linear=True,
    bcs_info=bcs_info,
    state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
)

vq2 = dolfinx.fem.Function(W, name='adjoint_2')
v2, q2 = ufl.split(vq2)
f2 = dolfinx.fem.Constant(domain, np.zeros(tdim))
nstoke_form = nu * inner(grad(u), grad(v2)) * ufl.dx + \
              inner(grad(u) * u, v2) * ufl.dx - \
              inner(p, ufl.div(v2)) * ufl.dx + \
              inner(div(u), q2) * ufl.dx - \
              inner(f2, v2) * ufl.dx
nstoke_problem = create_state_problem(
    name='nstoke', F_form=nstoke_form, state=up, adjoint=vq2, is_linear=False,
    bcs_info=bcs_info,
    state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
)


class NsStateProblem(StateProblem):
    def __init__(
            self,
            stoke_problem: GovDataBase,
            Nstoke_problem: GovDataBase,
    ):
        self.stoke_problem = stoke_problem
        self.nstoke_problem = Nstoke_problem
        self.state_problems = [self.nstoke_problem]

        self.has_solution = False
        self._compute_state_equations()

    def _compute_state_equations(self):
        # ------ define stoke bilinear problem
        stoke_form = self.stoke_problem.F_form
        replace_map = {}
        replace_map.update({self.stoke_problem.state: ufl.TrialFunction(self.stoke_problem.state.function_space)})
        replace_map.update({self.stoke_problem.adjoint: ufl.TestFunction(self.stoke_problem.adjoint.function_space)})
        stoke_eq_form = ufl.replace(stoke_form, replace_map)
        stoke_eq_form_lhs = ufl.lhs(stoke_eq_form)
        stoke_eq_form_rhs = ufl.rhs(stoke_eq_form)
        self.stoke_problem.set_state_eq_form(
            eqs_form=stoke_eq_form, lhs=stoke_eq_form_lhs, rhs=stoke_eq_form_rhs
        )

        # ------ define stoke navier nonlinear problem
        nstoke_form = self.nstoke_problem.F_form
        replace_map = {}
        replace_map.update({self.nstoke_problem.adjoint: ufl.TestFunction(self.nstoke_problem.adjoint.function_space)})
        nstoke_eq_form = ufl.replace(nstoke_form, replace_map)
        nstoke_eq_form_lhs = nstoke_eq_form
        nstoke_eq_form_rhs = 0.0
        self.nstoke_problem.set_state_eq_form(
            eqs_form=nstoke_eq_form, lhs=nstoke_eq_form_lhs, rhs=nstoke_eq_form_rhs
        )

    def solve(self, comm, **kwargs):
        res_dict = LinearProblemSolver.solve_by_petsc_form(
            comm=comm,
            uh=self.stoke_problem.state,
            a_form=self.stoke_problem.state_eq_dolfinx_form_lhs,
            L_form=self.stoke_problem.state_eq_dolfinx_form_rhs,
            bcs=self.stoke_problem.bcs,
            ksp_option=self.stoke_problem.state_ksp_option,
            **kwargs
        )

        if kwargs.get('with_debug', False):
            print(f"[DEBUG Stoke]: max_error:{res_dict['max_error']:.8f} "
                  f"cost_time:{res_dict['cost_time']:.2f}")

        jacobi_form = ufl.derivative(
            self.nstoke_problem.state_eq_form_lhs, self.nstoke_problem.state,
            ufl.TrialFunction(self.nstoke_problem.state.function_space)
        )
        res_dict = NonLinearProblemSolver.solve_by_petsc(
            F_form=self.nstoke_problem.state_eq_form_lhs,
            uh=self.nstoke_problem.state,
            jacobi_form=jacobi_form,
            bcs=self.nstoke_problem.bcs,
            comm=comm,
            ksp_option=self.nstoke_problem.state_ksp_option,
            **kwargs
        )

        if kwargs.get('with_debug', False):
            print(f"[DEBUG Navier Stoke]: max_error:{res_dict['max_error']:.8f} "
                  f"cost_time:{res_dict['cost_time']:.2f}")

        self.has_solution = True

        return self.has_solution


state_system = NsStateProblem(
    stoke_problem=stoke_problem,
    Nstoke_problem=nstoke_problem,
)

# ------ Define Control problem
bry_fixed_markers = [1, 4, 5, 6, 7]
bry_free_marker = [2, 3]

coordinate_space = domain.ufl_domain().ufl_coordinate_element()
V_S = dolfinx.fem.FunctionSpace(domain, coordinate_space)

bcs_info = []
for marker in bry_fixed_markers:
    bc_value = dolfinx.fem.Function(V_S, name=f"fix_bry_shape_{marker}")
    bc_dofs = MeshUtils.extract_entity_dofs(V_S, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker))
    bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, None)
    bcs_info.append((bc, V_S, bc_dofs, bc_value))

control_problem = create_shape_problem(
    domain=domain,
    bcs_info=bcs_info,
    lambda_lame=0.0,
    damping_factor=0.0,
    gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
)

# ------ Define Cost Function
# cost1_form = nu * inner(grad(u), grad(u)) * ufl.dx
# cost1_fun = IntegralFunction(cost1_form)

ds = MeshUtils.define_ds(domain, facet_tags)
n_vec = MeshUtils.define_facet_norm(domain)
tracking_goal = -1.0 * AssembleUtils.assemble_scalar(dolfinx.fem.form(
    ufl.dot(bc_in1_value, n_vec) * ds(input_marker)
)) / 3.0

cost_functional_list = []
for output_marker in output_markers:
    integrand_form = ufl.dot(u, n_vec) * ds(output_marker)
    cost_functional_list.append(
        ScalarTrackingFunctional(domain, integrand_form, tracking_goal)
    )

# ------ Define Optimal Problem
# volume_reg = VolumeRegularization(control_problem, mu=0.2, target_volume_rho=1.0)

opt_problem = OptimalShapeProblem(
    state_system=state_system,
    shape_problem=control_problem,
    # shape_regulariztions=ShapeRegularization(regularization_list=[
    #     volume_reg
    # ]),
    shape_regulariztions=ShapeRegularization([]),
    cost_functional_list=cost_functional_list,
    scalar_product=None
)

# # ------ Debug
# opt_problem.compute_state(domain.comm, with_debug=True)
# u_res = up.sub(0).collapse()
# p_res = up.sub(1).collapse()
# for marker in output_markers:
#     out_vel_form = ufl.dot(u_res, n_vec) * ds(marker)
#     out_flow = AssembleUtils.assemble_scalar(dolfinx.fem.form(out_vel_form))
#     print(f"Outflow {marker}: {out_flow}")
# VisUtils.show_arrow_res_vtk(grid, u_res, V, scale=0.1)
# raise ValueError

# ------ Recorder Init
u_recorder = VTKRecorder(file=os.path.join(proj_dir, 'opt', 'u_res.pvd'))
u_recorder.write_mesh(domain, 0)

log_dir = os.path.join(proj_dir, 'log')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.mkdir(log_dir)
tensor_recorder = TensorBoardRecorder(log_dir=log_dir)

# ------ begin to optimize
best_loss = np.inf
step = 0
while True:
    step += 1

    shape_grad: dolfinx.fem.Function = opt_problem.compute_gradient(domain.comm)

    shape_grad_np = shape_grad.x.array
    # shape_grad_np = shape_grad_np / np.linalg.norm(shape_grad_np, ord=2)
    shape_grad_np = shape_grad_np * -0.2

    displacement_np = np.zeros(domain.geometry.x.shape)
    displacement_np[:, :tdim] = shape_grad_np.reshape((-1, tdim))

    # dJ = dolfinx.fem.Function(shape_grad.function_space)
    # dJ.x.array[:] = shape_grad_np.reshape(-1)
    # VisUtils.show_vector_res_vtk(grid, dJ, dim=2, with_wrap=True)

    if step % 2 == 0:
        # dJ = dolfinx.fem.Function(shape_grad.function_space)
        # dJ.x.array[:] = shape_grad_np.reshape(-1)
        # VisUtils.show_vector_res_vtk(grid, dJ, dim=2, with_wrap=True)

        u_res = up.sub(0).collapse()
        u_recorder.write_function(u_res, step)

    MeshUtils.move(domain, displacement_np)
    loss = opt_problem.evaluate_cost_functional(domain.comm, update_state=True)
    loss = loss * 1.0  # just scale for vis

    tensor_recorder.write_scalar('loss', loss, step)

    # ------ record velocity
    out_flow_dict = {}
    for marker in output_markers:
        out_vel_form = ufl.dot(u, n_vec) * ds(marker)
        out_flow = AssembleUtils.assemble_scalar(dolfinx.fem.form(out_vel_form))
        out_flow_dict[f"out_{marker}"] = out_flow
    tensor_recorder.write_scalars('flow', out_flow_dict, step)

    best_loss = np.minimum(loss, best_loss)
    print(f"[###Step {step}] loss:{loss:.8f} / best_loss:{best_loss:.8f}")

    # if loss > best_loss * 1.5:
    #     break

    if step > 60:
        break

step += 1
u_res = up.sub(0).collapse()
u_recorder.write_function(u_res, step)

new_msh_file = os.path.join(proj_dir, 'opt_model.msh')
ReMesher.convert_domain_to_new_msh(
    orig_msh_file=msh_file,
    new_msh_file=new_msh_file,
    domain=domain,
    dim=tdim,
    vertex_indices=vertex_indices
)

# ------ Statistic Record
print("[Statistic]:")
opt_problem.compute_state(domain.comm, with_debug=True)
for marker in output_markers:
    out_vel_form = ufl.dot(u_res, n_vec) * ds(marker)
    out_flow = AssembleUtils.assemble_scalar(dolfinx.fem.form(out_vel_form))
    print(f"    Outflow {marker}: {out_flow}")
