"""
Incremental Pressure Correction Scheme (IPCS) Based on Adams-Bashforth Method
"""

import os
import shutil
from functools import partial
import dolfinx
import ufl
from ufl import inner, dot, grad, div, sym, nabla_grad, Identity
import numpy as np
from typing import Union
from petsc4py import PETSc

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, BoundaryUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder, TensorBoardRecorder
from scripts_py.version_9.dolfinx_Grad.equation_solver import LinearProblemSolver, NonLinearProblemSolver
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import AssembleUtils

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/fluid_shape8'
model_xdmf = os.path.join(proj_dir, 'last_model.xdmf')

# ------ example 1 environment parameters
# proj_dir = '/home/admin123456/Desktop/work/topopt_exps/fluid_shape4/tst'
# xdmf_file = os.path.join(proj_dir, 'geometry.xdmf')
#
# domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
#     file=xdmf_file, mesh_name='fluid', cellTag_name='fluid_cells', facetTag_name='fluid_facets'
# )
#
# input_marker = 2
# output_markers = [3]
# bry_markers = [4, 5]
#
#
# class InletVelocity(object):
#     def __init__(self, t):
#         self.t = t
#
#     def __call__(self, x):
#         values = np.zeros((tdim, x.shape[1]), dtype=PETSc.ScalarType)
#         values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41 ** 2)
#         return values
#
#
# inlet_velocity = InletVelocity(0.0)

# ------ environment parameters
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=model_xdmf, mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)
is_channel_fluid = True

input_marker = 1
output_markers = [5, 6, 7]
bry_markers = [2, 3, 4]


def inflow_velocity_exp(x, tdim):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[0] = 12.0 * (0.0 - x[1]) * (x[1] + 1.0)
    return values


inlet_velocity = partial(inflow_velocity_exp, tdim=2)

# -------------------------------------
tdim = domain.topology.dim
fdim = tdim - 1

P2_element = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
P1_element = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
V = dolfinx.fem.FunctionSpace(domain, P2_element)
Q = dolfinx.fem.FunctionSpace(domain, P1_element)

t = 0
T = 8  # Final time
dt = 1 / 300.0  # Time step size
# num_steps = int(T / dt)
# num_steps = 5000

# ----------------------------
bcs_u = []

u_inlet = dolfinx.fem.Function(V, name='inflow_u')
u_inlet.interpolate(inlet_velocity)
bc_in1 = dolfinx.fem.dirichletbc(
    u_inlet,
    MeshUtils.extract_entity_dofs(V, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, input_marker))
)
bcs_u.append(bc_in1)

for marker in bry_markers:
    bc_value = dolfinx.fem.Function(V, name=f"bry_u_{marker}")
    bc = dolfinx.fem.dirichletbc(
        bc_value,
        MeshUtils.extract_entity_dofs(V, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker))
    )
    bcs_u.append(bc)

bcs_p = []
for marker in output_markers:
    bc_out_value = dolfinx.fem.Function(Q, name=f"outflow_p_{marker}")
    bc_out = dolfinx.fem.dirichletbc(
        bc_out_value,
        MeshUtils.extract_entity_dofs(Q, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker))
    )
    bcs_p.append(bc_out)


# --------------------------------------
def epsilon(u: Union[dolfinx.fem.Function, ufl.Argument]):
    """
    Define strain-rate tensor: 0.5 * (grad(U) + grad(U).T)
    """
    return sym(nabla_grad(u))


def sigma(u, p, mu: dolfinx.fem.Constant):
    """
    Define stress tensor:
        mu: Dynamic viscosity
    """
    return 2.0 * mu * epsilon(u) - p * Identity(len(u))


n_vec = MeshUtils.define_facet_norm(domain)
ds = MeshUtils.define_ds(domain, facet_tags)
k = dolfinx.fem.Constant(domain, dt)  # time step
mu = dolfinx.fem.Constant(domain, 0.01)  # Dynamic viscosity
rho = dolfinx.fem.Constant(domain, 1.0)  # Density
f = dolfinx.fem.Constant(domain, np.zeros(tdim))  # body force

# ------ Define the variational problem for the first step
u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
v, q = ufl.TestFunction(V), ufl.TestFunction(Q)

u_n = dolfinx.fem.Function(V, name='velocity_n_step')
p_n = dolfinx.fem.Function(Q, name='pressure_n_step')
U = 0.5 * (u_n + u)

F1 = rho * ufl.dot((u - u_n) / k, v) * ufl.dx
F1 += rho * ufl.dot(ufl.dot(u_n, ufl.nabla_grad(u_n)), v) * ufl.dx
F1 += ufl.inner(sigma(U, p_n, mu), epsilon(v)) * ufl.dx
F1 += ufl.dot(p_n * n_vec, v) * ufl.ds - ufl.dot(mu * ufl.nabla_grad(U) * n_vec, v) * ufl.ds
F1 -= ufl.dot(f, v) * ufl.dx
lhs1 = ufl.lhs(F1)
rhs1 = ufl.rhs(F1)
a1 = dolfinx.fem.form(lhs1)
L1 = dolfinx.fem.form(rhs1)
A1 = AssembleUtils.assemble_mat(a1, bcs=bcs_u)
b1 = AssembleUtils.create_vector(L1)

solver1 = LinearProblemSolver.create_petsc_solver(
    comm=domain.comm, A_mat=A1,
    solver_setting={
        'ksp_type': PETSc.KSP.Type.BCGS, 'pc_type': PETSc.PC.Type.HYPRE,
        'pc_hypre_mat_solver_type': 'boomeramg'
    },
)

# ------ Define variational problem for step 2
u_ = dolfinx.fem.Function(V, name='u_')

lhs2 = ufl.dot(ufl.nabla_grad(p), ufl.nabla_grad(q)) * ufl.dx  # compute correction pressure
rhs2 = ufl.dot(ufl.nabla_grad(p_n), ufl.nabla_grad(q)) * ufl.dx - (rho / k) * ufl.div(u_) * q * ufl.dx

a2 = dolfinx.fem.form(lhs2)
L2 = dolfinx.fem.form(rhs2)
A2 = AssembleUtils.assemble_mat(a2, bcs=bcs_p)
b2 = AssembleUtils.create_vector(L2)

solver2 = LinearProblemSolver.create_petsc_solver(
    comm=domain.comm, A_mat=A2,
    solver_setting={
        'ksp_type': PETSc.KSP.Type.BCGS, 'pc_type': PETSc.PC.Type.HYPRE,
        'pc_hypre_mat_solver_type': 'boomeramg'
    }
)

# ------ Define variational problem for step 3
p_ = dolfinx.fem.Function(Q, name='pressure_n_plus_1_step')

lhs3 = rho * ufl.dot(u, v) * ufl.dx  # compute correction velocity
rhs3 = rho * ufl.dot(u_, v) * ufl.dx - k * ufl.dot(ufl.nabla_grad(p_ - p_n), v) * ufl.dx

a3 = dolfinx.fem.form(lhs3)
L3 = dolfinx.fem.form(rhs3)
A3 = AssembleUtils.assemble_mat(a3, bcs=[])
b3 = AssembleUtils.create_vector(L3)

solver3 = LinearProblemSolver.create_petsc_solver(
    comm=domain.comm, A_mat=A3,
    solver_setting={
        'ksp_type': PETSc.KSP.Type.CG, 'pc_type': PETSc.PC.Type.SOR,
    }
)

# ------ solve the question
record_dir = os.path.join(proj_dir, 'last_model_simulate')
if os.path.exists(record_dir):
    shutil.rmtree(record_dir)
os.mkdir(record_dir)

u_record_dir = os.path.join(record_dir, 'velocity')
os.mkdir(u_record_dir)
u_recorder = VTKRecorder(file=os.path.join(u_record_dir, 'velocity.pvd'))

p_record_dir = os.path.join(record_dir, 'pressure')
os.mkdir(p_record_dir)
p_recorder = VTKRecorder(file=os.path.join(p_record_dir, 'pressure.pvd'))

tensorBoard_dir = os.path.join(record_dir, 'log')
os.mkdir(tensorBoard_dir)
out_logger = TensorBoardRecorder(tensorBoard_dir)

outflow_data_convergence = {}
for marker in output_markers:
    outflow_data_convergence[f"marker_{marker}"] = {
        'form': dolfinx.fem.form(ufl.dot(u_n, n_vec) * ds(marker)),
        'value': 0.0, 'old_value': 0.0
    }

# --------------------------------
tol = 1e-6
warmup = 3
t = 0
step = 0
while True:
    step += 1
    t += dt

    # inlet_velocity.t = t
    # u_inlet.interpolate(inlet_velocity)

    computation_max_errors = {}
    # ------ Step 1: Tentative velocity step
    if not is_channel_fluid:
        A1.zeroEntries()
        AssembleUtils.assemble_mat(a1, bcs=bcs_u, A_mat=A1)

    AssembleUtils.assemble_vec(L1, b1, clear_vec=True)
    BoundaryUtils.apply_boundary_to_vec(b1, bcs=bcs_u, a_form=a1, clean_vec=False)

    res_dict = LinearProblemSolver.solve_by_petsc(
        b_vec=b1, solver=solver1, A_mat=A1, setOperators=False, with_debug=True
    )
    u_.vector.aypx(0.0, res_dict['res'])
    computation_max_errors['step1'] = res_dict['max_error']

    # ------ Step 2: Pressure step
    AssembleUtils.assemble_vec(L2, b2, clear_vec=True)
    BoundaryUtils.apply_boundary_to_vec(b2, bcs=bcs_p, a_form=a2, clean_vec=False)

    res_dict = LinearProblemSolver.solve_by_petsc(
        b_vec=b2, solver=solver2, A_mat=A2, setOperators=False, with_debug=True
    )
    p_.vector.aypx(0.0, res_dict['res'])
    computation_max_errors['step2'] = res_dict['max_error']

    # ------ Step 3: Velocity correction step
    AssembleUtils.assemble_vec(L3, b3, clear_vec=True)

    res_dict = LinearProblemSolver.solve_by_petsc(
        b_vec=b3, solver=solver3, A_mat=A3, setOperators=False, with_debug=True
    )
    u_.vector.aypx(0.0, res_dict['res'])
    computation_max_errors['step3'] = res_dict['max_error']

    # ------ Step 4 Update solution to current time step
    u_n.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]

    # ------ Step 5 record result
    if step % 50 == 0:
        u_recorder.write_function(u_n, t)
        p_recorder.write_function(p_n, t)
    out_logger.write_scalars('computation_errors', computation_max_errors, step=step)

    # ------ Step 6 check convergence
    for key in outflow_data_convergence.keys():
        outflow_data_convergence[key]['old_value'] = outflow_data_convergence[key]['value']
        outflow_data_convergence[key]['value'] = AssembleUtils.assemble_scalar(outflow_data_convergence[key]['form'])

    if step > warmup:
        data_cells, convergence_cells = {}, {}
        is_converge = True
        for key in outflow_data_convergence.keys():
            new_value = outflow_data_convergence[key]['value']
            old_value = outflow_data_convergence[key]['old_value']

            ratio = np.abs(new_value / (old_value + 1e-6) - 1.0)
            is_converge = is_converge and (ratio < tol)

            convergence_cells[key] = ratio
            data_cells[key] = new_value
        out_logger.write_scalars('outflow', data_cells, step)
        out_logger.write_scalars('outflow_convergence', convergence_cells, step)

    else:
        is_converge = False

    print(f"[Info iter:{step}] Step1 Error:{computation_max_errors['step1']:.8f}, "
          f"Step2 Error:{computation_max_errors['step2']:.8f}, "
          f"Step3 Error:{computation_max_errors['step3']:.8f}")

    if step > 10000:
        print('[Debug] Fail Converge')
        break

    if is_converge:
        if step > 10:
            print('[Debug] Successful Converge')
        else:
            print('[Debug] Time Step May Be Too Small')
        break
