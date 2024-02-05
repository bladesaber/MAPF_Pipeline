import dolfinx
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import CellType
import ufl
from ufl import inner, grad, dot, div
import os

from scripts_py.version_9.dolfinx_Grad.equation_solver import NonLinearProblemSolver
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder

N = 64
domain = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [N, N], CellType.triangle
)
grid = VisUtils.convert_to_grid(domain)
tdim = domain.topology.dim
fdim = tdim - 1

P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
TH = ufl.MixedElement([P2, P1])
W = dolfinx.fem.FunctionSpace(domain, TH)
W0, W1 = W.sub(0), W.sub(1)
V, V_to_W = W0.collapse()
Q, Q_to_W = W1.collapse()


# ------ define no slip boundary
def noslip_boundary(x):
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
        np.isclose(x[1], 0.0)
    )


bc0_value = dolfinx.fem.Function(V, name=f"no_slip")
bc0_dofs = MeshUtils.extract_entity_dofs(
    (W0, V), fdim, MeshUtils.extract_facet_entities(domain, None, noslip_boundary)
)
bc_noslip = dolfinx.fem.dirichletbc(bc0_value, bc0_dofs, W0)


# ------ define outflow boundary
def outflow_boundary(x):
    return np.isclose(x.T, [0, 0, 0]).all(axis=1)


bc1_value = dolfinx.fem.Function(Q, name=f"outflow_bry")
bc1_dofs = MeshUtils.extract_entity_dofs(
    (W1, Q), fdim, MeshUtils.extract_facet_entities(domain, None, outflow_boundary)
)
bc_out = dolfinx.fem.dirichletbc(bc1_value, bc1_dofs, W1)


# ------ define inflow bpundary
def lid(x):
    return np.isclose(x[1], 1.0)


def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))


bc2_value = dolfinx.fem.Function(V, name='inflow_bry')
bc2_value.interpolate(lid_velocity_expression)
bc2_dofs = MeshUtils.extract_entity_dofs(
    (W0, V), fdim, MeshUtils.extract_facet_entities(domain, None, lid)
)
bc_in = dolfinx.fem.dirichletbc(bc2_value, bc2_dofs, W0)

bcs = [bc_in, bc_noslip, bc_out]

up = dolfinx.fem.Function(W, name='state_1')
u, p = ufl.split(up)
vq = ufl.TestFunction(W)
v, q = ufl.split(vq)
f = dolfinx.fem.Constant(domain, np.zeros(tdim))
nu = dolfinx.fem.Constant(domain, 0.0)

F_form = nu * inner(grad(u), grad(v)) * ufl.dx + \
         inner(grad(u) * u, v) * ufl.dx - \
         inner(p, ufl.div(v)) * ufl.dx + \
         inner(div(u), q) * ufl.dx
jacobi_form = ufl.derivative(F_form, up, ufl.TrialFunction(up.function_space))

res_dir = '/home/admin123456/Desktop/work/topopt_exps/fluid_shape4/ns_sim'
Re_list = np.array([20, 100, 400, 1000, 2000, 2500])
for Re in Re_list:
    nuValue = 1. / Re
    nu.value = nuValue

    res_dict = NonLinearProblemSolver.solve_by_petsc(
        F_form=F_form,
        uh=up,
        jacobi_form=jacobi_form,
        bcs=bcs,
        comm=domain.comm,
        ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        with_debug=True
    )
    print(f"Re:{Re} max_error:{res_dict['max_error']}")

    u_res = res_dict['res'].sub(0).collapse()
    recorder = VTKRecorder(os.path.join(res_dir, f"Re_{Re}.pvd"), comm=domain.comm)
    recorder.write_function(u_res, 0)
