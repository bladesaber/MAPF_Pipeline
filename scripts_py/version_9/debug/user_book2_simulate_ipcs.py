import numpy as np
import dolfinx
import ufl
import os
from functools import partial
from ufl import grad, dot, inner, div
import pyvista

from scripts_py.version_9.dolfinx_Grad.fluid_tools.navier_stoke_simulators import FluidSimulator
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils

# ------ Example 1
# proj_dir = '/home/admin123456/Desktop/work/topopt_exps/user_book1'
#
# domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
#     file=os.path.join(proj_dir, 'last_model.xdmf'),
#     mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
# )
#
# input_marker = 1
# output_markers = [5, 6, 7]
# bry_markers = [2, 3, 4]
#
# dt = 1 / 300.0
# dynamic_viscosity = 0.01
# density = 1.0
# body_force = None
#
#
# def inflow_velocity_exp(x, tdim):
#     num = x.shape[1]
#     values = np.zeros((tdim, num))
#     values[0] = 12.0 * (0.0 - x[1]) * (x[1] + 1.0)
#     return values
#
#
# inlet_velocity = partial(inflow_velocity_exp, tdim=2)

# ------ Example 2
proj_dir = '/home/admin123456/Desktop/work/topopt_exps/user_book2'

domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=os.path.join(proj_dir, 'model.xdmf'),
    mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)

input_marker = 102
output_markers = [103, 104, 105]
bry_markers = [107]

dt = 1 / 300.0
dynamic_viscosity = 0.01
density = 1.0
body_force = None


def inflow_velocity_exp(xyz, tdim):
    num = xyz.shape[1]
    values = np.zeros((tdim, num))
    values[2] = 2.0
    return values


inlet_velocity = partial(inflow_velocity_exp, tdim=3)

simulator = FluidSimulator(domain, cell_tags, facet_tags)

# ------ IPCS Simulator
# simulator.define_ipcs_equation(
#     dt=dt,
#     dynamic_viscosity=dynamic_viscosity,
#     density=density,
#     is_channel_fluid=True
# )
simulator.define_navier_stoke_equation(Re=100.0)

# ------ define boundary
for marker in bry_markers:
    bc_value = dolfinx.fem.Function(simulator.V, name=f"bry_u{marker}")
    simulator.add_boundary(
        value=bc_value,
        marker=marker,
        is_volicity=True
    )

inflow_value = dolfinx.fem.Function(simulator.V, name='inflow_u')
inflow_value.interpolate(partial(inflow_velocity_exp, tdim=simulator.tdim))
simulator.add_boundary(
    value=inflow_value,
    marker=input_marker,
    is_volicity=True
)

for marker in output_markers:
    bc_value = dolfinx.fem.Function(simulator.Q, name=f"outflow_p_{marker}")
    simulator.add_boundary(
        value=bc_value,
        marker=marker,
        is_volicity=False
    )

# ------
# ipcs_dict = simulator.equation_map['ipcs']
# u_n, p_n = ipcs_dict['u_n'], ipcs_dict['p_n']

nstoke_dict = simulator.equation_map['navier_stoke']
up: dolfinx.fem.Function = nstoke_dict['up']
u_n, p_n = ufl.split(up)

data_convergence = {}
for marker in output_markers:
    data_convergence[f"marker_{marker}"] = {
        'form': dolfinx.fem.form(dot(u_n, simulator.n_vec) * simulator.ds(marker)),
        'cur_value': 0.0, 'old_value': 0.0
    }

outflow_dict = {}
for marker in output_markers:
    outflow_dict[f"marker_{marker}"] = dolfinx.fem.form(dot(u_n, simulator.n_vec) * simulator.ds(marker))
logger_dicts = {
    'outflow': outflow_dict,
    'energy': {'energy': dolfinx.fem.form(inner(u_n, u_n) * ufl.dx)},
    'energy_loss': {'energy_loss': dolfinx.fem.form(inner(grad(u_n), grad(u_n)) * ufl.dx)}
}

# simulator.simulate_ipcs(
#     proj_dir, name='ipcs',
#     log_iter=50, max_iter=8000, tol=1e-5, data_convergence=data_convergence,
#     logger_dicts=logger_dicts,
#     with_debug=True
# )

simulator.simulate_navier_stoke(
    proj_dir, name='NS',
    ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    logger_dicts=logger_dicts,
    with_debug=True
)
