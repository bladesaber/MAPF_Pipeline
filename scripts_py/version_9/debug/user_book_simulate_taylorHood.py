import numpy as np
import dolfinx
import ufl
import os
from functools import partial
from ufl import grad, dot, inner, div
import pyvista

from scripts_py.version_9.dolfinx_Grad.fluid_tools.navier_stoke_simulators import FluidSimulatorTaylorHood
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/user_book2'

# domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
#     file=os.path.join(proj_dir, 'last_model.xdmf'),
#     mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
# )

# MeshUtils.msh_to_XDMF(
#     name='model', dim=3,
#     msh_file=os.path.join(proj_dir, 'model.msh'), output_file=os.path.join(proj_dir, 'model.xdmf'),
# )
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=os.path.join(proj_dir, 'model.xdmf'),
    mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)

input_marker = 102
output_markers = [103, 104, 105]
bry_markers = [107]

Re = 100.0


def inflow_velocity_exp(x, tdim):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[2] = 1.0
    return values


inlet_velocity = partial(inflow_velocity_exp, tdim=3)

simulator = FluidSimulatorTaylorHood(domain, cell_tags, facet_tags, Re=Re)



simulator.set_bcs(bcs)

# ----------------------------------------------------------------------
logger_dicts = {
    'outflow': {
        'marker_5': dolfinx.fem.form(dot(simulator.u, simulator.n_vec) * simulator.ds(5)),
        'marker_6': dolfinx.fem.form(dot(simulator.u, simulator.n_vec) * simulator.ds(6)),
        'marker_7': dolfinx.fem.form(dot(simulator.u, simulator.n_vec) * simulator.ds(7)),
    },
    'energy': {
        'energy': dolfinx.fem.form(inner(simulator.u, simulator.u) * ufl.dx),
    },
    'energy_loss': {
        'energy_loss': dolfinx.fem.form(inner(grad(simulator.u), grad(simulator.u)) * ufl.dx),
    }
}
simulator.set_logger_dicts(logger_dicts)

# --------------------------------------------------------------
simulator.run_simulate(
    proj_dir, name='taylorHood_simulate',
    use_guass_init=True,
    stoke_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    nstoke_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    with_debug=True,
    record_mat_dir='/home/admin123456/Desktop/work/topopt_exps/user_book2/tst'
)
