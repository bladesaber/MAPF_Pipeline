import shutil
import numpy as np
import dolfinx
import ufl
import os
from functools import partial
from ufl import grad, dot, inner, div
import pyvista

from scripts_py.version_9.dolfinx_Grad.fluid_tools.fluid_simulators import FluidSimulator
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/user_book5'
MeshUtils.msh_to_XDMF(
    name='model', dim=3,
    msh_file=os.path.join(proj_dir, 'model.msh'), output_file=os.path.join(proj_dir, 'model.xdmf'),
)
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=os.path.join(proj_dir, 'model.xdmf'),
    mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)

input_marker = 17
output_markers = [18]
bry_markers = [19, 20]

dt = 1 / 350.0
dynamic_viscosity = 0.01
density = 1.0
body_force = None


def inflow_velocity_exp(x, tdim):
    num = x.shape[1]
    values = np.zeros((tdim, num))

    # values[0] = 1.0

    dist = 0.7 - np.sqrt(np.power(x[1] - 0.5, 2) + np.power(x[2] - 0.5, 2))
    values[0] = np.power(dist, 2.0) * 6.0

    return values


inlet_velocity = partial(inflow_velocity_exp, tdim=3)

simulator = FluidSimulator(domain, cell_tags, facet_tags)

# ------
# simulator.define_stoke_equation()
simulator.define_navier_stoke_equation(Re=100.0)
simulator.define_ipcs_equation(
    dt=dt,
    dynamic_viscosity=dynamic_viscosity,
    density=density,
    is_channel_fluid=True
)

# ------ define boundary
for marker in bry_markers:
    bc_value = dolfinx.fem.Function(simulator.V, name=f"bry_u{marker}")
    simulator.add_boundary(value=bc_value, marker=marker, is_velocity=True)

inflow_value = dolfinx.fem.Function(simulator.V, name='inflow_u')
inflow_value.interpolate(partial(inflow_velocity_exp, tdim=simulator.tdim))
simulator.add_boundary(value=inflow_value, marker=input_marker, is_velocity=True)

for marker in output_markers:
    bc_value = dolfinx.fem.Function(simulator.Q, name=f"outflow_p_{marker}")
    simulator.add_boundary(value=bc_value, marker=marker, is_velocity=False)


def run_stoke_solving_test():
    simulator.simulate_stoke(
        proj_dir=proj_dir, name='Stoke',
        ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        logger_dicts={},
        with_debug=True
    )


def run_ipcs_step_save_test():
    def post_function(sim: FluidSimulator, step: int, simulator_dir):
        if step % 200 == 0:
            step_dir = os.path.join(simulator_dir, f"step_{step}")
            if os.path.exists(step_dir):
                shutil.rmtree(step_dir)
            os.mkdir(step_dir)
            sim.save_result_to_pickle(step_dir)

    simulator.simulate_ipcs(
        proj_dir=proj_dir, name='ipcs', max_iter=1000, log_iter=50, tol=5e-6,
        data_convergence={},
        pre_function=None,
        post_function=post_function,
        with_debug=True
    )


def run_ipcs_step_load_test():
    from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder
    simulator.load_result('/home/admin123456/Desktop/work/topopt_exps/user_book5/simulate_ipcs/step_1000')
    u_n: dolfinx.fem.Function = simulator.equation_map['ipcs']['u_n']
    test_u_recorder = VTKRecorder('/home/admin123456/Desktop/work/topopt_exps/user_book5/simulate_ipcs/tst/u.pvd')
    test_u_recorder.write_function(u_n, 0)  # check whether result is reasonable


def run_navier_stoke_solving_test():
    simulator.simulate_navier_stoke(
        proj_dir=proj_dir, name='navierStoke',
        ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        logger_dicts={},
        with_debug=True
    )


def run_find_navier_stoke_initiation_test():
    simulator.find_navier_stoke_initiation(
        proj_dir=proj_dir, max_iter=5000, log_iter=50, trial_iter=100,
        ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        with_debug=True
    )


if __name__ == '__main__':
    run_find_navier_stoke_initiation_test()
