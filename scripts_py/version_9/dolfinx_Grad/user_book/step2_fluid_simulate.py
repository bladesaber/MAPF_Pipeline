import dolfinx
import ufl
import os
from functools import partial
from ufl import grad, dot, inner
import json
import argparse

from scripts_py.version_9.dolfinx_Grad.fluid_tools.fluid_simulator import FluidSimulator
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.user_book.step1_project_tool import ImportTool


def parse_args():
    parser = argparse.ArgumentParser(description="Fluid Simulation Tool")
    parser.add_argument('--json_file', type=str, default=None)
    parser.add_argument('--simulate_method', type=str, default=None)
    parser.add_argument('--init_mesh', type=int, default=True)
    args = parser.parse_args()
    return args


args = parse_args()
assert (args.json_file is not None) and (args.simulate_method is not None)
assert args.simulate_method in ['ipcs', 'naiver_stoke', 'stoke']

simulate_method = args.simulate_method
with open(args.json_file, 'r') as f:
    cfg = json.load(f)

if args.init_mesh:
    MeshUtils.msh_to_XDMF(
        name='model', dim=cfg['dim'],
        msh_file=os.path.join(cfg['proj_dir'], 'model.msh'), output_file=os.path.join(cfg['proj_dir'], 'model.xdmf'),
    )
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=os.path.join(cfg['proj_dir'], 'model.xdmf'),
    mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)

simulate_cfg = cfg['simulate_cfg']
input_markers = [int(marker) for marker in cfg['input_markers'].keys()]
output_markers = cfg['output_markers']
bry_markers = cfg['bry_free_markers'] + cfg['bry_fix_markers']

condition_module = ImportTool.import_module(cfg['proj_dir'], cfg['condition_package_name'])
condition_inflow_dict = {}
for marker in input_markers:
    marker_str = str(marker)
    marker_fun_name = cfg['input_markers'][marker_str]
    inflow_fun = ImportTool.get_module_function(condition_module, marker_fun_name)
    condition_inflow_dict[marker] = partial(inflow_fun, tdim=cfg['dim'])

simulator = FluidSimulator(domain, cell_tags, facet_tags)

# ------ Define Simulator
if simulate_method == 'ipcs':
    ipcs_cfg = simulate_cfg['ipcs']
    # body_force = ipcs_cfg['body_force']

    simulator.define_ipcs_equation(
        dt=ipcs_cfg['dt'],
        dynamic_viscosity=ipcs_cfg['dynamic_viscosity'],
        density=ipcs_cfg['density'],
        is_channel_fluid=ipcs_cfg['is_channel_fluid']
    )

elif simulate_method == 'naiver_stoke':
    naiver_stoke_cfg = simulate_cfg['naiver_stoke']
    simulator.define_navier_stoke_equation(Re=naiver_stoke_cfg['Re'])


elif simulate_method == 'stoke':
    stoke_cfg = simulate_cfg['stoke']
    simulator.define_stoke_equation()

else:
    raise NotImplementedError("[ERROR]: Non-Valid Simulate Method")

# ------ define boundary
for marker in bry_markers:
    bc_value = dolfinx.fem.Function(simulator.V, name=f"bry_u{marker}")
    simulator.add_boundary(value=bc_value, marker=marker, is_velocity=True)

for marker in condition_inflow_dict.keys():
    inflow_value = dolfinx.fem.Function(simulator.V, name='inflow_u')
    inflow_value.interpolate(condition_inflow_dict[marker])
    simulator.add_boundary(value=inflow_value, marker=marker, is_velocity=True)

for marker in output_markers:
    bc_value = dolfinx.fem.Function(simulator.Q, name=f"outflow_p_{marker}")
    simulator.add_boundary(value=bc_value, marker=marker, is_velocity=False)

# ------
if simulate_method == 'ipcs':
    ipcs_dict = simulator.equation_map['ipcs']
    u_n, p_n = ipcs_dict['u_n'], ipcs_dict['p_n']

elif simulate_method == 'naiver_stoke':
    nstoke_dict = simulator.equation_map['navier_stoke']
    up: dolfinx.fem.Function = nstoke_dict['up']
    u_n, p_n = ufl.split(up)

elif simulate_method == 'stoke':
    stoke_dict = simulator.equation_map['stoke']
    up: dolfinx.fem.Function = stoke_dict['up']
    u_n, p_n = ufl.split(up)

else:
    raise NotImplementedError

inflow_dict = {}
for marker in input_markers:
    inflow_dict[f"inflow_{marker}_p"] = dolfinx.fem.form(p_n * simulator.ds(marker))
outflow_dict = {}
for marker in output_markers:
    outflow_dict[f"outflow_{marker}_v"] = dolfinx.fem.form(dot(u_n, simulator.n_vec) * simulator.ds(marker))
logger_dicts = {
    'inflow': inflow_dict,
    'outflow': outflow_dict,
    'energy': {'energy': dolfinx.fem.form(inner(u_n, u_n) * ufl.dx)},
    'energy_loss': {'energy_loss': dolfinx.fem.form(inner(grad(u_n), grad(u_n)) * ufl.dx)}
}

if simulate_method == 'ipcs':
    data_convergence = {}
    for marker in output_markers:
        data_convergence[f"outflow_{marker}_v"] = {
            'form': dolfinx.fem.form(dot(u_n, simulator.n_vec) * simulator.ds(marker)),
            'cur_value': 0.0, 'old_value': 0.0
        }
    for marker in input_markers:
        data_convergence[f"inflow_{marker}_p"] = {
            'form': dolfinx.fem.form(p_n * simulator.ds(marker)),
            'cur_value': 0.0, 'old_value': 0.0
        }

    simulator.simulate_ipcs(
        proj_dir=cfg['proj_dir'], name='ipcs',
        log_iter=ipcs_cfg['log_iter'], max_iter=ipcs_cfg['max_iter'], tol=ipcs_cfg['tol'],
        data_convergence=data_convergence, logger_dicts=logger_dicts,
        with_debug=True
    )

elif simulate_method == 'naiver_stoke':
    simulator.simulate_navier_stoke(
        proj_dir=cfg['proj_dir'], name='NS',
        ksp_option=naiver_stoke_cfg['ksp_option'],
        logger_dicts=logger_dicts,
        with_debug=True
    )

elif simulate_method == 'stoke':
    simulator.simulate_stoke(
        proj_dir=cfg['proj_dir'], name='Stoke',
        ksp_option=stoke_cfg['ksp_option'],
        logger_dicts=logger_dicts,
        with_debug=True
    )
else:
    raise NotImplementedError
