import numpy as np
import dolfinx
import ufl
import os
from functools import partial
from ufl import grad, dot, inner, div
import pyvista
import argparse
import json

from scripts_py.version_9.dolfinx_Grad.fluid_tools.fluid_simulator import FluidSimulator
from scripts_py.version_9.dolfinx_Grad.user_book.step1_project_tool import ImportTool
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils


def parse_args():
    parser = argparse.ArgumentParser(description="Fluid Simulation Tool")
    parser.add_argument('--proj_dir', type=str, default=None)
    parser.add_argument('--json_names', type=str, nargs='+', default=[])
    parser.add_argument('--init_mesh', type=int, default=0)
    args = parser.parse_args()
    return args


args = parse_args()
assert len(args.json_names) > 0 and (args.proj_dir is not None)

cfgs = []
for cfg_name in args.json_names:
    with open(os.path.join(args.proj_dir, cfg_name), 'r') as f:
        cfgs.append(json.load(f))

init_mesh = args.init_mesh
for cfg in cfgs:
    if init_mesh:
        MeshUtils.msh_to_XDMF(
            name='model', dim=cfg['dim'],
            msh_file=os.path.join(cfg['proj_dir'], 'model.msh'),
            output_file=os.path.join(cfg['proj_dir'], 'model.xdmf'),
        )
        init_mesh = False  # Just imit once

    domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
        file=os.path.join(cfg['proj_dir'], 'model.xdmf'),
        mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
    )

    simulate_cfg = cfg['simulate_cfg']
    ipcs_cfg = simulate_cfg['ipcs']
    naiver_stoke_cfg = simulate_cfg['naiver_stoke']

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

    simulator.define_navier_stoke_equation(Re=naiver_stoke_cfg['Re'])
    simulator.define_ipcs_equation(
        dt=ipcs_cfg['dt'],
        dynamic_viscosity=ipcs_cfg['dynamic_viscosity'],
        density=ipcs_cfg['density'],
        is_channel_fluid=ipcs_cfg['is_channel_fluid']
    )

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

    record_dir = os.path.join(cfg['proj_dir'], f"{cfg['name']}_record")
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)

    simulator.find_navier_stoke_initiation(
        proj_dir=record_dir, max_iter=ipcs_cfg['max_iter'], log_iter=ipcs_cfg['log_iter'],
        trial_iter=ipcs_cfg['trial_iter'],
        ksp_option=naiver_stoke_cfg['ksp_option'],
        with_debug=True
    )

