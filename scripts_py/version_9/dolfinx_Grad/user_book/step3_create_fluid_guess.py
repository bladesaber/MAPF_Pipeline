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
    parser = argparse.ArgumentParser(description="Find Good Naiver Stoke Initiation")
    parser.add_argument('--json_files', type=str, nargs='+', default=[])
    parser.add_argument('--init_mesh', type=int, default=0)
    args = parser.parse_args()
    return args


def find_guess_up(cfg, args):
    run_cfgs = []
    if cfg.get('recombine_cfgs', False):
        for simulate_cfg_name in cfg['recombine_cfgs']:
            with open(os.path.join(cfg['proj_dir'], simulate_cfg_name), 'r') as f:
                run_cfgs.append(json.load(f))
    else:
        run_cfgs = [cfg.copy()]
    del cfg

    # ------ init global parameter
    run_cfg = run_cfgs[0]
    if args.init_mesh:
        MeshUtils.msh_to_XDMF(
            name='model', dim=run_cfg['dim'],
            msh_file=os.path.join(run_cfg['proj_dir'], 'model.msh'),
            output_file=os.path.join(run_cfg['proj_dir'], 'model.xdmf')
        )
    condition_module = ImportTool.import_module(run_cfg['proj_dir'], run_cfg['condition_package_name'])

    for run_cfg in run_cfgs:
        domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
            file=os.path.join(run_cfg['proj_dir'], 'model.xdmf'),
            mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
        )

        ipcs_cfg = run_cfg['simulate_cfg']['ipcs']
        navier_stoke_cfg = run_cfg['simulate_cfg']['navier_stoke']

        input_markers = [int(marker) for marker in run_cfg['input_markers'].keys()]
        output_markers = run_cfg['output_markers']
        bry_markers = run_cfg['bry_free_markers'] + run_cfg['bry_fix_markers']

        condition_inflow_dict = {}
        for marker in input_markers:
            marker_fun_name = run_cfg['input_markers'][str(marker)]
            inflow_fun = ImportTool.get_module_function(condition_module, marker_fun_name)
            condition_inflow_dict[marker] = partial(inflow_fun, tdim=run_cfg['dim'])

        simulator = FluidSimulator(run_cfg['name'], domain, cell_tags, facet_tags)
        simulator.define_ipcs_equation(
            dt=ipcs_cfg['dt'],
            dynamic_viscosity=ipcs_cfg['dynamic_viscosity'],
            density=ipcs_cfg['density'],
            is_channel_fluid=ipcs_cfg['is_channel_fluid']
        )
        simulator.define_navier_stoke_equation(Re=navier_stoke_cfg['Re'])

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

        record_dir = os.path.join(run_cfg['proj_dir'], f"{run_cfg['name']}_record")
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)

        # ------ prepare ipcs log dict
        u_n, p_n = simulator.get_up('ipcs')
        inflow_dict = {}
        for marker in input_markers:
            inflow_dict[f"inflow_{marker}_p"] = dolfinx.fem.form(p_n * simulator.ds(marker))
        outflow_dict = {}
        for marker in output_markers:
            outflow_dict[f"outflow_{marker}_v"] = dolfinx.fem.form(dot(u_n, simulator.n_vec) * simulator.ds(marker))
        ipcs_logger_dicts = {'inflow': inflow_dict, 'outflow': outflow_dict}

        # ------ prepare data convergence log
        ipcs_data_convergence = {}
        for marker in input_markers:
            ipcs_data_convergence[f"inflow_{marker}_p"] = {
                'form': dolfinx.fem.form(p_n * simulator.ds(marker)), 'cur_value': 0.0, 'old_value': 0.0
            }
        for marker in output_markers:
            ipcs_data_convergence[f"outflow_{marker}_v"] = {
                'form': dolfinx.fem.form(dot(u_n, simulator.n_vec) * simulator.ds(marker)),
                'cur_value': 0.0, 'old_value': 0.0
            }

        # ------ prepare navier stoke log dict
        u_n, p_n = simulator.get_up('navier_stoke')
        inflow_dict = {}
        for marker in input_markers:
            inflow_dict[f"inflow_{marker}_p"] = dolfinx.fem.form(p_n * simulator.ds(marker))
        outflow_dict = {}
        for marker in output_markers:
            outflow_dict[f"outflow_{marker}_v"] = dolfinx.fem.form(dot(u_n, simulator.n_vec) * simulator.ds(marker))
        ns_logger_dicts = {'inflow': inflow_dict, 'outflow': outflow_dict}

        # ------ run
        simulator.find_navier_stoke_initiation(
            proj_dir=record_dir, max_iter=ipcs_cfg['max_iter'], log_iter=ipcs_cfg['log_iter'],
            trial_iter=ipcs_cfg['trial_iter'],
            ksp_option=navier_stoke_cfg['ksp_option'],
            with_debug=True,
            logger_dicts=ipcs_logger_dicts,
            ns_logger_dicts=ns_logger_dicts,
            data_convergence=ipcs_data_convergence,
            tol=ipcs_cfg['tol']
        )


def main():
    args = parse_args()

    for json_file in args.json_files:
        with open(json_file, 'r') as f:
            cfg: dict = json.load(f)
        find_guess_up(cfg, args)


if __name__ == '__main__':
    main()
