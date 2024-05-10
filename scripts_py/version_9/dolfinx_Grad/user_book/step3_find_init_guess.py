import numpy as np
import dolfinx
import ufl
import os
from functools import partial
from ufl import grad, dot, inner, div
import pyvista
import argparse
import json

from scripts_py.version_9.dolfinx_Grad.fluid_tools.dolfin_simulator import FluidSimulator
from scripts_py.version_9.dolfinx_Grad.user_book.step1_project_tool import ImportTool
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.equation_solver import NonLinearProblemSolver


def parse_args():
    parser = argparse.ArgumentParser(description="Find Good Naiver Stoke Initiation")
    parser.add_argument('--json_files', type=str, nargs='+', default=[])
    parser.add_argument('--init_mesh', type=int, default=0)
    parser.add_argument('--ipcs_or_initGuess', type=str, default='initGuess')
    args = parser.parse_args()
    return args


def find_guess_up(json_file, args):
    with open(json_file, 'r') as f:
        cfg: dict = json.load(f)

    run_cfgs = []
    if cfg.get('recombine_cfgs', False):
        for simulate_cfg_name in cfg['recombine_cfgs']:
            json_file = os.path.join(cfg['proj_dir'], simulate_cfg_name)
            with open(json_file, 'r') as f:
                run_cfgs.append((json.load(f), json_file))
    else:
        run_cfgs = [(cfg.copy(), json_file)]
    del cfg

    # ------ init global parameter
    run_cfg = run_cfgs[0][0]
    if args.init_mesh:
        MeshUtils.msh_to_XDMF(
            name='model', dim=run_cfg['dim'],
            msh_file=os.path.join(run_cfg['proj_dir'], run_cfg['msh_file']),
            output_file=os.path.join(run_cfg['proj_dir'], run_cfg['xdmf_file'])
        )
    condition_module = ImportTool.import_module(run_cfg['proj_dir'], run_cfg['condition_package_name'])

    for run_cfg, cfg_path in run_cfgs:
        domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
            file=os.path.join(run_cfg['proj_dir'], run_cfg['xdmf_file']),
            mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
        )

        input_markers = [int(marker) for marker in run_cfg['input_markers'].keys()]
        output_markers = run_cfg['output_markers']
        bry_markers = run_cfg['bry_free_markers'] + run_cfg['bry_fix_markers']

        condition_inflow_dict = {}
        for marker in input_markers:
            marker_fun_name = run_cfg['input_markers'][str(marker)]
            inflow_fun = ImportTool.get_module_function(condition_module, marker_fun_name)
            condition_inflow_dict[marker] = partial(inflow_fun, tdim=run_cfg['dim'])

        simulator = FluidSimulator(run_cfg['name'], domain, cell_tags, facet_tags)

        # ------ prepare ipcs
        if args.ipcs_or_initGuess == 'ipcs':
            ipcs_cfg = run_cfg['simulate_cfg']['ipcs']
            simulator.define_ipcs_equation(
                dt=ipcs_cfg['dt'],
                dynamic_viscosity=ipcs_cfg['dynamic_viscosity'],
                density=ipcs_cfg['density'],
                is_channel_fluid=ipcs_cfg['is_channel_fluid']
            )

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

        # ------ prepare navier stoke
        navier_stoke_cfg = run_cfg['simulate_cfg']['navier_stoke']
        simulator.define_navier_stoke_equation(Re=navier_stoke_cfg['Re'])

        u_n, p_n = simulator.get_up('navier_stoke')
        inflow_dict = {}
        for marker in input_markers:
            inflow_dict[f"inflow_{marker}_p"] = dolfinx.fem.form(p_n * simulator.ds(marker))
        outflow_dict = {}
        for marker in output_markers:
            outflow_dict[f"outflow_{marker}_v"] = dolfinx.fem.form(dot(u_n, simulator.n_vec) * simulator.ds(marker))
        ns_logger_dicts = {'inflow': inflow_dict, 'outflow': outflow_dict}

        # ------ define boundary
        for marker in bry_markers:
            simulator.add_boundary(name=f"bry_u{marker}", value=0.0, marker=marker, is_velocity=True)

        for marker in condition_inflow_dict.keys():
            simulator.add_boundary(
                name='inflow_u', value=condition_inflow_dict[marker], marker=marker, is_velocity=True
            )

        for marker in output_markers:
            simulator.add_boundary(f"outflow_p_{marker}", value=0.0, marker=marker, is_velocity=False)

        record_dir = os.path.join(run_cfg['proj_dir'], f"{run_cfg['name']}_record")
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)

        if args.ipcs_or_initGuess == 'ipcs':
            res_dict = simulator.find_navier_stoke_initiation(
                proj_dir=record_dir, max_iter=ipcs_cfg['max_iter'], log_iter=ipcs_cfg['log_iter'],
                trial_iter=ipcs_cfg['trial_iter'],
                snes_option=navier_stoke_cfg['snes_option'],
                snes_criterion=navier_stoke_cfg['criterion'],
                ksp_option=navier_stoke_cfg['ksp_option'],
                with_debug=True,
                logger_dicts=ipcs_logger_dicts,
                ns_logger_dicts=ns_logger_dicts,
                data_convergence=ipcs_data_convergence,
                tol=ipcs_cfg['tol']
            )

        else:
            assert (run_cfg['velocity_init_pkl'] is not None) and (run_cfg['pressure_init_pkl'] is not None)
            simulator.load_result(
                file_info={'u_pkl_file': run_cfg['velocity_init_pkl'], 'p_pkl_file': run_cfg['pressure_init_pkl']}
            )
            res_dict = simulator.run_navier_stoke_process(
                proj_dir=record_dir, name='NS', guass_up=None,
                snes_option=navier_stoke_cfg['snes_option'],
                snes_criterion=navier_stoke_cfg['criterion'],
                ksp_option=navier_stoke_cfg['ksp_option'],
                logger_dicts=ns_logger_dicts,
                with_debug=True,  # Must Be True
                with_monitor=True,
            )

            if NonLinearProblemSolver.is_converged(res_dict['converged_reason']):
                save_dir = os.path.join(res_dict['simulator_dir'], 'res_pkl')
                os.mkdir(save_dir)
                simulator.save_result(save_dir, methods=['navier_stoke'])

                run_cfg['velocity_init_pkl'] = os.path.join(save_dir, 'navier_stoke_u.pkl')
                run_cfg['pressure_init_pkl'] = os.path.join(save_dir, 'navier_stoke_p.pkl')

                with open(cfg_path, 'w') as f:
                    json.dump(run_cfg, f, indent=4)
                print('[INFO] Success and Update Cfg init_pkl file')

            else:
                print(f"[ERROR]: Fail for Reason {res_dict['converged_reason']}")


def main():
    args = parse_args()

    for json_file in args.json_files:
        find_guess_up(json_file, args)


if __name__ == '__main__':
    main()
