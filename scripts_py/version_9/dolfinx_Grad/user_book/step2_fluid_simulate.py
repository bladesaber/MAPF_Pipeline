import shutil
import dolfinx
import ufl
import os
from functools import partial
from ufl import grad, dot, inner, sqrt
import json
import argparse

from scripts_py.version_9.dolfinx_Grad.fluid_tools.dolfin_simulator import FluidSimulator
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.user_book.step1_project_tool import ImportTool
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import AssembleUtils
from scripts_py.version_9.dolfinx_Grad.fluid_tools.openfoam_simulator import OpenFoamSimulator


def parse_args():
    parser = argparse.ArgumentParser(description="Fluid Simulation Tool")
    parser.add_argument('--json_files', type=str, nargs='+', default=[])
    parser.add_argument('--simulate_method', type=str, default=None)
    parser.add_argument('--init_mesh', type=int, default=0)
    parser.add_argument('--xdmf_tag', type=str, default=None)
    parser.add_argument('--save_result', type=int, default=0)
    args = parser.parse_args()
    return args


def dolfin_simulate(cfg, args, **kwargs):
    simulate_method = args.simulate_method

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
        assert args.xdmf_tag is not None
        MeshUtils.msh_to_XDMF(
            name='model', dim=run_cfg['dim'],
            msh_file=os.path.join(run_cfg['proj_dir'], run_cfg['msh_file']),
            output_file=os.path.join(run_cfg['proj_dir'], run_cfg['xdmf_file'])
        )

    condition_module = ImportTool.import_module(run_cfg['proj_dir'], run_cfg['condition_package_name'])

    for run_cfg in run_cfgs:
        domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
            file=os.path.join(run_cfg['proj_dir'], run_cfg[args.xdmf_tag]),
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

        if simulate_method == 'ipcs':
            simulate_cfg = run_cfg['simulate_cfg']['ipcs']
            simulator.define_ipcs_equation(
                dt=simulate_cfg['dt'],
                dynamic_viscosity=simulate_cfg['dynamic_viscosity'],
                density=simulate_cfg['density'],
                is_channel_fluid=simulate_cfg['is_channel_fluid']
            )
        elif simulate_method == 'navier_stoke':
            simulate_cfg = run_cfg['simulate_cfg']['navier_stoke']
            simulator.define_navier_stoke_equation(Re=simulate_cfg['Re'])
        elif simulate_method == 'stoke':
            simulate_cfg = run_cfg['simulate_cfg']['stoke']
            simulator.define_stoke_equation()
        else:
            return -1

        # ------ define boundary
        for marker in bry_markers:
            simulator.add_boundary(name=f"bry_u{marker}", value=0.0, marker=marker, is_velocity=True)

        for marker in condition_inflow_dict.keys():
            simulator.add_boundary(
                name='inflow_u', value=condition_inflow_dict[marker], marker=marker, is_velocity=True
            )

        for marker in output_markers:
            simulator.add_boundary(f"outflow_p_{marker}", value=0.0, marker=marker, is_velocity=False)

        # ------ define log dict
        u_n, p_n = simulator.get_up(simulate_method)
        force_dict, pressure_dict, norm_flow_dict, flow_dict = {}, {}, {}, {}

        for marker in input_markers:
            force_dict[f"force_{marker}"] = dolfinx.fem.form(p_n * simulator.ds(marker))
            area_value = AssembleUtils.assemble_scalar(
                dolfinx.fem.form(dolfinx.fem.Constant(simulator.domain, 1.0) * simulator.ds(marker))
            )
            pressure_dict[f"pressure_{marker}"] = dolfinx.fem.form((1.0 / area_value) * p_n * simulator.ds(marker))

        for marker in output_markers:
            norm_flow_dict[f"normFlow_{marker}"] = dolfinx.fem.form(dot(u_n, simulator.n_vec) * simulator.ds(marker))
            flow_dict[f"flow_{marker}"] = dolfinx.fem.form(sqrt(dot(u_n, u_n)) * simulator.ds(marker))

        logger_dicts = {
            'force': force_dict,
            'norm_flow': norm_flow_dict,
            'pressure': pressure_dict,
            'flow': flow_dict,
        }

        record_dir = os.path.join(run_cfg['proj_dir'], f"{run_cfg['name']}_record")
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)

        if simulate_method == 'ipcs':
            data_convergence = {}
            for marker in input_markers:
                data_convergence[f"inflow_{marker}_p"] = {
                    'form': dolfinx.fem.form(p_n * simulator.ds(marker)), 'cur_value': 0.0, 'old_value': 0.0
                }
            for marker in output_markers:
                data_convergence[f"outflow_{marker}_v"] = {
                    'form': dolfinx.fem.form(dot(u_n, simulator.n_vec) * simulator.ds(marker)),
                    'cur_value': 0.0, 'old_value': 0.0
                }
            res_dict = simulator.run_ipcs_process(
                proj_dir=record_dir, name='ipcs',
                log_iter=simulate_cfg['log_iter'], max_iter=simulate_cfg['max_iter'], tol=simulate_cfg['tol'],
                data_convergence=data_convergence, logger_dicts=logger_dicts, with_debug=True
            )

        elif simulate_method == 'navier_stoke':
            res_dict = simulator.run_navier_stoke_process(
                proj_dir=record_dir, name='NS',
                snes_option=simulate_cfg['snes_option'],
                snes_criterion=simulate_cfg['criterion'],
                ksp_option=simulate_cfg['ksp_option'],
                logger_dicts=logger_dicts,
                with_debug=True
            )

        elif simulate_method == 'stoke':
            res_dict = simulator.run_stoke_process(
                proj_dir=record_dir, name='Stoke', ksp_option=simulate_cfg['ksp_option'], logger_dicts=logger_dicts,
                with_debug=True,
                record_mat_dir=os.path.join(run_cfg['proj_dir'], 'debug_dir')
            )
        else:
            return -1

        if args.save_result:
            save_dir = os.path.join(res_dict['simulator_dir'], 'res_pkl')
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.mkdir(save_dir)
            simulator.save_result(save_dir, methods=[args.simulate_method])


def openfoam_simulate(cfg, args, **kwargs):
    run_cfgs = []
    if cfg.get('recombine_cfgs', False):
        for simulate_cfg_name in cfg['recombine_cfgs']:
            with open(os.path.join(cfg['proj_dir'], simulate_cfg_name), 'r') as f:
                run_cfgs.append(json.load(f))
    else:
        run_cfgs = [cfg.copy()]
    del cfg

    for run_cfg in run_cfgs:
        simulator = OpenFoamSimulator(run_cfg)

        record_dir = os.path.join(run_cfg['proj_dir'], f"{run_cfg['name']}_record")
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)

        tmp_dir = os.path.join(record_dir, 'simulate_openfoam')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)

        simulator.run_simulate(
            tmp_dir, orig_msh_file=os.path.join(run_cfg['proj_dir'], run_cfg['msh_file']), convert_msh2=True
        )


def main():
    args = parse_args()

    for i, json_file in enumerate(args.json_files):
        with open(json_file, 'r') as f:
            cfg: dict = json.load(f)

            if args.simulate_method in ['ipcs', 'navier_stoke', 'stoke']:
                dolfin_simulate(cfg, args, debug=i)
            else:
                openfoam_simulate(cfg, args)


if __name__ == '__main__':
    main()
