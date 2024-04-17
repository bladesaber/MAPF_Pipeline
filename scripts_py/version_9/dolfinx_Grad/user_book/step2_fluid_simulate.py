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


# todo
#   1.引用其他求解软件的结果，然后用特定阶的有限元方法来插值作为init
#   2.引用其他求解器例如scipy
#   3.网格划分可能是比较大的影响因素
#   4.改用其他精确解方法
#   5.改用其他粗糙解方法

def parse_args():
    parser = argparse.ArgumentParser(description="Fluid Simulation Tool")
    parser.add_argument('--json_files', type=str, nargs='+', default=[])
    parser.add_argument('--simulate_method', type=str, default=None)
    parser.add_argument('--init_mesh', type=int, default=0)
    args = parser.parse_args()
    return args


def simulate(cfg, args, **kwargs):
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
            bc_value = dolfinx.fem.Function(simulator.V, name=f"bry_u{marker}")
            simulator.add_boundary(value=bc_value, marker=marker, is_velocity=True)

        for marker in condition_inflow_dict.keys():
            inflow_value = dolfinx.fem.Function(simulator.V, name='inflow_u')
            inflow_value.interpolate(condition_inflow_dict[marker])
            simulator.add_boundary(value=inflow_value, marker=marker, is_velocity=True)

        for marker in output_markers:
            bc_value = dolfinx.fem.Function(simulator.Q, name=f"outflow_p_{marker}")
            simulator.add_boundary(value=bc_value, marker=marker, is_velocity=False)

        # ------ define log dict
        u_n, p_n = simulator.get_up(simulate_method)
        inflow_dict = {}
        for marker in input_markers:
            inflow_dict[f"inflow_{marker}_p"] = dolfinx.fem.form(p_n * simulator.ds(marker))
        outflow_dict = {}
        for marker in output_markers:
            outflow_dict[f"outflow_{marker}_v"] = dolfinx.fem.form(dot(u_n, simulator.n_vec) * simulator.ds(marker))
        logger_dicts = {
            'inflow': inflow_dict,
            'outflow': outflow_dict,
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
            simulator.simulate_ipcs(
                proj_dir=record_dir, name='ipcs',
                log_iter=simulate_cfg['log_iter'], max_iter=simulate_cfg['max_iter'], tol=simulate_cfg['tol'],
                data_convergence=data_convergence, logger_dicts=logger_dicts, with_debug=True
            )

        elif simulate_method == 'navier_stoke':
            simulator.simulate_navier_stoke(
                proj_dir=record_dir, name='NS', ksp_option=simulate_cfg['ksp_option'], logger_dicts=logger_dicts,
                with_debug=True
            )

        elif simulate_method == 'stoke':
            simulator.simulate_stoke(
                proj_dir=record_dir, name='Stoke', ksp_option=simulate_cfg['ksp_option'], logger_dicts=logger_dicts,
                with_debug=True,
                record_mat_dir=os.path.join(run_cfg['proj_dir'], 'debug_dir')
            )
        else:
            return -1


def main():
    args = parse_args()
    assert args.simulate_method in ['ipcs', 'navier_stoke', 'stoke']

    for i, json_file in enumerate(args.json_files):
        with open(json_file, 'r') as f:
            cfg: dict = json.load(f)
            simulate(cfg, args, debug=i)


if __name__ == '__main__':
    main()
