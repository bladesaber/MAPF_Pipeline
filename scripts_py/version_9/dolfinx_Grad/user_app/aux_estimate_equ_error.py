import shutil
import dolfinx
import pyvista
import ufl
import os
from functools import partial
from ufl import grad, dot, inner, sqrt
import json
import argparse
import numpy as np

from scripts_py.version_9.dolfinx_Grad.fluid_tools.dolfin_simulator import DolfinSimulator
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.user_book.step1_project_tool import ImportTool
from scripts_py.version_9.dolfinx_Grad.equation_solver import LinearProblemSolver, NonLinearProblemSolver
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import AssembleUtils


def parse_args():
    parser = argparse.ArgumentParser(description="Fluid Equation Investigation")
    parser.add_argument('--json_files', type=str, nargs='+', default=None)
    parser.add_argument('--xdmf_tag', type=str, default=None)
    parser.add_argument('--velocity_tag', type=str, default=None)
    parser.add_argument('--pressure_tag', type=str, default=None)
    parser.add_argument('--software', type=str, default='dolfinx')
    args = parser.parse_args()
    return args


def estimate_error_dolfin(json_file, args):
    simulate_method = 'navier_stoke'
    with open(json_file, 'r') as f:
        cfg: dict = json.load(f)

    run_cfgs = []
    if cfg.get('recombine_cfgs', False):
        for simulate_cfg_name in cfg['recombine_cfgs']:
            json_file = os.path.join(cfg['proj_dir'], simulate_cfg_name)
            with open(json_file, 'r') as f:
                run_cfgs.append(json.load(f))
    else:
        run_cfgs = [cfg.copy()]
    del cfg

    run_cfg = run_cfgs[0]
    condition_module = ImportTool.import_module(run_cfg['proj_dir'], run_cfg['condition_package_name'])

    for run_cfg in run_cfgs:
        domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
            os.path.join(run_cfg['proj_dir'], run_cfg[args.xdmf_tag]), 'model', 'model_cells', 'model_facets'
        )

        input_markers = [int(marker) for marker in run_cfg['input_markers'].keys()]
        output_markers = run_cfg['output_markers']
        bry_markers = run_cfg['bry_free_markers'] + run_cfg['bry_fix_markers']

        condition_inflow_dict = {}
        for marker in input_markers:
            marker_fun_name = run_cfg['input_markers'][str(marker)]
            inflow_fun = ImportTool.get_module_function(condition_module, marker_fun_name)
            condition_inflow_dict[marker] = partial(inflow_fun, tdim=run_cfg['dim'])

        simulator = DolfinSimulator(run_cfg['name'], domain, cell_tags, facet_tags)

        simulate_cfg = run_cfg['simulate_cfg']['navier_stoke']
        simulator.define_navier_stoke_equation(nu_value=simulate_cfg['kinematic_viscosity_nu'])

        # ------ define boundary
        for marker in bry_markers:
            simulator.add_boundary(name=f"bry_u{marker}", value=0.0, marker=marker, is_velocity=True)

        for marker in condition_inflow_dict.keys():
            simulator.add_boundary('inflow_u', value=condition_inflow_dict[marker], marker=marker, is_velocity=True)

        for marker in output_markers:
            simulator.add_boundary(f"outflow_p_{marker}", value=0.0, marker=marker, is_velocity=False)

        simulator.load_result(
            file_info={'u_pkl_file': run_cfg[args.velocity_tag], 'p_pkl_file': run_cfg[args.pressure_tag]},
            load_type='pkl'
        )

        # ------ output result
        up: dolfinx.fem.Function = simulator.equation_map[simulate_method]['up']
        u_n, p_n = simulator.get_up(simulate_method)

        force_dict, pressure_dict, norm_flow_dict, flow_dict = {}, {}, {}, {}
        for marker in input_markers:
            # force_dict[f"force_{marker}"] = dolfinx.fem.form(p_n * simulator.ds(marker))
            area_value = AssembleUtils.assemble_scalar(
                dolfinx.fem.form(dolfinx.fem.Constant(simulator.domain, 1.0) * simulator.ds(marker))
            )
            pressure_dict[f"pressure_{marker}"] = dolfinx.fem.form((1.0 / area_value) * p_n * simulator.ds(marker))
        for marker in output_markers:
            norm_flow_dict[f"normFlow_{marker}"] = dolfinx.fem.form(dot(u_n, simulator.n_vec) * simulator.ds(marker))
            # flow_dict[f"flow_{marker}"] = dolfinx.fem.form(sqrt(dot(u_n, u_n)) * simulator.ds(marker))

        logger_dicts = {
            # 'force': force_dict, 'flow': flow_dict,
            'norm_flow': norm_flow_dict, 'pressure': pressure_dict,
        }

        print(f"[INFO]: config: {run_cfg['name']}")
        for tag_name in logger_dicts:
            print(f"{tag_name}:")
            for name in logger_dicts[tag_name]:
                print(f"---{name}: {AssembleUtils.assemble_scalar(logger_dicts[tag_name][name])}")

        res_dict = simulator.estimate_equation_lost(method=simulate_method)
        print(f"[INFO] Navier Stoke Equation Error:{res_dict['max_error']}")

        record_dir = os.path.join(run_cfg['proj_dir'], f"{run_cfg['name']}_record")
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)
        save_dir = os.path.join(record_dir, 'equ_estimate')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)

        vel_recorder = VTKRecorder(os.path.join(save_dir, 'velocity.pvd'))
        vel_recorder.write_function(up.sub(0).collapse(), step=0)
        pressure_recorder = VTKRecorder(os.path.join(save_dir, 'pressure.pvd'))
        pressure_recorder.write_function(up.sub(1).collapse(), step=0)

        # # ------ Recompute
        # print('\n[INFO]: Recompute')
        # opt_cfg = run_cfg['optimize_cfg']
        # simulator.simulate_navier_stoke_equation(
        #     snes_option=opt_cfg['snes_option'],
        #     ksp_option=opt_cfg['state_ksp_option'],
        #     criterion=opt_cfg['snes_criterion'],
        #     with_debug=True, with_monitor=True
        # )
        #
        # for tag_name in logger_dicts:
        #     print(f"{tag_name}:")
        #     for name in logger_dicts[tag_name]:
        #         print(f"---{name}: {AssembleUtils.assemble_scalar(logger_dicts[tag_name][name])}")
        #
        # res_dict = simulator.estimate_equation_lost(method=simulate_method)
        # print(f"[INFO] {run_cfg['name']} Navier Stoke Equation Error:{res_dict['max_error']}")

        print('/********************************************************/')


def estimate_error_openfoam(json_file, args):
    simulate_method = 'navier_stoke'
    with open(json_file, 'r') as f:
        cfg: dict = json.load(f)

    run_cfgs = []
    if cfg.get('recombine_cfgs', False):
        for simulate_cfg_name in cfg['recombine_cfgs']:
            json_file = os.path.join(cfg['proj_dir'], simulate_cfg_name)
            with open(json_file, 'r') as f:
                run_cfgs.append(json.load(f))
    else:
        run_cfgs = [cfg.copy()]
    del cfg

    run_cfg = run_cfgs[0]
    condition_module = ImportTool.import_module(run_cfg['proj_dir'], run_cfg['condition_package_name'])

    for run_cfg in run_cfgs:
        domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
            os.path.join(run_cfg['proj_dir'], run_cfg[args.xdmf_tag]), 'model', 'model_cells', 'model_facets'
        )
        input_markers = [int(marker) for marker in run_cfg['input_markers'].keys()]
        output_markers = run_cfg['output_markers']
        bry_markers = run_cfg['bry_free_markers'] + run_cfg['bry_fix_markers']

        condition_inflow_dict = {}
        for marker in input_markers:
            marker_fun_name = run_cfg['input_markers'][str(marker)]
            inflow_fun = ImportTool.get_module_function(condition_module, marker_fun_name)
            condition_inflow_dict[marker] = partial(inflow_fun, tdim=run_cfg['dim'])

        simulator = DolfinSimulator(
            run_cfg['name'], domain, cell_tags, facet_tags, velocity_order=1, pressure_order=1
        )
        simulate_cfg = run_cfg['simulate_cfg']['navier_stoke']
        simulator.define_navier_stoke_equation(nu_value=simulate_cfg['kinematic_viscosity_nu'], constant_nu=False)

        # ------ define boundary
        for marker in bry_markers:
            simulator.add_boundary(name=f"bry_u{marker}", value=0.0, marker=marker, is_velocity=True)

        for marker in condition_inflow_dict.keys():
            simulator.add_boundary('inflow_u', value=condition_inflow_dict[marker], marker=marker, is_velocity=True)

        for marker in output_markers:
            simulator.add_boundary(f"outflow_p_{marker}", value=0.0, marker=marker, is_velocity=False)

        simulator.load_result(file_info={'vtk_file': run_cfg['openfoam_vtk']}, load_type='vtk')

        up: dolfinx.fem.Function = simulator.equation_map[simulate_method]['up']
        u_n, p_n = simulator.get_up(simulate_method)

        pressure_dict, norm_flow_dict = {}, {}
        for marker in input_markers:
            area_value = AssembleUtils.assemble_scalar(
                dolfinx.fem.form(dolfinx.fem.Constant(simulator.domain, 1.0) * simulator.ds(marker))
            )
            pressure_dict[f"pressure_{marker}"] = dolfinx.fem.form((1.0 / area_value) * p_n * simulator.ds(marker))
        for marker in output_markers:
            norm_flow_dict[f"normFlow_{marker}"] = dolfinx.fem.form(dot(u_n, simulator.n_vec) * simulator.ds(marker))

        logger_dicts = {'norm_flow': norm_flow_dict, 'pressure': pressure_dict}
        for tag_name in logger_dicts:
            print(f"{tag_name}:")
            for name in logger_dicts[tag_name]:
                print(f"---{name}: {AssembleUtils.assemble_scalar(logger_dicts[tag_name][name])}")

        res_dict = simulator.estimate_equation_lost(method=simulate_method)
        print(f"[INFO] {run_cfg['name']} Navier Stoke Equation Error:{res_dict['max_error']}")

        record_dir = os.path.join(run_cfg['proj_dir'], f"{run_cfg['name']}_record")
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)

        save_dir = os.path.join(record_dir, 'equ_estimate')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)

        vel_recorder = VTKRecorder(os.path.join(save_dir, 'velocity.pvd'))
        vel_recorder.write_function(up.sub(0).collapse(), step=0)
        pressure_recorder = VTKRecorder(os.path.join(save_dir, 'pressure.pvd'))
        pressure_recorder.write_function(up.sub(1).collapse(), step=0)

        # ------ Recompute
        # print('\n[INFO]: Recompute')
        # opt_cfg = run_cfg['optimize_cfg']
        # res_dict = simulator.simulate_navier_stoke_equation(
        #     snes_option=opt_cfg['snes_option'],
        #     ksp_option=opt_cfg['state_ksp_option'],
        #     criterion=opt_cfg['snes_criterion'],
        #     with_debug=True, with_monitor=True,
        #     A_assemble_method='Identity_row'
        # )
        #
        # for tag_name in logger_dicts:
        #     print(f"{tag_name}:")
        #     for name in logger_dicts[tag_name]:
        #         print(f"---{name}: {AssembleUtils.assemble_scalar(logger_dicts[tag_name][name])}")
        #
        # res_dict = simulator.estimate_equation_lost(method=simulate_method)
        # print(f"[INFO] {run_cfg['name']} Navier Stoke Equation Error:{res_dict['max_error']}")

        print('/********************************************************/')


def main():
    args = parse_args()

    if args.software == 'dolfinx':
        for json_file in args.json_files:
            estimate_error_dolfin(json_file, args)

    elif args.software == 'openfoam':
        for json_file in args.json_files:
            estimate_error_openfoam(json_file, args)


if __name__ == '__main__':
    main()
