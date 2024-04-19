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
from scripts_py.version_9.dolfinx_Grad.equation_solver import LinearProblemSolver, NonLinearProblemSolver
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder, XDMFRecorder


def parse_args():
    parser = argparse.ArgumentParser(description="Fluid Simulation Tool")
    parser.add_argument('--json_file', type=str, default=None)
    parser.add_argument('--simulate_method', type=str, default=None)
    parser.add_argument('--csv_file', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.simulate_method in ['navier_stoke', 'stoke']

    simulate_method = args.simulate_method

    with open(args.json_file, 'r') as f:
        run_cfg = json.load(f)

    condition_module = ImportTool.import_module(run_cfg['proj_dir'], run_cfg['condition_package_name'])

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

    if simulate_method == 'navier_stoke':
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

    # simulator.load_result(csv_file=args.csv_file)
    simulator.load_initiation_pickle(
        u_pickle_file='/home/admin123456/Desktop/work/topopt_exps/user_proj_7/mesh_A_tst/mesh_A_record/simulate_ipcs/res_pkl/ipcs_u.pkl',
        p_pickle_file='/home/admin123456/Desktop/work/topopt_exps/user_proj_7/mesh_A_tst/mesh_A_record/simulate_ipcs/res_pkl/ipcs_p.pkl'
    )

    up: dolfinx.fem.Function = simulator.equation_map[simulate_method]['up']
    # vel_recorder = VTKRecorder(os.path.join(args.save_dir, 'velocity.pvd'))
    # vel_recorder.write_function(up.sub(0).collapse(), step=0)
    # pressure_recorder = VTKRecorder(os.path.join(args.save_dir, 'pressure.pvd'))
    # pressure_recorder.write_function(up.sub(1).collapse(), step=0)

    if simulate_method == 'navier_stoke':
        res_dict = NonLinearProblemSolver.equation_investigation(
            lhs_form=simulator.equation_map['navier_stoke']['lhs'],
            bcs=simulator.equation_map['navier_stoke']['bcs'],
            uh=up
        )

    elif simulate_method == 'stoke':
        res_dict = LinearProblemSolver.equation_investigation(
            a_form=simulator.equation_map['stoke']['lhs_form'],
            L_form=simulator.equation_map['stoke']['rhs_form'],
            bcs=simulator.equation_map['stoke']['bcs'],
            uh=up
        )

    else:
        return -1

    info = '[INFO] InitEnv '
    for key in res_dict.keys():
        info += f"{key}:{res_dict[key]} "
    print(info)

    if simulate_method == 'navier_stoke':
        res_dict = simulator.simulate_navier_stoke(
            proj_dir=args.save_dir, name='NS',
            snes_setting=simulate_cfg['snes_setting'], ksp_option=simulate_cfg['ksp_option'],
            with_debug=True
        )

    else:
        simulator.simulate_stoke(
            proj_dir=args.save_dir, name='Stoke', ksp_option=simulate_cfg['ksp_option'], with_debug=True
        )

    if simulate_method == 'navier_stoke':
        res_dict = NonLinearProblemSolver.equation_investigation(
            lhs_form=dolfinx.fem.form(simulator.equation_map['navier_stoke']['lhs']),
            bcs=simulator.equation_map['navier_stoke']['bcs'],
            uh=up
        )

    else:
        res_dict = LinearProblemSolver.equation_investigation(
            a_form=dolfinx.fem.form(simulator.equation_map['stoke']['lhs_form']),
            L_form=dolfinx.fem.form(simulator.equation_map['stoke']['rhs_form']),
            bcs=simulator.equation_map['stoke']['bcs'],
            uh=up
        )

    info = '[INFO] FinalEnv '
    for key in res_dict.keys():
        info += f"{key}:{res_dict[key]} "
    print(info)


if __name__ == '__main__':
    main()
