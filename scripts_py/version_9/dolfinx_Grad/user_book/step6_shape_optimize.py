"""
Note
    1.边界层导致会导致网格不均衡，优化时不要划分边界层
    2.相对于Poincare-Steklov operator，default方法更稳定
    3.所有网格大小越均匀，稳定性越高
"""

import os
import shutil
import numpy as np
import dolfinx
import ufl
from ufl import grad, dot, inner, div, sqrt
from functools import partial
import argparse
import json
import pyvista
from typing import List, Union

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.fluid_tools.fluid_shape_opt import FluidConditionalModel, \
    FluidShapeRecombineLayer
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization, \
    VolumeRegularization
from scripts_py.version_9.dolfinx_Grad.recorder_utils import TensorBoardRecorder
from scripts_py.version_9.dolfinx_Grad.collision_objs import ObstacleCollisionObj
from scripts_py.version_9.dolfinx_Grad.surface_fields import SparsePointsRegularization
from scripts_py.version_9.dolfinx_Grad.user_book.step1_project_tool import ImportTool
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import AssembleUtils
from scripts_py.version_9.dolfinx_Grad.fluid_tools.dolfin_simulator import DolfinSimulator
from scripts_py.version_9.dolfinx_Grad.fluid_tools.openfoam_simulator import OpenFoamSimulator


def parse_args():
    parser = argparse.ArgumentParser(description="Find Good Naiver Stoke Initiation")
    parser.add_argument('--json_files', type=str, nargs='+', default=[])
    parser.add_argument('--with_debug', type=int, default=0)
    parser.add_argument('--load_guess_res', type=int, default=0)
    parser.add_argument('--with_obstacle', type=int, default=0)
    parser.add_argument('--obstacle_json_files', type=str, nargs='+', default=[])
    parser.add_argument('--res_dir', type=str, default=None)
    parser.add_argument('--simulate_method', type=str, default=None)
    args = parser.parse_args()
    return args


def load_obstacle(args):
    if args.with_obstacle:
        obs_recursive_files: List[str] = args.obstacle_json_files
        for run_cfg_file in args.json_files:
            with open(run_cfg_file, 'r') as f:
                run_cfg: dict = json.load(f)

            if not run_cfg.get('recombine_cfgs', False):
                if run_cfg['obstacle_dir'] is not None:
                    for obs_name in run_cfg['obstacle_names']:
                        obs_json_file = os.path.join(run_cfg['obstacle_dir'], f"{obs_name}.json")
                        obs_recursive_files.append(obs_json_file)
            else:
                for sub_cfg_name in run_cfg['recombine_cfgs']:
                    with open(os.path.join(run_cfg['proj_dir'], sub_cfg_name), 'r') as f:
                        run_cfg = json.load(f)
                        if run_cfg['obstacle_dir'] is not None:
                            for obs_name in run_cfg['obstacle_names']:
                                obs_json_file = os.path.join(run_cfg['obstacle_dir'], f"{obs_name}.json")
                                obs_recursive_files.append(obs_json_file)

        obs_objs, obs_history = [], []
        for obs_json_file in obs_recursive_files:
            if obs_json_file in obs_history:
                continue
            obs_history.append(obs_json_file)

            with open(obs_json_file, 'r') as f:
                obs_cfg = json.load(f)

            obs_obj = ObstacleCollisionObj.load(
                obs_cfg['name'], point_radius=obs_cfg['point_radius'], dim=obs_cfg['dim'],
                # file=os.path.join(obs_cfg['obstacle_dir'], f"{obs_cfg['name']}.{obs_cfg['file_format']}")
                file=os.path.join(obs_cfg['obstacle_dir'], obs_cfg['filter_obs'])
            )
            obs_objs.append(obs_obj)
        return obs_objs

    return []


def load_simulator(
        run_cfg: dict, args,
        domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags,
        velocity_order: int = 2, pressure_order: int = 1,
):
    if args.simulate_method == 'navier_stoke':
        condition_module = ImportTool.import_module(run_cfg['proj_dir'], run_cfg['condition_package_name'])
        input_markers = [int(marker) for marker in run_cfg['input_markers'].keys()]
        output_markers = run_cfg['output_markers']
        bry_markers = run_cfg['bry_free_markers'] + run_cfg['bry_fix_markers']

        condition_inflow_dict = {}
        for marker in input_markers:
            marker_fun_name = run_cfg['input_markers'][str(marker)]
            inflow_fun = ImportTool.get_module_function(condition_module, marker_fun_name)
            condition_inflow_dict[marker] = partial(inflow_fun, tdim=run_cfg['dim'])

        simulator = DolfinSimulator(run_cfg['name'], domain, cell_tags, facet_tags, velocity_order, pressure_order)
        simulate_cfg = run_cfg['simulate_cfg']['navier_stoke']
        simulator.define_navier_stoke_equation(nu_value=simulate_cfg['kinematic_viscosity_nu'])

        # ------ define boundary
        for marker in bry_markers:
            simulator.add_boundary(f"bry_u{marker}", value=0.0, marker=marker, is_velocity=True)

        for marker in condition_inflow_dict.keys():
            simulator.add_boundary('inflow_u', value=condition_inflow_dict[marker], marker=marker, is_velocity=True)

        for marker in output_markers:
            simulator.add_boundary(f"outflow_p_{marker}", value=0.0, marker=marker, is_velocity=False)

        if args.load_guess_res:
            simulator.load_result(
                file_info={
                    'u_pkl_file': run_cfg['velocity_init_pkl'], 'p_pkl_file': run_cfg['pressure_init_pkl']
                },
                load_type='pkl'
            )

    elif args.simulate_method == 'openfoam':
        simulator = OpenFoamSimulator(
            run_cfg['name'], domain, cell_tags, facet_tags, run_cfg['simulate_cfg']['openfoam']
        )

    else:
        raise ValueError("[ERROR]: Non-Valid Simulator")

    return simulator


def load_base_model(cfg: dict, args, tag_name=None):
    domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
        file=os.path.join(cfg['proj_dir'], cfg['xdmf_file']),
        mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
    )

    simulator = load_simulator(cfg, args, domain, cell_tags, facet_tags)
    opt_save_dir = os.path.join(cfg['proj_dir'], 'opt_result')
    if os.path.exists(opt_save_dir):
        shutil.rmtree(opt_save_dir)
    os.mkdir(opt_save_dir)

    input_markers = [int(marker) for marker in cfg['input_markers'].keys()]
    output_markers = cfg['output_markers']
    bry_markers = cfg['bry_free_markers'] + cfg['bry_fix_markers']
    bry_fixed_markers = cfg['bry_fix_markers'] + input_markers + output_markers
    bry_free_markers = cfg['bry_free_markers']

    condition_module = ImportTool.import_module(cfg['proj_dir'], cfg['condition_package_name'])
    condition_inflow_dict = {}
    for marker in input_markers:
        marker_str = str(marker)
        marker_fun_name = cfg['input_markers'][marker_str]
        inflow_fun = ImportTool.get_module_function(condition_module, marker_fun_name)
        condition_inflow_dict[marker] = partial(inflow_fun, tdim=cfg['dim'])

    opt_cfg = cfg['optimize_cfg']
    opt = FluidConditionalModel(
        bry_markers=bry_markers,
        deformation_cfg=opt_cfg['deformation_cfg'],
        point_radius=opt_cfg['point_radius'],
        run_strategy_cfg=opt_cfg['run_strategy_cfg'],
        obs_avoid_cfg=opt_cfg['obs_avoid_cfg'],
        simulator=simulator,
        nu_value=opt_cfg['kinematic_viscosity'],
        tag_name=tag_name,
        velocity_order=2, pressure_order=1,
        save_opt_cfg={
            'orig_msh_file': os.path.join(cfg['proj_dir'], cfg['msh_file']), 'save_dir': opt_save_dir
        }
    )

    # --- define boundary conditions
    for marker in bry_markers:
        opt.add_state_boundary(name=f"bry_u{marker}", value=0.0, marker=marker, is_velocity=True)

    for marker in condition_inflow_dict.keys():
        opt.add_state_boundary('inflow_u', value=condition_inflow_dict[marker], marker=marker, is_velocity=True)

    for marker in output_markers:
        opt.add_state_boundary(f"outflow_p_{marker}", value=0.0, marker=marker, is_velocity=False)

    for marker in bry_fixed_markers:
        opt.add_control_boundary(name=f"fix_bry_shape_{marker}", value=0.0, marker=marker)

    opt.state_initiation(
        snes_option=opt_cfg['snes_option'],
        snes_criterion=opt_cfg['snes_criterion'],
        state_ksp_option=opt_cfg['state_ksp_option'],
        adjoint_ksp_option=opt_cfg['adjoint_ksp_option'],
        gradient_ksp_option=opt_cfg['gradient_ksp_option'],
    )

    # --- define cost functions
    cost_functional_list, cost_weights = [], {}
    for cost_cfg in opt_cfg['cost_functions']:
        if cost_cfg['name'] == 'MiniumEnergy':
            cost_functional_list.append(
                IntegralFunction(domain=opt.domain, form=inner(grad(opt.u), grad(opt.u)) * ufl.dx, name='MiniumEnergy')
            )
        else:
            raise ValueError("[ERROR]: Non-Valid Method")
        cost_weights[cost_cfg['name']] = cost_cfg['weight']

    # --- define regularization
    shape_regularization_list = []
    for regularization_cfg in opt_cfg['regularization_functions']:
        if regularization_cfg['name'] == 'VolumeRegularization':
            shape_regularization_list.append(
                VolumeRegularization(
                    opt.control_problem, mu=regularization_cfg['mu'],
                    target_volume_rho=regularization_cfg['target_volume_rho'], method=regularization_cfg['method'],
                )
            )
        else:
            raise ValueError("[ERROR]: Non-Valid Method")
    shape_regularization = ShapeRegularization(shape_regularization_list)

    conflict_regularization = SparsePointsRegularization(
        opt.control_problem, cfg=opt_cfg['obs_avoid_cfg'], mu=opt_cfg['obs_avoid_cfg']['weight']
    )

    scalar_product_method: dict = opt_cfg['scalar_product_method']
    if scalar_product_method['method'] == 'Poincare-Steklov operator':
        scalar_product_method.update({
            'cell_tags': cell_tags,
            'facet_tags': facet_tags,
            'bry_free_markers': bry_free_markers,
            'bry_fixed_markers': bry_fixed_markers,
        })

    opt.optimization_initiation(
        cost_functional_list=cost_functional_list,
        cost_weight=cost_weights,
        shapeRegularization=shape_regularization,
        scalar_product_method=scalar_product_method,
        conflict_regularization=conflict_regularization
    )

    pressure_dict, norm_flow_dict = {}, {}
    for marker in input_markers:
        area_value = AssembleUtils.assemble_scalar(
            dolfinx.fem.form(dolfinx.fem.Constant(opt.domain, 1.0) * opt.ds(marker))
        )
        pressure_dict[f"pressure_{marker}_{opt.name}"] = dolfinx.fem.form((1.0 / area_value) * opt.p * opt.ds(marker))

    for marker in output_markers:
        norm_flow_dict[f"normFlow_{marker}_{opt.name}"] = dolfinx.fem.form(dot(opt.u, opt.n_vec) * opt.ds(marker))

    logger_dicts = {
        'norm_flow': norm_flow_dict, 'pressure': pressure_dict,
        'volume': {f"volume_{opt.name}": dolfinx.fem.form(dolfinx.fem.Constant(opt.domain, 1.0) * ufl.dx)}
    }

    record_dir = os.path.join(cfg['proj_dir'], f"{cfg['name']}_record")
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)

    log_dict = opt.init_optimize_state(
        record_dir=record_dir,
        logger_dicts=logger_dicts,
        with_debug=args.with_debug
    )

    return opt, log_dict


def main():
    args = parse_args()
    assert args.res_dir is not None

    models: List[Union[FluidConditionalModel, FluidShapeRecombineLayer]] = []
    log_list = []
    check_names = []
    for run_cfg_file in args.json_files:
        with open(run_cfg_file, 'r') as f:
            run_cfg: dict = json.load(f)

        if not run_cfg.get('recombine_cfgs', False):
            model, log_dict = load_base_model(run_cfg, args, tag_name=None)
            log_list.append(log_dict)

        else:
            model = FluidShapeRecombineLayer(tag_name=run_cfg['tag_name'])
            for sub_cfg_name in run_cfg['recombine_cfgs']:
                with open(os.path.join(run_cfg['proj_dir'], sub_cfg_name), 'r') as f:
                    run_cfg = json.load(f)

                sub_model, log_dict = load_base_model(run_cfg, args, tag_name=model.tag_name)
                model.add_condition_opt(sub_model)
                log_list.append(log_dict)

        if model.name in check_names:
            raise ValueError("[ERROR]: Duplicate Config Name")
        else:
            check_names.append(model.name)
        models.append(model)

    mesh_objs = []
    for model in models:
        if isinstance(model, FluidConditionalModel):
            mesh_objs.append(model.mesh_obj)
        elif isinstance(model, FluidShapeRecombineLayer):
            for name in model.opt_dicts.keys():
                mesh_objs.append(model.opt_dicts[name].mesh_obj)
    obs_objs = load_obstacle(args)

    tensor_board_dir = os.path.join(args.res_dir, 'log')
    if os.path.exists(tensor_board_dir):
        shutil.rmtree(tensor_board_dir)
    os.mkdir(tensor_board_dir)
    log_recorder = TensorBoardRecorder(tensor_board_dir)
    FluidConditionalModel.write_log_tensorboard(log_recorder, log_list, 0)

    step = 0
    best_loss_list: np.ndarray = None
    best_loss_tol = 0.05
    while True:
        step += 1

        loss_list, log_list, is_converge = [], [], True
        res_dict = {}
        for model in models:
            grad_res_dict = model.compute_iteration_deformation(
                obs_objs=obs_objs, mesh_objs=mesh_objs, step=step, diffusion_method='diffusion_loss_weight',
                with_debug=args.with_debug
            )
            if not grad_res_dict['state']:
                print(f"[INFO]: {model.tag_name} Grad Computation Fail")
                return -1

            res_dict[model.name] = grad_res_dict

        for model in models:
            grad_res_dict = res_dict[model.name]
            move_res_dict = model.move_mesh(grad_res_dict['diffusion'])
            if not move_res_dict['state']:
                print(f"[INFO]: {model.tag_name} Move Mesh Fail")
                return -1

            loss, log_dict = model.update_optimize_state(with_log_info=True, step=step)  # todo 这里加优先级
            is_converge = is_converge and move_res_dict['is_converge']

            if isinstance(loss, List):
                loss_list.extend(loss)
                log_list.extend(log_dict)
            else:
                loss_list.append(loss)
                log_list.append(log_dict)

        FluidConditionalModel.write_log_tensorboard(log_recorder, log_list, step)
        loss_list = np.array(loss_list)
        loss_iter = np.sum(loss_list)

        if best_loss_list is None:
            best_loss_list = loss_list
        else:
            best_loss_list = np.minimum(best_loss_list, loss_list)

        print(f"[###Info {step}] loss:{loss_iter:.8f}")

        if is_converge:
            break

        if np.any(loss_list > best_loss_list * (1.0 + best_loss_tol)):
            break

        if step > 100:
            break

        for model in models:
            model.save_msh(args.with_debug)


if __name__ == '__main__':
    main()
