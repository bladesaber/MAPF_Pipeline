import os
import shutil
import numpy as np
import dolfinx
import ufl
from ufl import grad, dot, inner, div
from functools import partial
import argparse
import json
import pyvista

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.fluid_tools.fluid_shapeOpt_obstacle import FluidShapeRecombineModel
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization, \
    VolumeRegularization
from scripts_py.version_9.dolfinx_Grad.recorder_utils import TensorBoardRecorder
from scripts_py.version_9.dolfinx_Grad.collision_objs import ObstacleCollisionObj
from scripts_py.version_9.dolfinx_Grad.surface_fields import SparsePointsRegularization
from scripts_py.version_9.dolfinx_Grad.user_book.step1_project_tool import ImportTool
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder


def parse_args():
    parser = argparse.ArgumentParser(description="Fluid Simulation Tool")
    parser.add_argument('--proj_dir', type=str, default=None)
    parser.add_argument('--json_names', type=str, nargs='+', default=[])
    parser.add_argument('--init_mesh', type=int, default=0)
    parser.add_argument('--with_debug', type=int, default=0)
    parser.add_argument('--load_guess_res', type=int, default=0)
    parser.add_argument('--with_obs', type=int, default=1)
    args = parser.parse_args()
    return args


args = parse_args()
assert len(args.json_names) > 0 and (args.proj_dir is not None)

opt_dicts = {}
for cfg_name in args.json_names:
    with open(os.path.join(args.proj_dir, cfg_name), 'r') as f:
        cfg = json.load(f)
        opt_dicts[cfg['name']] = {'cfg': cfg}

obs_objs = []
obs_objs_src_history = []
log_list = []

init_mesh = args.init_mesh
for opt_name in opt_dicts.keys():
    opt_dict = opt_dicts[opt_name]
    cfg = opt_dict['cfg']

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

    optimize_cfg = cfg['optimize_cfg']
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

    if args.with_obs:
        for obs_name in cfg['obstacle_names']:
            obs_file = os.path.join(cfg['obstacle_dir'], f"{obs_name}.json")
            if obs_file in obs_objs_src_history:
                continue

            obs_objs_src_history.append(obs_file)
            with open(obs_file, 'r') as f:
                obs_json = json.load(f)

            obs_obj = ObstacleCollisionObj.load(
                obs_json['name'], point_radius=obs_json['point_radius'], dim=obs_json['dim'],
                file=os.path.join(cfg['obstacle_dir'], f"{obs_json['name']}.{obs_json['file_format']}")
            )
            obs_objs.append(obs_obj)

    # ------------------------------------------------------------------------------------------------
    opt_strategy_cfg: dict = optimize_cfg['conflict_cfg']
    opt_strategy_cfg.update({
        'max_step_limit': optimize_cfg['max_step_limit'],
        'opt_tol_rho': optimize_cfg['opt_tol_rho'],
        'init_stepSize': optimize_cfg['init_stepSize'],
        'stepSize_lower': optimize_cfg['stepSize_lower'],
    })

    opt = FluidShapeRecombineModel(
        name=opt_name, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags,
        Re=optimize_cfg['Re'], bry_markers=bry_markers,
        isStokeEqu=optimize_cfg['isStokeEqu'],
        deformation_cfg=optimize_cfg['deformation_cfg'],
        point_radius=optimize_cfg['point_radius'],
        opt_strategy_cfg=opt_strategy_cfg
    )
    if args.load_guess_res:
        opt.load_initiation_pickle(
            u_pickle_file=os.path.join(cfg['proj_dir'], optimize_cfg['u_initation_pickle']),
            p_pickle_file=os.path.join(cfg['proj_dir'], optimize_cfg['p_initation_pickle']),
        )

    for marker in bry_markers:
        bc_value = dolfinx.fem.Function(opt.V, name=f"bry_u{marker}")
        opt.add_state_boundary(bc_value, marker, is_velocity=True)

    for marker in condition_inflow_dict.keys():
        inflow_value = dolfinx.fem.Function(opt.V, name='inflow_u')
        inflow_value.interpolate(condition_inflow_dict[marker])
        opt.add_state_boundary(value=inflow_value, marker=marker, is_velocity=True)

    for marker in output_markers:
        bc_out_value = dolfinx.fem.Function(opt.Q, name=f"outflow_p_{marker}")
        opt.add_state_boundary(bc_out_value, marker, is_velocity=False)

    for marker in bry_fixed_markers:
        bc_value = dolfinx.fem.Function(opt.V_S, name=f"fix_bry_shape_{marker}")
        opt.add_control_boundary(bc_value, marker)

    opt.state_initiation(
        state_ksp_option=optimize_cfg['state_ksp_option'],
        adjoint_ksp_option=optimize_cfg['adjoint_ksp_option'],
        gradient_ksp_option=optimize_cfg['gradient_ksp_option'],
    )

    cost_functional_list = [
        IntegralFunction(
            domain=opt.domain,
            form=inner(grad(opt.u), grad(opt.u)) * ufl.dx,
            name=f"minium_energy"
        )
    ]
    cost_weight = {
        'minium_energy': 1.0
    }

    shape_regularization = ShapeRegularization([
        VolumeRegularization(
            opt.control_problem,
            mu=0.5,
            target_volume_rho=0.6,
            method='percentage_div'
        )
    ])
    conflict_regularization = SparsePointsRegularization(
        opt.control_problem, mu=5.0,
        opt_strategy_cfg=opt.opt_strategy_cfg
    )

    scalar_product_method: dict = optimize_cfg['scalar_product_method']
    if scalar_product_method['method'] == 'Poincare-Steklov operator':
        scalar_product_method.update({
            'cell_tags': cell_tags,
            'facet_tags': facet_tags,
            'bry_free_markers': bry_free_markers,
            'bry_fixed_markers': bry_fixed_markers,
        })

    opt.optimization_initiation(
        cost_functional_list=cost_functional_list,
        cost_weight=cost_weight,
        shapeRegularization=shape_regularization,
        scalar_product_method=scalar_product_method,
        conflict_regularization=conflict_regularization
    )

    inflow_dict = {}
    for marker in input_markers:
        inflow_dict[f"inflow_{marker}_p"] = dolfinx.fem.form(opt.p * opt.ds(marker))
    outflow_dict = {}
    for marker in output_markers:
        outflow_dict[f"outflow_{marker}_v"] = dolfinx.fem.form(dot(opt.u, opt.n_vec) * opt.ds(marker))
    logger_dicts = {'inflow': inflow_dict, 'outflow': outflow_dict}

    record_dir = os.path.join(cfg['proj_dir'], f"{cfg['name']}_record")
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)

    tensorBoard_dir = os.path.join(record_dir, 'log')
    if os.path.exists(tensorBoard_dir):
        shutil.rmtree(tensorBoard_dir)
    os.mkdir(tensorBoard_dir)
    log_recorder = TensorBoardRecorder(tensorBoard_dir)

    log_dict = opt.init_solve_cfg(
        record_dir=record_dir,
        logger_dicts=logger_dicts,
        with_debug=False
    )

    opt_dict.update({'opt': opt})
    log_list.append(log_dict)

tensorBoard_dir = os.path.join(args.proj_dir, 'log')
if os.path.exists(tensorBoard_dir):
    shutil.rmtree(tensorBoard_dir)
os.mkdir(tensorBoard_dir)
log_recorder = TensorBoardRecorder(tensorBoard_dir)

FluidShapeRecombineModel.log_dict(log_recorder, log_list, 0)


def run_loop():
    step = 0
    while True:
        step += 1

        for name in opt_dicts.keys():
            opt_dict = opt_dicts[name]
            opt: FluidShapeRecombineModel = opt_dict['opt']
            grad_res_dict = opt.single_solve(obs_objs=obs_objs, mesh_objs=[], step=step, with_debug=args.with_debug)

            if not grad_res_dict['state']:
                return {'state': False, 'info': '[INFO]: Grad Computation Fail'}

            opt_dict.update({
                'displacement': grad_res_dict['displacement'],
                'loss': opt.solve_mapper['loss_storge'].value
            })

        diffusion = FluidShapeRecombineModel.merge_diffusion(opt_dicts, method='diffusion_loss_weight')

        log_list, is_converge, loss_list = [], True, []
        for name in opt_dicts.keys():
            opt: FluidShapeRecombineModel = opt_dicts[name]['opt']
            move_res_dict = opt.move_mesh(diffusion)

            if not move_res_dict['state']:
                return {'state': False, 'info': '[INFO]: Move Mesh Fail'}

            is_converge = is_converge and move_res_dict['is_converge']
            loss, log_dict = opt.update_opt_info(with_log_info=True, step=step)
            log_list.append(log_dict)
            loss_list.append(loss)

        FluidShapeRecombineModel.log_dict(log_recorder, log_list, step)
        print(f"[###Info {step}] loss:{np.sum(loss_list):.8f}")

        if is_converge:
            return {'state': False, 'info': '[INFO]: Converge Success'}

        if step > 100:
            return {'state': False, 'info': '[INFO]: Out of Max Iteration'}


res_dict = run_loop()
print(res_dict['info'])
