import os
import shutil
import numpy as np
import pandas as pd
import open3d as o3d
import json
from typing import Dict
import argparse
import h5py

from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from flask_cors import *

from scripts_py.version_9.mapf_pkg.networker_block import SimpleBlocker
from scripts_py.version_9.mapf_pkg.algo_utils import create_obs_pcd, discrete_pipe_position
from scripts_py.version_9.mapf_pkg.cbs_utils import CbsSolver
from scripts_py.version_9.mapf_pkg.smooth_optimizer import PathOptimizer
from scripts_py.version_9.mapf_pkg.shape_utils import ShapeUtils

app = Flask(__name__)


class ServerManager(object):
    def __init__(self, server_dir):
        self.server_dir = server_dir
        self.path_optimizers_dict = {}

    def init_proj_dir(self, proj_id: str):
        proj_dir = os.path.join(self.server_dir, proj_id)
        if not os.path.exists(proj_dir):
            os.mkdir(proj_dir)
            os.mkdir(os.path.join(proj_dir, 'obstacle'))
            os.mkdir(os.path.join(proj_dir, 'path_result'))
            os.mkdir(os.path.join(proj_dir, 'smooth_result'))
            with open(os.path.join(proj_dir, 'env.json'), 'w') as f:
                json.dump(
                    {
                        'obs_cfgs': {},
                        'pipe_cfgs': {}
                    },
                    f, indent=4
                )

    def get_proj_info(self, proj_id: str):
        proj_dir = os.path.join(self.server_dir, proj_id)
        return {
            'status': os.path.exists(proj_dir),
            'proj_dir': proj_dir,
            'obstacle_dir': os.path.join(proj_dir, 'obstacle'),
            'env_json': os.path.join(proj_dir, 'env.json'),
            'path_result_dir': os.path.join(proj_dir, 'path_result'),
            'smooth_result_dir': os.path.join(proj_dir, 'smooth_result'),
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Flask Server')
    parser.add_argument(
        '--server_dir', type=str, default='/home/admin123456/Desktop/work/flask_server'
    )
    args = parser.parse_args()
    return args


@app.route('/', methods=['GET'])
def GetTest():
    return 'Hello World By Q.'


@app.route('/PostTest', methods=['POST'])
@cross_origin()
def PostTest():
    params: Dict = request.form.to_dict()
    # return send_from_directory(
    #     '/home/admin123456/Desktop/work/path_examples',
    #     's3_01/static_obstacle.csv'
    # )
    return 'Hello World By Q.'


@app.route('/file_system', methods=['POST'])
@cross_origin()
def file_system():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    proj_dir = os.path.join(server_manager.server_dir, proj_id)
    cfg = json.loads(params['cfg'])

    if cfg['operation'] == 'mkdir':
        new_dir = os.path.join(proj_dir, cfg['path'])
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        return jsonify({'success': True})

    elif cfg['operation'] == 'rmtree':
        cur_dir = os.path.join(proj_dir, cfg['path'])
        if os.path.exists(cur_dir) and os.path.isdir(cur_dir):
            shutil.rmtree(cur_dir)
        return jsonify({'success': True})

    elif cfg['operation'] == 'rm':
        cur_file = os.path.join(proj_dir, cfg['path'])
        if os.path.exists(cur_file) and os.path.isfile(cur_file):
            os.remove(cur_file)
        return jsonify({'success': True})

    elif cfg['operation'] == 'fetch':
        if os.path.exists(proj_dir):
            file = os.path.join(proj_dir, cfg['path'])
            if os.path.exists(file) and os.path.isfile(file):
                return send_file(file, as_attachment=True)
        return jsonify({'success': False, 'info': "project isn't exist."})

    elif cfg['operation'] == 'write':
        if os.path.exists(proj_dir):
            receive_files = request.files
            data = receive_files.get('file')
            data.save(os.path.join(proj_dir, cfg['path']))
            return jsonify({'success': True})
        return jsonify({'success': False, 'info': "project isn't exist."})

    elif cfg['operation'] == 'init_project':
        server_manager.init_proj_dir(proj_id)

    return jsonify({'success': False, 'info': "non valid operation."})


@app.route('/env_update', methods=['POST'])
@cross_origin()
def env_update():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    cfg: Dict = json.loads(params['cfg'])

    proj_cfg = server_manager.get_proj_info(proj_id)
    if not proj_cfg['status']:
        return jsonify({'success': False, 'info': "project isn't exist."})

    with open(proj_cfg['env_json'], 'r') as f:
        env_json = json.load(f)

    obs_cfgs = cfg['obstacle_cfgs']
    for obs_name, obs_cfg in obs_cfgs.items():
        obs_data = request.files.get(f"{obs_name}")
        stl_file = os.path.join(proj_cfg['obstacle_dir'], f"{obs_name}.stl")

        # filename = obs_data.filename
        # obs_format = filename.split('.')[-1]
        # if obs_format not in ['stl', 'stp', 'step', 'STL', 'STP', 'STEP']:
        #     return jsonify({'success': False, 'info': 'non valid file format.'})
        # if obs_format in ['stp', 'step', 'STP', 'STEP']:
        #     with TemporaryDirectory() as tmp_dir:
        #         save_tmp_file = os.path.join(tmp_dir, filename)
        #         obs_data.save(save_tmp_file)
        #         solid = cq.importers.importStep(save_tmp_file)
        #         cq.exporters.export(solid, stl_file)
        # else:
        #     obs_data.save(stl_file)
        obs_data.save(stl_file)

        mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(stl_file)
        pcd_o3d: o3d.geometry.PointCloud = mesh.sample_points_poisson_disk(obs_cfg['sample_num'])
        pcd_np = np.asarray(pcd_o3d.points)

        df = pd.DataFrame(pcd_np, columns=['x', 'y', 'z'])
        df['radius'] = 0.0
        obs_csv = os.path.join(proj_cfg['obstacle_dir'], f"{obs_name}.csv")
        df.to_csv(obs_csv)

        obs_cfg.update({
            'csv_file': f"obstacle/{obs_name}.csv",
            'stl_file': f"obstacle/{obs_name}.stl",
        })
        env_json['obs_cfgs'].update({obs_name: obs_cfg})

    contain_names = list(obs_cfgs.keys())
    for name in list(env_json['obs_cfgs'].keys()):
        if name not in contain_names:
            obs_cfg = env_json['obs_cfgs'][name]
            os.remove(os.path.join(proj_cfg['proj_dir'], obs_cfg['csv_file']))
            os.remove(os.path.join(proj_cfg['proj_dir'], obs_cfg['stl_file']))
            del env_json['obs_cfgs'][name]

    # pipe_cfgs: {position, direction, group_idx, radius, is_input}
    env_json['pipe_cfgs'] = cfg['pipe_cfgs']

    with open(proj_cfg['env_json'], 'w') as f:
        json.dump(env_json, f, indent=4)

    return jsonify({'success': True, 'json': env_json})


# @app.route('/create_search_network', methods=['POST'])
# def create_search_network():
#     """
#     cfg: {
#         pipes:{
#             pipe_name: {radius, position, group_idx}
#         },
#         block_sequence: [{block_id, groups:[{group_idx, names}]}]
#     }
#     """
#     params: Dict = request.form.to_dict()
#     # proj_id = params['proj_id']
#     cfg = json.loads(params['cfg'])
#
#     pipes_cfg = cfg['pipes']
#     blocks_seq = cfg['block_sequence']
#     SimpleBlocker.create_block_network(pipes_cfg, blocks_seq)
#
#     return jsonify({'Status': True, 'Info': '', 'pipes': pipes_cfg, 'block_sequence': blocks_seq})


@app.route('/search_setup', methods=['POST'])
@cross_origin()
def search_setup():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    proj_cfg = server_manager.get_proj_info(proj_id)
    cfg = json.loads(params['cfg'])

    with open(proj_cfg['env_json'], 'r') as f:
        env_json = json.load(f)

    obs_cfg = {}
    for obs_name in cfg['obstacle_names']:
        if obs_name in env_json['obs_cfgs'].keys():
            csv_file = os.path.join(proj_cfg['proj_dir'], env_json['obs_cfgs'][obs_name]['csv_file'])
            obs_df = pd.read_csv(csv_file, index_col=0)
            obs_cfg[obs_name] = {'pcd': obs_df[['x', 'y', 'z']].values}

    setup_name = cfg['search_name']
    setup_dir = os.path.join(proj_cfg['path_result_dir'], setup_name)
    if not os.path.exists(setup_dir):
        os.mkdir(setup_dir)

    setup_json = cfg['setup_json']

    # ------ step 1 make pipe position discrete
    grid_env = setup_json['grid_env']
    grid_min_np = np.array(grid_env['grid_min'])
    grid_max_np = np.array(grid_env['grid_max'])
    size_of_xyz = np.array([grid_env['size_of_x'], grid_env['size_of_y'], grid_env['size_of_z']])
    length_of_xyz = (grid_max_np - grid_min_np) / size_of_xyz
    pipes_cfg = setup_json['pipes']
    pipes_cfg = discrete_pipe_position(pipes_cfg, grid_min=grid_min_np, length_of_xyz=length_of_xyz)

    # ------ step 2 create obstacle
    obs_pcd_np = create_obs_pcd(pipes_cfg, obs_cfg)
    obs_df = pd.DataFrame(obs_pcd_np, columns=['x', 'y', 'z'])
    obs_df['radius'] = 0.0

    obstacle_save_file = os.path.join(setup_dir, 'static_obstacle.csv')
    obs_df.to_csv(obstacle_save_file)
    grid_env['obstacle_tag'] = 'grid_obstacle'
    grid_env['obstacle_file'] = obstacle_save_file
    grid_env['x_grid_length'] = length_of_xyz[0]
    grid_env['y_grid_length'] = length_of_xyz[1]
    grid_env['z_grid_length'] = length_of_xyz[2]

    # ------ step 3 create dynamic constraint
    constraint_df = pd.DataFrame(columns=['x', 'y', 'z', 'radius', 'group_idx'])
    save_file = os.path.join(setup_dir, 'dynamic_constraints.csv')
    grid_env['dynamic_constrain_file'] = save_file
    constraint_df.to_csv(save_file)

    with open(os.path.join(setup_dir, 'algorithm_setup.json'), 'w') as f:
        json.dump(setup_json, f, indent=4)

    return jsonify({'success': True, 'setup_json': setup_json})


@app.route('/search_run', methods=['POST'])
@cross_origin()
def search_run():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    proj_cfg = server_manager.get_proj_info(proj_id)
    run_cfg = json.loads(params['cfg'])
    setup_name = run_cfg['search_name']
    setup_dir = os.path.join(proj_cfg['path_result_dir'], setup_name)

    with open(os.path.join(setup_dir, 'algorithm_setup.json'), 'r') as f:
        setup_json = json.load(f)

    grid_cfg: Dict = setup_json['grid_env']
    pipe_cfg = setup_json['pipes']

    aux_constrain_df = None
    # if grid_cfg.get('dynamic_constrain_file', False):
    #     aux_constrain_df = pd.read_csv(grid_cfg['dynamic_constrain_file'], index_col=0)

    block_res_list = []
    last_leaf_info = {}
    success_flag = True

    for block_info in setup_json['search_tree']['block_sequence']:
        solver = CbsSolver(grid_cfg, pipe_cfg)
        root = solver.init_block_root(
            block_info['groups'], last_leafs_info=last_leaf_info, aux_constrain_df=aux_constrain_df
        )
        group_idxs = list(solver.task_infos.keys())

        is_success, res_node = solver.solve_block(
            root, group_idxs,
            max_iter=int(run_cfg['max_iter']),
            max_node_limit=int(run_cfg['max_node'])
        )
        success_flag = success_flag & is_success

        if is_success:
            last_leaf_info = solver.convert_node_to_leaf_info(res_node, group_idxs, last_leaf_info)
            block_res_list.append(solver.save_path(res_node, group_idxs, solver.task_infos))

        else:
            return jsonify({'success': False, 'info': f"[INFO]: False at solving {block_info['block_id']}"})

    if success_flag:
        save_file = os.path.join(setup_dir, 'search_result.hdf5')
        with h5py.File(save_file, 'w') as f:
            h5df_group = {}
            for block_res in block_res_list:
                for group_idx, group_res in block_res.items():
                    if group_idx not in h5df_group.keys():
                        group_set = f.create_group(str(group_idx))
                        group_set.create_dataset("group_idx", data=[group_idx])
                        h5df_group[group_idx] = group_set.create_group("path_result")
                    for task_name, xyzr in group_res.items():
                        print(f"[Info]: Saving {task_name} ......")
                        h5df_group[group_idx].create_dataset(task_name, data=xyzr)

        search_df = pd.DataFrame(columns=['x', 'y', 'z', 'radius', 'group_idx', 'task_name'])
        search_data = h5py.File(save_file)
        for group_idx in search_data.keys():
            for task_name, task_xyzr in search_data[group_idx]['path_result'].items():
                task_df = pd.DataFrame(np.array(task_xyzr), columns=['x', 'y', 'z', 'radius'])
                task_df['group_idx'] = group_idx
                task_df['task_name'] = task_name
                search_df = pd.concat([search_df, task_df], axis=0, ignore_index=True)
        search_df.to_csv(os.path.join(setup_dir, 'search_result.csv'))

        return jsonify({'success': True, 'result_file': f'path_result/{setup_name}/search_result.csv'})
    else:
        return jsonify({'success': False, 'info': 'Search Fail'})


@app.route('/configure_smooth_env', methods=['POST'])
@cross_origin()
def configure_smooth_env():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    proj_cfg = server_manager.get_proj_info(proj_id)
    run_cfg = json.loads(params['cfg'])

    search_name = run_cfg['search_name']
    search_dir = os.path.join(proj_cfg['path_result_dir'], search_name)
    smooth_name = run_cfg['smooth_name']
    smooth_dir = os.path.join(proj_cfg['smooth_result_dir'], smooth_name)

    if not os.path.exists(smooth_dir):
        os.mkdir(smooth_dir)

    with open(os.path.join(search_dir, 'algorithm_setup.json'), 'r') as f:
        algo_json = json.load(f)

    pipe_cfg = algo_json['pipes']
    grid_cfg = algo_json['grid_env']
    for pipe_name in pipe_cfg.keys():
        pipe_cfg[pipe_name].update({'radius_scale': 1.0})

    search_file = os.path.join(search_dir, 'search_result.hdf5')
    smooth_json = {
        'cfg': {
            'path_result_hdf5': search_file,
            'grid_cfg': grid_cfg,
            'pipe_cfg': pipe_cfg,
        },
        'path_list': [
            {
                'name': None,
                'begin_name': None,
                'end_name': None
            }
        ],
        'conflict_setting': {
            'obstacle_conflict_weight': 1000.0,
            'path_conflict_weight': 100.0,
            'self_conflict_weight': 100.0
        },
        'setting': []
    }

    for _, res_cell in h5py.File(search_file).items():
        group_idx = res_cell['group_idx'][0]
        group_setting = {
            'group_idx': int(group_idx),
            'graph_cfg': {
                'segment_degree_min': 5,
                'segment_degree_max': None,
            },
            'segments': {},
            'paths': {}
        }
        smooth_json['setting'].append(group_setting)

    return jsonify({'success': True, 'smooth_json': smooth_json})


@app.route('/configure_smooth_setting', methods=['POST'])
@cross_origin()
def configure_smooth_setting():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    proj_cfg = server_manager.get_proj_info(proj_id)
    run_cfg = json.loads(params['cfg'])

    smooth_name = run_cfg['smooth_name']
    smooth_dir = os.path.join(proj_cfg['smooth_result_dir'], smooth_name)

    smooth_json = run_cfg['smooth_json']
    grid_cfg = smooth_json['cfg']['grid_cfg']
    pipe_cfg = smooth_json['cfg']['pipe_cfg']

    # ------ step 1 generate obstacle
    obs_xyz = pd.read_csv(grid_cfg['obstacle_file'], index_col=0)[['x', 'y', 'z', 'radius']].values[:, :3]
    for pipe_name in pipe_cfg.keys():
        pipe_info = pipe_cfg[pipe_name]
        obs_xyz = ShapeUtils.remove_obstacle_point_in_pipe(
            obs_pcd=obs_xyz,
            center=pipe_info['discrete_position'],
            direction=pipe_info['direction'],
            radius=pipe_info['radius'] * pipe_info['radius_scale'],
            tol_scale=1.05,
            is_input=pipe_info['is_input']
        )
    obs_df = pd.DataFrame(obs_xyz, columns=['x', 'y', 'z'])
    obs_df['radius'] = 0.0
    smooth_obstacle_file = os.path.join(smooth_dir, 'smooth_obstacle.csv')
    obs_df.to_csv(smooth_obstacle_file)
    grid_cfg.update({'smooth_obstacle_file': smooth_obstacle_file})

    # ------ step 2 setup optimizer
    obs_xyzr = pd.read_csv(grid_cfg['smooth_obstacle_file'], index_col=0)[['x', 'y', 'z', 'radius']].values
    res_data, init_seg_cfg = h5py.File(smooth_json['cfg']['path_result_hdf5']), {}
    for group_setting in smooth_json['setting']:
        group_idx = group_setting['group_idx']
        init_seg_cfg[group_idx] = {
            'path_dict': {},
            'graph_cfg': group_setting['graph_cfg']
        }
        for name, xyzr in res_data[str(group_idx)]['path_result'].items():
            init_seg_cfg[group_idx]['path_dict'][name] = np.array(xyzr)

    # ------ Step 2: add path
    opt = PathOptimizer(init_seg_cfg, obs_xyzr, smooth_json['conflict_setting'], grid_cfg)

    for path_info in smooth_json['path_list']:
        begin_name, end_name = path_info['begin_name'], path_info['end_name']
        group_idx = pipe_cfg[begin_name]['group_idx']
        opt.network_cells[group_idx].add_path(
            name=path_info['name'],
            begin_xyz=np.array(pipe_cfg[begin_name]['discrete_position']),
            begin_vec=np.array(pipe_cfg[begin_name]['direction']),
            end_xyz=np.array(pipe_cfg[end_name]['discrete_position']),
            end_vec=np.array(pipe_cfg[end_name]['direction'])
        )

    # ------ Just for human edit
    for setting in smooth_json['setting']:
        group_idx = setting['group_idx']

        for seg_idx, seg_cell in opt.network_cells[group_idx].segments.items():
            if seg_idx in setting['segments'].keys():
                continue
            setting['segments'][seg_idx] = {
                'bspline_degree': 3,
                'bspline_density': None,
                'update_bspline_length_threshold': 0.1,
                'spring_radius_conflict_ratio': 0.01,
                'min_conflict_thickness': 0.05,
                'radius_scale': 1.0,
                'color': np.random.uniform(0.0, 1.0, size=(3,)).tolist(),
                'costs': [
                    {
                        'method': 'control_spring_length',
                        'type': 'square_sum',
                        'weight': 1.0
                    },
                    {
                        "method": "control_distribute_constraint",
                        "type": "variance",
                        "weight": 5.0
                    },
                    {
                        'method': 'curvature',
                        'type': 'upper_value',
                        'radius_scale': 3.0,
                        "weight": 5.0
                    },
                    # {
                    #     "method": "control_spring_variance",
                    #     "type": "range_mean",
                    #     "range_value": 0.05,
                    #     "weight": 1.0
                    # }
                ]
            }

        for path_name, path_list in opt.network_cells[group_idx].path_dict.items():
            if path_name in setting['paths'].keys():
                continue
            setting['paths'][path_name] = []
            for _ in path_list:
                setting['paths'][path_name].append([
                    {
                        'method': 'connector_curvature',
                        'type': 'upper_value',
                        'radius_scale': 3.0,
                        'left_num': 3,
                        'right_num': 3,
                        "weight": 5.0
                    }
                ])

    server_manager.path_optimizers_dict[smooth_name] = opt

    return jsonify({'success': True, 'smooth_json': smooth_json})


@app.route('/configure_smooth_optimizer', methods=['POST'])
@cross_origin()
def configure_smooth_optimizer():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    proj_cfg = server_manager.get_proj_info(proj_id)
    run_cfg = json.loads(params['cfg'])

    smooth_name = run_cfg['smooth_name']
    smooth_dir = os.path.join(proj_cfg['smooth_result_dir'], smooth_name)
    smooth_json = run_cfg['smooth_json']

    for setting in smooth_json['setting']:
        group_idx = setting['group_idx']
        server_manager.path_optimizers_dict[smooth_name].init_optimize_cfg(
            group_idx, segment_infos=setting['segments'], path_infos=setting['paths']
        )

    with open(os.path.join(smooth_dir, 'smooth_setup.json'), 'w') as f:
        json.dump(smooth_json, f, indent=4)

    return jsonify({'success': True})


@app.route('/smooth_run', methods=['POST'])
@cross_origin()
def smooth_run():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    proj_cfg = server_manager.get_proj_info(proj_id)
    run_cfg = json.loads(params['cfg'])

    smooth_name = run_cfg['smooth_name']
    smooth_opt = server_manager.path_optimizers_dict[smooth_name]

    # ------ prepare tensor
    smooth_opt.prepare_tensor_and_self_graph()

    # ------ start running
    loss_record, loss_info = smooth_opt.run(max_iter=run_cfg['run_times'], lr=run_cfg['lr'])

    return jsonify({'Status': False, 'Info': '', 'Loss_info': {
        'loss_record': list(loss_record), 'loss_info': loss_info
    }})


# @app.route('/fetch_control_points', methods=['POST'])
# def fetch_control_points():
#     params: Dict = request.form.to_dict()
#     proj_id = params['proj_id']
#     cfg = json.loads(params['cfg'])
#     info = server_manager.get_proj_info(proj_id)
#     run_name = cfg['run_name']
#     run_dir = os.path.join(info['smooth_result_dir'], run_name)
#
#     if f"{proj_id}_{run_name}" not in server_manager.path_optimizers_dict.keys():
#         return jsonify({'Status': False, 'Info': ''})
#
#     opt: PathOptimizer = server_manager.path_optimizers_dict[f"{proj_id}_{run_name}"]
#
#     df_dicts = []
#     for group_idx, net_cell in opt.network_cells.items():
#         for seg_idx, seg_cell in net_cell.segments.items():
#             for node_idx in seg_cell.flags:
#                 node_info = net_cell.node2info[node_idx]
#                 xyzr = net_cell.control_xyzr_np[node_info['pcd_idx']]
#                 df_dicts.append({
#                     'node_idx': node_idx,
#                     'group_idx': group_idx,
#                     'segment_idx': seg_idx,
#                     'pcd_idx': node_info['pcd_idx'],
#                     'fix': node_info['fix'],
#                     'type': node_info['type'],
#                     'x': xyzr[0], 'y': xyzr[1], 'z': xyzr[2], 'radius': xyzr[3]
#                 })
#     df = pd.DataFrame(df_dicts)
#     node_csv = os.path.join(run_dir, 'node_info.csv')
#     df.to_csv(node_csv)
#     return jsonify({'Status': True, 'Info': '', 'file': f"smooth_result/{run_name}/node_info.csv"})
#
#
# @app.route('/update_control_points', methods=['POST'])
# def update_control_points():
#     """
#     cfg: [{node_idx, group_idx, fix}]
#     """
#     params: Dict = request.form.to_dict()
#     proj_id = params['proj_id']
#     cfg = json.loads(params['cfg'])
#     run_name = cfg['run_name']
#
#     if f"{proj_id}_{run_name}" not in server_manager.path_optimizers_dict.keys():
#         return jsonify({'Status': False, 'Info': ''})
#
#     opt: PathOptimizer = server_manager.path_optimizers_dict[f"{proj_id}_{run_name}"]
#
#     change_fix = False
#     for info in cfg:
#         group_idx, node_idx = info['group_idx'], info['node_idx']
#         if info.get('fix', False):
#             opt.network_cells[group_idx].node2info[node_idx].update({'fix': bool(info['fix'])})
#             change_fix = True
#
#         if info.get('xyzr', False):
#             net_cell = opt.network_cells[group_idx]
#             pcd_idx = net_cell.node2info[node_idx]['pcd_idx']
#             net_cell.control_xyzr_np[pcd_idx, :] = np.array(info['xyzr'])
#
#     if change_fix:
#         for group_idx, net_cell in opt.network_cells.items():
#             fix_idxs = []
#             for node_idx, info in net_cell.node2info.items():
#                 if info['fix']:
#                     fix_idxs.append(info['pcd_idx'])
#             net_cell.fix_pcd_idxs = np.array(fix_idxs)
#
#     return jsonify({'Status': True, 'Info': ''})
#
#
# @app.route('/update_smooth_setting', methods=['POST'])
# def update_smooth_setting():
#     params: Dict = request.form.to_dict()
#     proj_id = params['proj_id']
#     cfg = json.loads(params['cfg'])
#     info = server_manager.get_proj_info(proj_id)
#
#     run_name = cfg['run_name']
#     run_dir = os.path.join(info['smooth_result_dir'], run_name)
#
#     if f"{proj_id}_{run_name}" not in server_manager.path_optimizers_dict.keys():
#         return jsonify({'Status': False, 'Info': ''})
#     opt = server_manager.path_optimizers_dict[f"{proj_id}_{run_name}"]
#
#     for setting in cfg['setting']:
#         group_idx = setting['group_idx']
#         for key in list(setting['segments'].keys()):
#             cell = setting['segments'].pop(key)
#             setting['segments'][cell['seg_idx']] = cell
#         opt.network_cells[group_idx].update_optimization_info(
#             segment_infos=setting['segments'],
#             path_infos=setting['paths_cost']
#         )
#
#     with open(os.path.join(run_dir, 'smooth_setup.json'), 'r') as f:
#         smooth_setup_json = json.load(f)
#     smooth_setup_json.update({'setting': cfg['setting']})
#     with open(os.path.join(run_dir, 'smooth_setup.json'), 'w') as f:
#         json.dump(smooth_setup_json, f)
#
#     return jsonify({'Status': True, 'Info': ''})
#
#
# @app.route('/fetch_smooth_result', methods=['POST'])
# def fetch_smooth_result():
#     params: Dict = request.form.to_dict()
#     proj_id = params['proj_id']
#     cfg = json.loads(params['cfg'])
#     info = server_manager.get_proj_info(proj_id)
#     run_name = cfg['run_name']
#     run_dir = os.path.join(info['smooth_result_dir'], run_name)
#
#     run_name = cfg['run_name']
#     if f"{proj_id}_{run_name}" not in server_manager.path_optimizers_dict.keys():
#         return jsonify({'Status': False, 'Info': ''})
#
#     opt = server_manager.path_optimizers_dict[f"{proj_id}_{run_name}"]
#
#     # df_dicts = []
#     # for group_idx, net_cell in opt.network_cells.items():
#     #     for seg_idx, cell in net_cell.segments.items():
#     #         if cfg['with_control_points']:
#     #             control_pcd = net_cell.control_xyzr_np[cell.pcd_idxs, :3]
#     #             for xyz in control_pcd:
#     #                 df_dicts.append({
#     #                     'group_idx': group_idx,
#     #                     'seg_idx': seg_idx,
#     #                     'x': xyz[0], 'y': xyz[1], 'z': xyz[2], 'radius': 0.0,
#     #                     'type': 'control'
#     #                 })
#     #
#     #         if cfg['with_spline_points']:
#     #             spline_xyzr = cell.get_output_bspline_xyzr_np(net_cell.control_xyzr_np)
#     #             for xyzr in spline_xyzr:
#     #                 df_dicts.append({
#     #                     'group_idx': group_idx,
#     #                     'seg_idx': seg_idx,
#     #                     'x': xyzr[0], 'y': xyzr[1], 'z': xyzr[2], 'radius': xyzr[3],
#     #                     'type': 'spline'
#     #                 })
#     # df = pd.DataFrame(df_dicts)
#     # res_csv = os.path.join(run_dir, 'smooth_result.csv')
#     # df.to_csv(res_csv)
#
#     save_file = os.path.join(run_dir, 'smooth_result.hdf5')
#     with h5py.File(save_file, 'w') as f:
#         for group_idx, net_cell in opt.network_cells.items():
#             group_set = f.create_group(str(group_idx))
#             group_set.create_dataset("group_idx", data=[group_idx])
#             control_points_layer = group_set.create_group("control_points")
#             spline_layer = group_set.create_group("spline")
#             path_layer = group_set.create_group("path")
#
#             for seg_idx, seg_cell in net_cell.segments.items():
#                 control_points_layer.create_dataset(
#                     f"seg_{seg_cell.idx}", data=net_cell.control_xyzr_np[seg_cell.pcd_idxs, :]
#                 )
#                 spline_layer.create_dataset(
#                     f"seg_{seg_cell.idx}", data=seg_cell.get_output_bspline_xyzr_np(net_cell.control_xyzr_np)
#                 )
#
#             for name, path_list in net_cell.path_dict.items():
#                 for j, path_cell in enumerate(path_list):
#                     path_layer.create_dataset(
#                         f"{name}_{j}", data=path_cell.get_output_bspline_xyzr_np(
#                             net_cell.control_xyzr_np, with_terminate_vec=True
#                         )
#                     )
#
#     return jsonify({'Status': True, 'Info': '', 'file': f"smooth_result/{run_name}/smooth_result.hdf5"})
#
#
# @app.route('/release_smooth_agent', methods=['POST'])
# def release_smooth_agent():
#     params: Dict = request.form.to_dict()
#     proj_id = params['proj_id']
#     cfg = json.loads(params['cfg'])
#     run_name = cfg['run_name']
#
#     if f"{proj_id}_{run_name}" not in server_manager.path_optimizers_dict.keys():
#         return jsonify({'Status': True, 'Info': ''})
#
#     del server_manager.path_optimizers_dict[f"{proj_id}_{run_name}"]
#     return jsonify({'Status': True, 'Info': ''})


if __name__ == '__main__':
    server_args = parse_args()
    if server_args.server_dir is not None:
        server_manager = ServerManager(server_args.server_dir)
        app.run(host='192.168.1.155', port=9999, debug=True)
