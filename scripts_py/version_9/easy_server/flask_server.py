import os
import shutil
import numpy as np
import pandas as pd
import open3d as o3d
import json
from typing import Dict
import cadquery as cq
import argparse
import h5py
from tempfile import TemporaryDirectory

from flask import Flask, request, jsonify, send_from_directory, Response, send_file

from scripts_py.version_9.mapf_pkg.networker_block import SimpleBlocker
from scripts_py.version_9.mapf_pkg.algo_utils import create_obs_pcd, discrete_pipe_position
from scripts_py.version_9.mapf_pkg.cbs_utils import CbsSolver
from scripts_py.version_9.mapf_pkg.smooth_optimizer import PathOptimizer

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
            'env.json': os.path.join(proj_dir, 'env.json'),
            'path_result_dir': os.path.join(proj_dir, 'path_result'),
            'smooth_result_dir': os.path.join(proj_dir, 'smooth_result'),
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Flask Server')
    parser.add_argument(
        '--server_dir', type=str,
        default='/home/admin123456/Desktop/work/flask_server'
    )
    args = parser.parse_args()
    return args


@app.route('/', methods=['GET'])
def GetTest():
    return 'Hello World By Q.'


@app.route('/PostTest', methods=['POST'])
def PostTest():
    params: Dict = request.form.to_dict()
    # return send_from_directory(
    #     '/home/admin123456/Desktop/work/path_examples',
    #     's3_01/static_obstacle.csv'
    # )
    return 'Hello World By Q.'


@app.route('/file_manage', methods=['POST'])
def file_manage():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    cfg = json.loads(params['cfg'])

    if cfg['operation'] == 'generate':
        server_manager.init_proj_dir(proj_id)
    elif cfg['operation'] == 'delete':
        info = server_manager.get_proj_info(proj_id)
        if info['status']:
            operate_path = os.path.join(info['proj_dir'], cfg['path'])
            if os.path.isdir(operate_path):
                shutil.rmtree(operate_path)
            else:
                os.remove(operate_path)
    elif cfg['operation'] == 'fetch':
        info = server_manager.get_proj_info(proj_id)
        if info['status']:
            file = os.path.join(info['proj_dir'], cfg['path'])
            if os.path.exists(file) and os.path.isfile(file):
                return send_file(file, as_attachment=False)
        return jsonify({'Status': False, 'Info': ''})
    elif cfg['operation'] == 'write':
        info = server_manager.get_proj_info(proj_id)
        if info['status']:
            receive_files = request.files
            receive_files.get(cfg['file']).save(os.path.join(info['proj_dir'], cfg['path']))
            return jsonify({'Status': True, 'Info': ''})
        return jsonify({'Status': False, 'Info': ''})
    else:
        return jsonify({'Status': False, 'Info': ''})
    return jsonify({'Status': True, 'Info': ''})


@app.route('/obstacle_manage', methods=['POST'])
def obstacle_manage():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    cfg = json.loads(params['cfg'])
    info = server_manager.get_proj_info(proj_id)

    if cfg['operation'] == 'update':
        receive_files = request.files
        obs_data = receive_files.get('obs_file')
        obs_filename = obs_data.filename
        obs_name, obs_format = obs_filename.split('.')[-1]
        if obs_format not in ['stl', 'stp', 'step', 'STL', 'STP', 'STEP']:
            return jsonify({'status': False, 'info': 'Error File Format'})

        stl_file = os.path.join(info['obstacle_dir'], f"{obs_name}.stl")
        if obs_format in ['stp', 'step', 'STP', 'STEP']:
            with TemporaryDirectory() as tmp_dir:
                save_tmp_file = os.path.join(tmp_dir, obs_filename)
                obs_data.save(save_tmp_file)
                solid = cq.importers.importStep(save_tmp_file)
                cq.exporters.export(solid, stl_file)
        else:
            obs_data.save(stl_file)

        mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(stl_file)
        pcd_o3d: o3d.geometry.PointCloud = mesh.sample_points_poisson_disk(cfg['sample_num'])
        pcd_np = np.asarray(pcd_o3d.points)

        df = pd.DataFrame(pcd_np, columns=['x', 'y', 'z'])
        # df['radius'] = cfg['radius']
        df['radius'] = 0.0
        obs_csv = os.path.join(info['obstacle_dir'], f"{obs_name}.csv")
        df.to_csv(obs_csv)

        with open(info['env_json'], 'r') as f:
            env_json = json.load(f)
        env_json['obs_cfgs'].update({obs_name: {
            'csv_file': f"obstacle/{obs_name}.csv",
            'stl_file': f"obstacle/{obs_name}.stl",
        }})
        with open(info['env_json'], 'w') as f:
            json.dump(env_json, f, indent=4)

        return jsonify({
            'Status': True, 'Info': '',
            'obs_info': {
                'csv_file': f"obstacle/{obs_name}.csv",
                'stl_file': f"obstacle/{obs_name}.stl",
            }
        })

    elif cfg['operation'] == 'delete':
        obs_file = os.path.join(info['obstacle_dir'], f"{cfg['name']}.csv")
        os.remove(obs_file)
        with open(info['env_json'], 'r') as f:
            env_json = json.load(f)
        del env_json['obs_cfgs'][cfg['name']]
        with open(info['env_json'], 'w') as f:
            json.dump(env_json, f, indent=4)
    else:
        return jsonify({'Status': False, 'Info': ''})


@app.route('/pipe_manage', methods=['POST'])
def pipe_manage():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    cfg = json.loads(params['cfg'])
    info = server_manager.get_proj_info(proj_id)

    with open(info['env_json'], 'r') as f:
        env_json = json.load(f)
    if cfg['operation'] == 'update':
        env_json['pipe_cfgs'].update({
            'position': cfg['position'],
            'direction': cfg['direction'],
            'group_idx': cfg['group_idx'],
            "radius": cfg['radius'],
            "is_input": cfg['is_input'],
        })
    elif cfg['operation'] == 'delete':
        del env_json['pipe_cfgs'][cfg['name']]
    else:
        return jsonify({'Status': False, 'Info': ''})

    with open(info['env_json'], 'w') as f:
        json.dump(env_json, f, indent=4)
    return jsonify({'Status': True, 'Info': ''})


@app.route('/create_search_network', methods=['POST'])
def create_search_network():
    """
    cfg: {
        pipes:{
            pipe_name: {radius, position, group_idx}
        },
        block_sequence: [{block_id, groups:[{group_idx, names}]}]
    }
    """
    params: Dict = request.form.to_dict()
    # proj_id = params['proj_id']
    cfg = json.loads(params['cfg'])

    pipes_cfg = cfg['pipes']
    blocks_seq = cfg['block_sequence']
    SimpleBlocker.create_block_network(pipes_cfg, blocks_seq)

    return jsonify({'Status': True, 'Info': '', 'pipes': pipes_cfg, 'block_sequence': blocks_seq})


@app.route('/search_setup', methods=['POST'])
def search_setup():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    cfg = json.loads(params['cfg'])
    info = server_manager.get_proj_info(proj_id)

    # ------ step 1 init dir
    run_name = cfg['run_name']
    run_dir = os.path.join(info['path_result_dir'], run_name)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    # ------ step 2 make pipe position discrete
    grid_env = cfg['grid_env']
    pipes_cfg = cfg['pipes']
    obs_cfg = {}
    for name in cfg['obstacles']:
        obs_file = os.path.join(info['obstacle_dir'], f"{name}.csv")
        obs_df = pd.read_csv(obs_file, index_col=0)
        obs_cfg[name] = obs_df[['x', 'y', 'z']].values

    grid_min_np = np.array(grid_env['grid_min'])
    grid_max_np = np.array(grid_env['grid_max'])
    size_of_xyz = np.array([grid_env['size_of_x'], grid_env['size_of_y'], grid_env['size_of_z']])
    length_of_xyz = (grid_max_np - grid_min_np) / size_of_xyz
    pipes_cfg = discrete_pipe_position(pipes_cfg, grid_min=grid_min_np, length_of_xyz=length_of_xyz)

    # ------ step 3 create obstacle
    obs_pcd_np = create_obs_pcd(pipes_cfg, obs_cfg)
    obs_df = pd.DataFrame(obs_pcd_np, columns=['x', 'y', 'z'])
    obs_df['radius'] = 0.0
    obstacle_save_file = os.path.join(info['proj_dir'], 'static_obstacle.csv')
    obs_df.to_csv(obstacle_save_file)

    grid_env['obstacle_tag'] = 'grid_obstacle'
    grid_env['obstacle_file'] = obstacle_save_file
    grid_env['x_grid_length'] = length_of_xyz[0]
    grid_env['y_grid_length'] = length_of_xyz[1]
    grid_env['z_grid_length'] = length_of_xyz[2]

    # ------ step 4 create dynamic constraint
    receive_files = request.files
    constraint_data = receive_files.get('dynamic_constraints_file')
    constraint_file = os.path.join(run_dir, 'dynamic_constraints.csv')
    constraint_data.save(constraint_file)
    grid_env['dynamic_constrain_file'] = constraint_file

    # ------ step 5 setup algorithm
    algorithm_setup_json = {
        'grid_env': grid_env,
        'search_tree': {
            'method': 'simple',
            'block_sequence': cfg['block_sequence']
        },
        'pipes': pipes_cfg
    }

    with open(os.path.join(run_dir, 'algorithm_setup.json'), 'w') as f:
        json.dump(algorithm_setup_json, f, indent=4)

    return jsonify({'Status': True, 'Info': '', 'setup_json': algorithm_setup_json})


@app.route('/search_run', methods=['POST'])
def search_run():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    cfg = json.loads(params['cfg'])
    info = server_manager.get_proj_info(proj_id)

    run_name = cfg['run_name']
    run_dir = os.path.join(info['path_result_dir'], run_name)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    setup_file = os.path.join(run_dir, 'algorithm_setup.json')
    with open(setup_file, 'r') as f:
        setup_json = json.load(f)

    grid_cfg: Dict = setup_json['grid_env']
    pipe_cfg = setup_json['pipes']
    aux_constrain_df = pd.read_csv(grid_cfg['dynamic_constrain_file'], index_col=0)

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
            root, group_idxs, max_iter=cfg['max_iter'], max_node_limit=cfg['max_node']
        )
        success_flag = success_flag & is_success

        if is_success:
            last_leaf_info = solver.convert_node_to_leaf_info(res_node, group_idxs, last_leaf_info)
            block_res_list.append(solver.save_path(res_node, group_idxs, solver.task_infos))

        else:
            break

    if success_flag:
        save_file = os.path.join(run_dir, 'search_result.hdf5')
        with h5py.File(save_file, 'w') as f:
            h5df_group = {}
            for block_res in block_res_list:
                for group_idx, group_res in block_res.items():
                    if group_idx not in h5df_group.keys():
                        group_set = f.create_group(str(group_idx))
                        group_set.create_dataset("group_idx", data=[group_idx])
                        h5df_group[group_idx] = group_set.create_group("path_result")
                    for task_name, xyzr in group_res.items():
                        h5df_group[group_idx].create_dataset(task_name, data=xyzr)
        return jsonify({'Status': True, 'Info': '', 'file_name': f'result_dir/{run_name}/search_result.hdf5'})
    else:
        return jsonify({'Status': False, 'Info': 'Search Fail'})


@app.route('/smooth_setup', methods=['POST'])
def smooth_setup():
    """
    cfg:{
        grid_env:{size_of_x, size_of_y, size_of_z, grid_min, x_grid_length, y_grid_length, z_grid_length}
        pipes:{name:{group_idx, discrete_position, direction}}
        conflict_setting:{obstacle_weight, path_conflict_weight},
        path_list:[name, begin_name, end_name],
        setting:[{group_idx, segment_degree, segments}]
    }
    """
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    cfg = json.loads(params['cfg'])
    info = server_manager.get_proj_info(proj_id)

    run_name = cfg['run_name']
    run_dir = os.path.join(info['smooth_result_dir'], run_name)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    receive_files = request.files

    grid_path_data = receive_files.get('grid_path_file')
    grid_path_file = os.path.join(run_dir, grid_path_data.filename)
    grid_path_data.save(grid_path_file)

    path_res = {}
    for _, res_cell in h5py.File(grid_path_file).items():
        group_idx = int(res_cell['group_idx'][0])
        path_res[group_idx] = {}
        for name, xyzr in res_cell['path_result'].items():
            path_res[group_idx][name] = xyzr

    obstacle_data = receive_files.get('obs_file')
    obs_file = os.path.join(run_dir, obstacle_data.filename)
    obstacle_data.save(obs_file)
    obs_xyzr = pd.read_csv(obs_file, index_col=0)[['x', 'y', 'z', 'radius']].values

    grid_cfg = cfg['grid_env']
    pipe_cfg = cfg['pipes']

    opt = PathOptimizer(grid_cfg, path_res, obs_xyzr, cfg['conflict_setting'])

    # ------ Step1: add path
    for path_info in cfg['path_list']:
        begin_name, end_name = path_info['begin_name'], path_info['end_name']
        group_idx = pipe_cfg[begin_name]['group_idx']
        opt.network_cells[group_idx].add_path(
            name=path_info['name'],
            begin_xyz=np.array(pipe_cfg[begin_name]['discrete_position']),
            begin_vec=np.array(pipe_cfg[begin_name]['direction']),
            end_xyz=np.array(pipe_cfg[end_name]['discrete_position']),
            end_vec=np.array(pipe_cfg[end_name]['direction'])
        )

    # ------ Step2: refit graph
    for setting in cfg['segment_degree']:
        opt.network_cells[setting['group_idx']].refit_graph(setting['segment_degree'])

    # ------ Step3: generate default edit
    for setting in cfg['setting']:
        group_idx = setting['group_idx']

        for seg_idx, seg_cell in opt.network_cells[group_idx].segments.items():
            if str(seg_idx) in setting['segments'].keys():
                continue
            setting['segments'][seg_idx] = {
                'bspline_degree': 3,
                'bspline_num': 40,
                'output_bspline_num': 80,
                'seg_idx': seg_idx,
                'costs': [
                    {
                        'method': 'control_square_length_cost',
                        'type': 'auto',
                        'weight': 1.0
                    }
                ],
                'relax_conflict_ratio': 0.01,
                'min_relax_conflict_thickness': 0.05,
                'color': np.random.uniform(0.0, 1.0, size=(3,)).tolist()
            }

        for path_name, path_list in opt.network_cells[group_idx].path_dict.items():
            if path_name in setting['paths_cost'].keys():
                continue
            setting['paths_cost'][path_name] = []
            for _ in path_list:
                setting['paths_cost'][path_name].append([
                    {
                        'method': 'curvature_cost',
                        'radius_scale': 3.0,
                        'radius_weight': 0.1,
                        'cos_threshold': 0.95,
                        'cos_exponent': 1.5,
                        'cos_weight': 1.0
                    },
                    {
                        'method': 'connector_control_cos_cost',
                        'weight': 1.0
                    }
                ])

    smooth_setup_json = {
        'grid_env': grid_cfg,
        'pipes': pipe_cfg,
        'obstacle_file': obs_file,
        'path_list': cfg['path_list'],
        'conflict_setting': cfg['conflict_setting'],
        'setting': cfg['setting']
    }
    with open(os.path.join(run_dir, 'smooth_setup.json'), 'w') as f:
        json.dump(smooth_setup_json, f)

    server_manager.path_optimizers_dict[f"{proj_id}_{run_name}"] = opt

    return jsonify({'Status': True, 'Info': '', 'setup_json': smooth_setup_json})


@app.route('/smooth_run', methods=['POST'])
def smooth_run():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    cfg = json.loads(params['cfg'])
    run_name = cfg['run_name']

    if f"{proj_id}_{run_name}" not in server_manager.path_optimizers_dict.keys():
        return jsonify({'Status': False, 'Info': ''})
    opt = server_manager.path_optimizers_dict[f"{proj_id}_{run_name}"]

    # ------ prepare tensor
    for group_idx, net_cell in opt.network_cells.items():
        net_cell.prepare_tensor()

    # ------ start running
    loss_record, loss_info = opt.run(max_iter=cfg['run_times'], lr=cfg['lr'])

    return jsonify({'Status': False, 'Info': '', 'Loss_info': {
        'loss_record': list(loss_record),
        'loss_info': loss_info
    }})


@app.route('/fetch_control_points', methods=['POST'])
def fetch_control_points():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    cfg = json.loads(params['cfg'])
    info = server_manager.get_proj_info(proj_id)
    run_name = cfg['run_name']
    run_dir = os.path.join(info['smooth_result_dir'], run_name)

    if f"{proj_id}_{run_name}" not in server_manager.path_optimizers_dict.keys():
        return jsonify({'Status': False, 'Info': ''})

    opt: PathOptimizer = server_manager.path_optimizers_dict[f"{proj_id}_{run_name}"]

    df_dicts = []
    for group_idx, net_cell in opt.network_cells.items():
        for seg_idx, seg_cell in net_cell.segments.items():
            for node_idx in seg_cell.flags:
                node_info = net_cell.node2info[node_idx]
                xyzr = net_cell.control_xyzr_np[node_info['pcd_idx']]
                df_dicts.append({
                    'node_idx': node_idx,
                    'group_idx': group_idx,
                    'segment_idx': seg_idx,
                    'pcd_idx': node_info['pcd_idx'],
                    'fix': node_info['fix'],
                    'type': node_info['type'],
                    'x': xyzr[0], 'y': xyzr[1], 'z': xyzr[2], 'radius': xyzr[3]
                })
    df = pd.DataFrame(df_dicts)
    node_csv = os.path.join(run_dir, 'node_info.csv')
    df.to_csv(node_csv)
    return jsonify({'Status': True, 'Info': '', 'file': f"smooth_result/{run_name}/node_info.csv"})


@app.route('/update_control_points', methods=['POST'])
def update_control_points():
    """
    cfg: [{node_idx, group_idx, fix}]
    """
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    cfg = json.loads(params['cfg'])
    run_name = cfg['run_name']

    if f"{proj_id}_{run_name}" not in server_manager.path_optimizers_dict.keys():
        return jsonify({'Status': False, 'Info': ''})

    opt: PathOptimizer = server_manager.path_optimizers_dict[f"{proj_id}_{run_name}"]

    change_fix = False
    for info in cfg:
        group_idx, node_idx = info['group_idx'], info['node_idx']
        if info.get('fix', False):
            opt.network_cells[group_idx].node2info[node_idx].update({'fix': bool(info['fix'])})
            change_fix = True

        if info.get('xyzr', False):
            net_cell = opt.network_cells[group_idx]
            pcd_idx = net_cell.node2info[node_idx]['pcd_idx']
            net_cell.control_xyzr_np[pcd_idx, :] = np.array(info['xyzr'])

    if change_fix:
        for group_idx, net_cell in opt.network_cells.items():
            fix_idxs = []
            for node_idx, info in net_cell.node2info.items():
                if info['fix']:
                    fix_idxs.append(info['pcd_idx'])
            net_cell.fix_pcd_idxs = np.array(fix_idxs)

    return jsonify({'Status': True, 'Info': ''})


@app.route('/update_smooth_setting', methods=['POST'])
def update_smooth_setting():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    cfg = json.loads(params['cfg'])
    info = server_manager.get_proj_info(proj_id)

    run_name = cfg['run_name']
    run_dir = os.path.join(info['smooth_result_dir'], run_name)

    if f"{proj_id}_{run_name}" not in server_manager.path_optimizers_dict.keys():
        return jsonify({'Status': False, 'Info': ''})
    opt = server_manager.path_optimizers_dict[f"{proj_id}_{run_name}"]

    for setting in cfg['setting']:
        group_idx = setting['group_idx']
        for key in list(setting['segments'].keys()):
            cell = setting['segments'].pop(key)
            setting['segments'][cell['seg_idx']] = cell
        opt.network_cells[group_idx].update_optimization_info(
            segment_infos=setting['segments'],
            path_infos=setting['paths_cost']
        )

    with open(os.path.join(run_dir, 'smooth_setup.json'), 'r') as f:
        smooth_setup_json = json.load(f)
    smooth_setup_json.update({'setting': cfg['setting']})
    with open(os.path.join(run_dir, 'smooth_setup.json'), 'w') as f:
        json.dump(smooth_setup_json, f)

    return jsonify({'Status': True, 'Info': ''})


@app.route('/fetch_smooth_result', methods=['POST'])
def fetch_smooth_result():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    cfg = json.loads(params['cfg'])
    info = server_manager.get_proj_info(proj_id)
    run_name = cfg['run_name']
    run_dir = os.path.join(info['smooth_result_dir'], run_name)

    run_name = cfg['run_name']
    if f"{proj_id}_{run_name}" not in server_manager.path_optimizers_dict.keys():
        return jsonify({'Status': False, 'Info': ''})

    opt = server_manager.path_optimizers_dict[f"{proj_id}_{run_name}"]

    # df_dicts = []
    # for group_idx, net_cell in opt.network_cells.items():
    #     for seg_idx, cell in net_cell.segments.items():
    #         if cfg['with_control_points']:
    #             control_pcd = net_cell.control_xyzr_np[cell.pcd_idxs, :3]
    #             for xyz in control_pcd:
    #                 df_dicts.append({
    #                     'group_idx': group_idx,
    #                     'seg_idx': seg_idx,
    #                     'x': xyz[0], 'y': xyz[1], 'z': xyz[2], 'radius': 0.0,
    #                     'type': 'control'
    #                 })
    #
    #         if cfg['with_spline_points']:
    #             spline_xyzr = cell.get_output_bspline_xyzr_np(net_cell.control_xyzr_np)
    #             for xyzr in spline_xyzr:
    #                 df_dicts.append({
    #                     'group_idx': group_idx,
    #                     'seg_idx': seg_idx,
    #                     'x': xyzr[0], 'y': xyzr[1], 'z': xyzr[2], 'radius': xyzr[3],
    #                     'type': 'spline'
    #                 })
    # df = pd.DataFrame(df_dicts)
    # res_csv = os.path.join(run_dir, 'smooth_result.csv')
    # df.to_csv(res_csv)

    save_file = os.path.join(run_dir, 'smooth_result.hdf5')
    with h5py.File(save_file, 'w') as f:
        for group_idx, net_cell in opt.network_cells.items():
            group_set = f.create_group(str(group_idx))
            group_set.create_dataset("group_idx", data=[group_idx])
            control_points_layer = group_set.create_group("control_points")
            spline_layer = group_set.create_group("spline")
            path_layer = group_set.create_group("path")

            for seg_idx, seg_cell in net_cell.segments.items():
                control_points_layer.create_dataset(
                    f"seg_{seg_cell.idx}", data=net_cell.control_xyzr_np[seg_cell.pcd_idxs, :]
                )
                spline_layer.create_dataset(
                    f"seg_{seg_cell.idx}", data=seg_cell.get_output_bspline_xyzr_np(net_cell.control_xyzr_np)
                )

            for name, path_list in net_cell.path_dict.items():
                for j, path_cell in enumerate(path_list):
                    path_layer.create_dataset(
                        f"{name}_{j}", data=path_cell.get_output_bspline_xyzr_np(
                            net_cell.control_xyzr_np, with_terminate_vec=True
                        )
                    )

    return jsonify({'Status': True, 'Info': '', 'file': f"smooth_result/{run_name}/smooth_result.hdf5"})


@app.route('/release_smooth_agent', methods=['POST'])
def release_smooth_agent():
    params: Dict = request.form.to_dict()
    proj_id = params['proj_id']
    cfg = json.loads(params['cfg'])
    run_name = cfg['run_name']

    if f"{proj_id}_{run_name}" not in server_manager.path_optimizers_dict.keys():
        return jsonify({'Status': True, 'Info': ''})

    del server_manager.path_optimizers_dict[f"{proj_id}_{run_name}"]
    return jsonify({'Status': True, 'Info': ''})


if __name__ == '__main__':
    server_args = parse_args()
    if server_args.server_dir is not None:
        server_manager = ServerManager(server_args.server_dir)
        app.run(debug=True)
