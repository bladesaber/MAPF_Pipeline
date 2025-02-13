import os
import shutil
import numpy as np
import pandas as pd
import open3d as o3d
import json
from typing import Dict
import argparse
import h5py
from io import BytesIO, StringIO
import tempfile

from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from flask_cors import *

from scripts_py.version_9.easy_server.cad_server import CadServer

from scripts_py.version_9.mapf_pkg.networker_block import SimpleBlocker
from scripts_py.version_9.mapf_pkg.algo_utils import create_obs_pcd, discrete_pipe_position
from scripts_py.version_9.mapf_pkg.cbs_utils import CbsSolver
# from scripts_py.version_9.mapf_pkg.smooth_optimizer import PathOptimizer
from scripts_py.version_9.mapf_pkg.shape_utils import ShapeUtils
from scripts_py.version_9.mapf_pkg.cad_utils import OcctProfile, OcctUtil, OcctIO

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


@app.route('/<path:filename>', methods=['GET'])
@cross_origin()
def serve_html(filename):
    return send_from_directory('/home/admin123456/Desktop/work/flask_server/storage', filename)


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


@app.route('/export_STEP', methods=['POST'])
@cross_origin()
def export_STEP():
    json_data = json.load(BytesIO(request.data))
    df = pd.read_csv(StringIO(json_data['data']))

    spline_xyz = df[['x', 'y', 'z']].values.astype(float)
    spline_vec = df[['v_x', 'v_y', 'v_z']].values.astype(float)
    radius_list = df['r'].values.astype(float)

    wire_list = []
    for xyz, vec, radius in zip(spline_xyz, spline_vec, radius_list):
        circle = OcctProfile.make_circle(xyz, vec, radius)
        circle_wire = OcctUtil.make_wire([OcctUtil.make_edge(circle)])
        wire_list.append(circle_wire)
    solid = OcctProfile.make_thruSections(wire_list, is_solid=True, ruled=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_file = os.path.join(temp_dir, 'generate.STEP')
        OcctIO.write_STEP(save_file, solid)
        return send_file(save_file, as_attachment=False)


@app.route('/cad_generate_pipe_shell', methods=['POST'])
@cross_origin()
def cad_generate_pipe_shell():
    pipe_cfg = json.loads(json.loads(request.data)['data'])
    shell = CadServer.make_arbitrary_thruSections(pipe_cfg['path_cfg'], ruled=pipe_cfg['ruled'])
    content = CadServer.generate_shape_json(pipe_cfg['name'], shell)
    return jsonify(content)


@app.route('/cad_generate_bspline_surface', methods=['POST'])
@cross_origin()
def cad_generate_bspline_surface():
    surface_cfg = json.loads(json.loads(request.data)['data'])
    shell = CadServer.make_spline_surface(
        boundaries_info=surface_cfg['boundaries_info'],
        guide_info=surface_cfg['guide_info'],
        sample_num=surface_cfg['sample_num']
    )
    content = CadServer.generate_shape_json(surface_cfg['name'], shell)
    return jsonify(content)


if __name__ == '__main__':
    server_args = parse_args()
    if server_args.server_dir is not None:
        server_manager = ServerManager(server_args.server_dir)
        app.run(host='192.168.110.47', port=9999, debug=True)
