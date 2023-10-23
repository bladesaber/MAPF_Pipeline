"""

author qianyuzhang (@SinoDynamics)
23/10/13
Handle the requests from the api_frontend and bridge the 3d_trajectory_planning_methods to generate and send results to frontend
23/10/22
To-do-1: the smoothing output in 'smoother_result' needs to be serialized in func 'serialize_smooth_result()' which will be sent to api_frontend.py
"""

import json
from json import JSONEncoder
import numpy as np
import os
# from scripts.version_9.app.o3d_envApp import CustomApp
import argparse
from typing import Dict
from scripts.version_9.mapf_pipeline_py.grid_fit_env import fit_env
from scripts.version_9.mapf_pipeline_py.cbs_planner import CBS_Planner
from scripts.version_9.mapf_pipeline_py.path_smoother import FlexPathSmoother

DEFAULT_PROJS_DIR = os.path.join(os.getcwd(), 'Examples'),

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def serialize_smooth_result():
    temp = 1
def locate_project(project_id):
    proj_name = 'Exmaple' + str(project_id)
    project_dir = os.path.join(DEFAULT_PROJS_DIR, proj_name)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    return project_dir

def write_json_to_file(file_path, data):
    with open(file_path, 'w') as json_file:
        json.dump(data,json_file)

def saveEnvJson(project_id, json_dict):
    proj_dir = locate_project(project_id)
    env_cfg_filepath = os.path.join(proj_dir,'env_cfg.json')
    if not os.path.exists(env_cfg_filepath):
        os.makedirs(env_cfg_filepath)
    write_json_to_file(env_cfg_filepath, json_dict)
    print('save environment configuration successfully.')

def generatePipesLinkConfigJson(project_id, json_dict):
    proj_dir = locate_project(project_id)
    pipeLink_setting_filepath = os.path.join(proj_dir,'pipeLink_setting.json')
    if not os.path.exists(pipeLink_setting_filepath):
        os.makedirs(pipeLink_setting_filepath)
    write_json_to_file(pipeLink_setting_filepath, json_dict)
    print('generate Pipes Link Configuration successfully.')

def generateOptSettingsJson(project_id, json_dict):
    proj_dir = locate_project(project_id)
    opt_settings_filepath = os.path.join(proj_dir,'optimize_setting.json')
    if not os.path.exists(opt_settings_filepath):
        os.makedirs(opt_settings_filepath)
    write_json_to_file(opt_settings_filepath, json_dict)
    print('generate Optimization Settings successfully.')

def loadJson(project_id, json_type):
    proj_dir = locate_project(project_id)

    if json_type == 'load_Env_Config':
        env_cfg_filepath = os.path.join(proj_dir, 'env_cfg.json')
        try:
            if os.path.exists(env_cfg_filepath):
                with open(env_cfg_filepath, 'r') as f:
                    Env_Config = json.load(f)
                    print('successfully loadJson: Environment_Configuration')
                    return Env_Config
        except:
            raise ValueError('applied file not exists')

    elif json_type == 'load_PipeLinks_Config':
        pipeLink_setting_filepath = os.path.join(proj_dir, 'pipeLink_setting.json')
        try:
            if os.path.exists(pipeLink_setting_filepath):
                with open(pipeLink_setting_filepath, 'r') as f:
                    PipeLinks_Config = json.load(f)
                    print('successfully loadJson: PipeLinks Configuration')
                    return PipeLinks_Config

        except:
            raise ValueError('applied file not exists')

    elif json_type == 'load_OptSettings':
        opt_settings_filepath = os.path.join(proj_dir,'optimize_setting.json')
        try:
            if os.path.exists(opt_settings_filepath):
                with open(opt_settings_filepath, 'r') as f:
                    OptSettings = json.load(f)
                    print('successfully loadJson: Optimization Settings')
                    return OptSettings
        except:
            raise ValueError('applied file not exists')

    elif json_type == 'load_Grid_Env_Config':
        grid_env_filepath = os.path.join(proj_dir,'grid_env_cfg.json')
        try:
            if os.path.exists(grid_env_filepath):
                with open(grid_env_filepath, 'r') as f:
                    Grid_Env_Config = json.load(f)
                    print('successfully loadJson: Grid Environment Configuration')
                    return Grid_Env_Config
        except:
            raise ValueError('applied file not exists')

def discretize_env(project_id):
    try:
        proj_dir = locate_project(project_id)
        env_cfg_filepath = os.path.join(proj_dir, 'env_cfg.json')
        grid_env_filepath = os.path.join(proj_dir, 'grid_env_cfg.json')

        parser = argparse.ArgumentParser(description="Grid Environment")
        parser.add_argument(
            "--env_json", type=str, help="project json file",
            default=env_cfg_filepath
        )
        parser.add_argument("--scale", type=float, help="scale ratio", default=0.8)
        parser.add_argument("--create_shell", type=int, help="create shell (bool)", default=0)
        parser.add_argument("--scale_reso", type=float, help="create shell (bool)", default=1.5)
        args = parser.parse_args()
        fit_env(args)
        with open (grid_env_filepath,r) as f:
            grid_env_cfg = json.load(f)
        print('discretize_env successfully')
        return grid_env_cfg # need to be updated as a json contained project_id
    except:
        raise ValueError('environment discretize error')

def search_path(project_id):
    try:
        proj_dir = locate_project(project_id)
        grid_env_cfg_filepath = os.path.join(proj_dir, 'grid_env_cfg.json')

        # parser = argparse.ArgumentParser()
        # parser.add_argument(
        #     "--config_file", type=str, help="the name of config json file",
        #     default=grid_env_filepath
        # )
        # parser.add_argument("--save_file", type=str, help="project directory", default="result.npy")
        # args = parser.parse_args()

        debug_dir = os.path.join(proj_dir, 'debug')
        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)

        save_file = os.path.join(proj_dir, 'result.npy')
        Json_result = os.path.join(proj_dir, 'searching_result.json')

        cbs_planner = CBS_Planner(grid_env_cfg_filepath, debug_dir=debug_dir)
        cbs_planner.init_environment()
        cbs_planner.solve(save_file=save_file)

        # Load the numpy array from .npy file
        ndarray_data = np.load(save_file, allow_pickle=True)

        # Convert the numpy array to a regular Python list
        python_list = ndarray_data.tolist()

        # Use NumpyArrayEncoder when dumping to JSON
        with open(Json_result, 'w') as json_file:
            json.dump(python_list, json_file, cls=NumpyArrayEncoder)
        print('search_path successfully')
        return(json_file)
    except:
        raise ValueError('paths search error')

def optimize_path(project_id):
    try:
        proj_dir = locate_project(project_id)
        # env_cfg_filepath = os.path.join(proj_dir, 'env_cfg.json')
        grid_env_cfg_filepath = os.path.join(proj_dir, 'grid_env_cfg.json')
        with open (grid_env_cfg_filepath,'r') as f:
            env_cfg = json.load(f)

        optimize_setting_filepath = os.path.join(proj_dir, 'optimize_setting.json')
        with open (optimize_setting_filepath,'r') as f:
            optimize_setting = json.load(f)

        pipe_setting_filepath = os.path.join(proj_dir, 'pipeLink_setting.json')
        with open(pipe_setting_filepath, 'r') as f:
            path_links = json.load(f)
        for groupIdx in list(path_links.keys()):
            path_links[int(groupIdx)] = path_links[groupIdx]
            del path_links[groupIdx]

        result_filepath = os.path.join(proj_dir, 'result.npy')
        result_pipes: Dict = np.load(result_filepath, allow_pickle=True).item()
        Json_result = os.path.join(proj_dir, 'path_smoothed_result.json')

        default_optimize_times = 200

        default_verbose = 0

        smoother = FlexPathSmoother(
            env_cfg,
            optimize_setting=optimize_setting,
            with_elasticBand=True,
            with_kinematicEdge=True,
            with_obstacleEdge=True,
            with_pipeConflictEdge=True,
        )
        smoother.init_environment()

        smoother.createWholeNetwork(result_pipes)
        # smoother.plotGroupPointCloudEnv()

        smoother.definePath(path_links)

        smoother.create_flexGraphRecord()

        smoother.optimize(outer_times=default_optimize_times, verbose=default_verbose)

        optimize_dir = os.path.join(proj_dir, 'smoother_result')
        if not os.path.exists(optimize_dir):
            os.mkdir(optimize_dir)

        smoother.output_result(optimize_dir)

        smoothed_result = serialize_smooth_result(optimize_dir)

        with open(Json_result, 'w') as json_file:
            json.dump(smoothed_result,json_file)
        print('smooth_path successfully')
        return (json_file)

    except:
        raise ValueError('paths smooth error')