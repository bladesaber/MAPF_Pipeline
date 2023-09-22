import numpy as np
import pandas as pd
import argparse
import json
import os
import math

from scripts.version_9.app.env_utils import Shape_Utils
from scripts.version_9.mapf_pipeline_py.spanTree_TaskAllocator import SizeTreeTaskRunner

'''
2023-09-22
栅格化在部分情况会出问题，主要是pipe的grid position会移动，部分情况下会移动到障碍物内部
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Grid Environment")
    parser.add_argument(
        "--env_json", type=str, help="project json file",
        default="/home/admin123456/Desktop/work/example1/env_cfg.json"
    )
    parser.add_argument("--scale", type=float, help="scale ratio", default=0.35)
    parser.add_argument("--create_shell", type=int, help="create shell (bool)", default=1)
    args = parser.parse_args()
    return args


def create_task_tree(pipes_cfg):
    task_trees = {}
    for group_idx_str in pipes_cfg:
        group_pipes_info = pipes_cfg[group_idx_str]
        allocator = SizeTreeTaskRunner()

        for name in group_pipes_info.keys():
            info = group_pipes_info[name]
            allocator.add_node(name, pose=info['position'], radius=info['radius'])
        allocator_res = allocator.getTaskTrees(method='method2')

        res = []
        for name_i, name_j, radius in allocator_res:
            res.append({'name0': name_i, 'name1': name_j, 'radius': radius})
        task_trees[group_idx_str] = res
    return task_trees


def fit_env(args):
    with open(args.env_json, 'r') as f:
        env_cfg = json.load(f)

    global_params: dict = env_cfg['global_params']
    pipes_cfg: dict = env_cfg['pipe_cfgs']
    obstacle_cfgs: dict = env_cfg['obstacle_cfgs']

    scale = args.scale
    if scale <= 0. or scale > 1.:
        print('[Warning]: invalid scale parameter')
        return

    minimum_reso = np.inf
    for group_idx_str in pipes_cfg.keys():
        group_infos = pipes_cfg[group_idx_str]
        for pipe_name in group_infos.keys():
            pipe_info = group_infos[pipe_name]
            minimum_reso = np.minimum(pipe_info['radius'], minimum_reso)
    scale_reso = minimum_reso * scale * 1.25

    # ------ fit env
    global_params['envScaleX'] = global_params['envX'] * scale
    global_params['envScaleY'] = global_params['envY'] * scale
    global_params['envScaleZ'] = global_params['envZ'] * scale
    global_params['envX'] = math.floor(global_params['envScaleX'])
    global_params['envY'] = math.floor(global_params['envScaleY'])
    global_params['envZ'] = math.floor(global_params['envScaleZ'])
    global_params['scale'] = scale

    # ------ fit obstacle
    obstacle_dict = {}
    for obstacle_name in obstacle_cfgs.keys():
        info = obstacle_cfgs[obstacle_name]
        if info['shape'] == 'Box':
            fit_desc = {
                'xmin': info['xmin'] * scale, 'xmax': info['xmax'] * scale,
                'ymin': info['ymin'] * scale, 'ymax': info['ymax'] * scale,
                'zmin': info['zmin'] * scale, 'zmax': info['zmax'] * scale,
                'shape_reso': scale_reso
            }
            info.update(fit_desc)
            obs_xyzs = Shape_Utils.create_BoxPcd(
                info['xmin'], info['ymin'], info['zmin'], info['xmax'], info['ymax'], info['zmax'], info['shape_reso']
            )
        elif info['shape'] == 'Cylinder':
            fit_desc = {
                'position': list(np.array(info['position']) * scale),
                'radius': info['radius'] * scale,
                'height': info['height'] * scale,
                'shape_reso': scale_reso,
            }
            info.update(fit_desc)
            obs_xyzs = Shape_Utils.create_CylinderPcd(
                np.array(info['position']), info['radius'], info['height'], info['direction'], info['shape_reso']
            )
        else:
            raise NotImplementedError

        obstacle_dict[obstacle_name] = obs_xyzs

    # ------ fit pipe
    for group_idx_str in pipes_cfg.keys():
        group_infos = pipes_cfg[group_idx_str]
        for pipe_name in group_infos.keys():
            info = group_infos[pipe_name]

            xyz = np.array(info['position']) * scale
            xyz_grid = np.round(xyz, decimals=0)
            xyz_grid[0] = np.maximum(np.minimum(xyz_grid[0], global_params['envScaleX']), 0)
            xyz_grid[1] = np.maximum(np.minimum(xyz_grid[1], global_params['envScaleY']), 0)
            xyz_grid[2] = np.maximum(np.minimum(xyz_grid[2], global_params['envScaleZ']), 0)

            fit_desc = {
                'position': [int(xyz_grid[0]), int(xyz_grid[1]), int(xyz_grid[2])],
                'scale_position': list(xyz),
                'radius': info['radius'] * scale,
            }
            info.update(fit_desc)

    task_tree = create_task_tree(pipes_cfg)
    env_cfg['task_tree'] = task_tree

    shell_filters = {}
    for group_idx_str in pipes_cfg.keys():
        groupPipes_info = pipes_cfg[group_idx_str]

        for pipe_name in groupPipes_info.keys():
            info = groupPipes_info[pipe_name]
            shell_name = f"{pipe_name}_shell"

            position_dif = np.array(info['scale_position']) - np.array(info['position'])
            direction = info['direction']
            radius = info['radius']
            xmin_dist, xmax_dist = radius + 0.15, radius + 0.15
            ymin_dist, ymax_dist = radius + 0.15, radius + 0.15
            zmin_dist, zmax_dist = radius + 0.15, radius + 0.15

            if direction[0] == 1:
                if position_dif[0] > 0:
                    xmax_dist += position_dif[0]
                elif position_dif[0] < 0:
                    xmin_dist += abs(position_dif[0])
            elif direction[1] == 1:
                if position_dif[1] > 0:
                    ymax_dist += position_dif[1]
                elif position_dif[1] < 0:
                    ymin_dist += abs(position_dif[1])
            else:
                if position_dif[2] > 0:
                    zmax_dist += position_dif[2]
                elif position_dif[2] < 0:
                    zmin_dist += abs(position_dif[2])

            shell_xyzs, shell_range = Shape_Utils.create_PipeBoxShell(
                xyz=np.array(info['position']),
                xmin_dist=xmin_dist, xmax_dist=xmax_dist,
                ymin_dist=ymin_dist, ymax_dist=ymax_dist,
                zmin_dist=zmin_dist, zmax_dist=zmax_dist,
                direction=info['direction'],
                reso=np.minimum(scale_reso, info['radius'] * 0.4)
            )
            shell_filters[shell_name] = shell_range

            if args.create_shell > 0:
                obstacle_cfgs[shell_name] = {
                    'shape': 'shell',
                    'pipe_name': pipe_name
                }
                obstacle_dict[shell_name] = shell_xyzs

    for name in obstacle_cfgs.keys():
        info = obstacle_cfgs[name]
        xyzs = obstacle_dict[name]

        if info['shape'] != 'shell':
            for shell_name in shell_filters.keys():
                shell_range = shell_filters[shell_name]
                xyzs = Shape_Utils.removePointInBoxShell(xyzs, shell_range)
        else:
            for shell_name in shell_filters.keys():
                if shell_name == name:
                    continue
                shell_range = shell_filters[shell_name]
                xyzs = Shape_Utils.removePointInBoxShell(xyzs, shell_range)

        xyzs = Shape_Utils.removePointOutBoundary(
            xyzs,
            xmin=0, xmax=global_params['envScaleX'],
            ymin=0, ymax=global_params['envScaleY'],
            zmin=0, zmax=global_params['envScaleZ']
        )
        obstacle_dict[name] = xyzs

    obstacle_dfs = []
    for name in obstacle_dict.keys():
        df = pd.DataFrame(obstacle_dict[name], columns=['x', 'y', 'z'])
        df['radius'] = 0.0
        df['tag'] = name
        obstacle_dfs.append(df)
    obstacle_dfs = pd.concat(obstacle_dfs, axis=0, ignore_index=True)
    obstacle_file = os.path.join(env_cfg['project_dir'], 'scale_obstacle.csv')
    obstacle_dfs.to_csv(obstacle_file)
    env_cfg['obstacle_path'] = obstacle_file

    cfg_file = os.path.join(env_cfg['project_dir'], 'grid_env_cfg.json')
    with open(cfg_file, 'w') as f:
        json.dump(env_cfg, f, indent=4)


def main():
    args = parse_args()
    fit_env(args)


if __name__ == '__main__':
    main()
