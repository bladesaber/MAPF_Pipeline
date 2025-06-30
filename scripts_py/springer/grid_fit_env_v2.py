import numpy as np
import pandas as pd
import argparse
import json
import os
import math

from scripts_py.version_9.mapf_pkg.shape_utils import Shape_Utils
from scripts_py.version_9.mapf_pipeline_py.spanTree_TaskAllocator import SizeTreeTaskRunner


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

    # limit scale of step_length which is 1.0, is 4 * radius
    scale_limit = (1.0 / 2.5) / minimum_reso
    if scale < scale_limit:
        print(f"[Warning]: scale parameter is too small, the limit is {scale_limit}")
        return

    if args.scale_reso is None:
        scale_reso = minimum_reso * scale
    else:
        scale_reso = args.scale_reso

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
            if info['isSolid']:
                obs_xyzs = Shape_Utils.create_BoxSolidPcd(
                    info['xmin'], info['ymin'], info['zmin'], info['xmax'], info['ymax'], info['zmax'], info['shape_reso']
                )
            else:
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
            if info['isSolid']:
                obs_xyzs = Shape_Utils.create_CylinderSolidPcd(
                    np.array(info['position']), info['radius'], info['height'], info['direction'], info['shape_reso']
                )
            else:
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
            shell_filters[shell_name] = {
                'center': np.array(info['position']),
                'radius': info['radius'],
                'direction': info['direction'],
                'reso': scale_reso,
                'scale_dist': scale_reso / 2.0 * 1.5
            }

    for name in obstacle_cfgs.keys():
        info = obstacle_cfgs[name]
        xyzs = obstacle_dict[name]

        if info['shape'] != 'shell':
            for shell_name in shell_filters.keys():
                remove_info = shell_filters[shell_name]
                xyzs = Shape_Utils.removePointInSphereShell(
                    xyzs, center=remove_info['center'], radius=remove_info['radius'], with_bound=False,
                    direction=remove_info['direction'], reso=remove_info['reso'], scale_dist=remove_info['scale_dist']
                )

        xyzs = Shape_Utils.removePointOutBoundary(
            xyzs,
            xmin=0, xmax=global_params['envScaleX'],
            ymin=0, ymax=global_params['envScaleY'],
            zmin=0, zmax=global_params['envScaleZ']
        )
        obstacle_dict[name] = {
            'xyzs': xyzs,
            'radius': info['shape_reso'] / 4.0
        }

    obstacle_dfs = []
    for name in obstacle_dict.keys():
        df = pd.DataFrame(obstacle_dict[name]['xyzs'], columns=['x', 'y', 'z'])
        df['radius'] = obstacle_dict[name]['radius']
        df['tag'] = name
        obstacle_dfs.append(df)
    obstacle_dfs = pd.concat(obstacle_dfs, axis=0, ignore_index=True)
    obstacle_file = os.path.join(env_cfg['project_dir'], 'scale_obstacle.csv')
    obstacle_dfs.to_csv(obstacle_file)
    env_cfg['obstacle_path'] = obstacle_file

    cfg_file = os.path.join(env_cfg['project_dir'], 'grid_env_cfg.json')
    with open(cfg_file, 'w') as f:
        json.dump(env_cfg, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Grid Environment")
    parser.add_argument(
        "--env_json", type=str, help="project json file",
        default="/home/admin123456/Desktop/work/example7/env_cfg.json"
    )
    parser.add_argument("--scale", type=float, help="scale ratio", default=1.0)
    parser.add_argument("--scale_reso", type=float, help="", default=1.0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    fit_env(args)


if __name__ == '__main__':
    main()
