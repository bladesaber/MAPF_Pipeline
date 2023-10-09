import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict
import math
import os, argparse, shutil
import json

from scripts.visulizer import VisulizerVista


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="the name of config json file",
                        # default='/home/admin123456/Desktop/work/example4/grid_env_cfg.json'
                        default='/home/admin123456/Desktop/work/example4/env_cfg.json'
                        )
    parser.add_argument("--smooth_result_dir", type=str, help="",
                        # default=None
                        default="/home/admin123456/Desktop/work/example4/smoother_result"
                        )
    parser.add_argument("--path_result_file", type=str, help="",
                        # default="/home/admin123456/Desktop/work/example4/result.npy"
                        default=None
                        )
    args = parser.parse_args()
    return args


def show_smooth_path():
    args = parse_args()
    smooth_result_dir = args.smooth_result_dir

    if (smooth_result_dir is None) or (smooth_result_dir == ""):
        return

    with open(args.config_file) as f:
        env_config = json.load(f)

    obstacle_df = pd.read_csv(env_config['obstacle_path'], index_col=0)

    path_names = os.listdir(smooth_result_dir)
    random_colors = np.random.uniform(0.0, 1.0, size=(len(path_names), 3))
    vis = VisulizerVista()

    for i, pathName in enumerate(path_names):
        path_dir = os.path.join(smooth_result_dir, pathName)

        path_info_file = os.path.join(path_dir, 'setting.json')
        with open(path_info_file, 'r') as f:
            path_info = json.load(f)

        group_idx = path_info["groupIdx"]
        xyzs_file = os.path.join(path_dir, 'xyzs.csv')
        xyzs = pd.read_csv(
            xyzs_file, # index_col=0
        )[['x', 'y', 'z']].values

        radius_file = os.path.join(path_dir, 'radius.csv')
        radius = pd.read_csv(
            radius_file, # index_col=0
        )['radius'].values

        tube_mesh = VisulizerVista.create_complex_tube(xyzs, capping=True, radius=None, scalars=radius)
        line_mesh = VisulizerVista.create_line(xyzs)

        # vis.plot(tube_mesh, color=random_colors[i, :], opacity=0.6)
        # vis.plot(line_mesh, color=random_colors[i, :], opacity=0.8)
        vis.plot(tube_mesh, color=random_colors[group_idx, :], opacity=1.0)
        vis.plot(line_mesh, color=random_colors[group_idx, :], opacity=1.0)

    obstacle_xyzs = obstacle_df[obstacle_df['tag'] != 'wall'][['x', 'y', 'z']].values
    obstacle_mesh = VisulizerVista.create_pointCloud(obstacle_xyzs)
    vis.plot(obstacle_mesh, (0.5, 0.5, 0.5))

    vis.show()


def show_planning_path():
    args = parse_args()

    result_file = args.path_result_file
    if (result_file is None) or (result_file == ""):
        return

    with open(args.config_file) as f:
        env_config = json.load(f)

    obstacle_df = pd.read_csv(env_config['obstacle_path'], index_col=0)

    result_pipes: Dict = np.load(result_file, allow_pickle=True).item()

    group_idxs = list(result_pipes.keys())

    vis = VisulizerVista()
    random_colors = np.random.uniform(0.0, 1.0, size=(len(group_idxs), 3))

    for group_idx in group_idxs:
        path_infos = result_pipes[group_idx]
        for path_info in path_infos:
            path_xyzrl = path_info['path_xyzrl']
            radius = path_info['radius']

            tube_mesh = VisulizerVista.create_tube(path_xyzrl[:, :3], radius=radius)
            line_mesh = VisulizerVista.create_line(path_xyzrl[:, :3])
            vis.plot(tube_mesh, color=tuple(random_colors[group_idx]), opacity=0.65)
            vis.plot(line_mesh, color=(1, 0, 0))

    obstacle_xyzs = obstacle_df[obstacle_df['tag'] != 'wall'][['x', 'y', 'z']].values
    obstacle_mesh = VisulizerVista.create_pointCloud(obstacle_xyzs)
    vis.plot(obstacle_mesh, (0.5, 0.5, 0.5))

    vis.show()


def main():
    show_planning_path()
    show_smooth_path()


if __name__ == '__main__':
    main()
