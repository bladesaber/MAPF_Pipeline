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
                        default=None
                        # default="/home/admin123456/Desktop/work/application/envGridConfig.json"
                        )
    parser.add_argument("--smooth_result_dir", type=str, help="",
                        default=None
                        # default="/home/admin123456/Desktop/work/application/smoother_result"
                        )
    parser.add_argument("--path_result_file", type=str, help="",
                        default=None
                        # default="/home/admin123456/Desktop/work/application/result.npy"
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

    obstacle_df = pd.read_csv(env_config['obstaclePath'], index_col=0)

    pathNames = os.listdir(smooth_result_dir)
    random_colors = np.random.uniform(0.0, 1.0, size=(len(pathNames), 3))
    vis = VisulizerVista()

    for i, pathName in enumerate(pathNames):
        path_dir = os.path.join(smooth_result_dir, pathName)

        pathInfo_file = os.path.join(path_dir, 'setting.json')
        with open(pathInfo_file, 'r') as f:
            pathInfo = json.load(f)

        groupIdx = pathInfo["groupIdx"]

        xyzs_file = os.path.join(path_dir, 'xyzs.csv')
        xyzs = pd.read_csv(
            xyzs_file,
            # index_col=0
        )[['x', 'y', 'z']].values

        radius_file = os.path.join(path_dir, 'radius.csv')
        radius = pd.read_csv(
            radius_file,
            # index_col=0
        )['radius'].values

        tube_mesh = VisulizerVista.create_complex_tube(xyzs, capping=True, radius=None, scalars=radius)
        line_mesh = VisulizerVista.create_line(xyzs)

        # vis.plot(tube_mesh, color=random_colors[i, :], opacity=0.95)
        # vis.plot(line_mesh, color=random_colors[i, :], opacity=0.8)
        vis.plot(tube_mesh, color=random_colors[groupIdx, :], opacity=1.0)
        # vis.plot(line_mesh, color=random_colors[groupIdx, :], opacity=1.0)

    obstacle_xyzs = obstacle_df[obstacle_df['tag'] != 'wall'][['x', 'y', 'z']].values
    obstacle_mesh = VisulizerVista.create_pointCloud(obstacle_xyzs)
    vis.plot(obstacle_mesh, (0.5, 0.5, 0.5))

    vis.show()

def show_planning_path():
    args = parse_args()

    with open(args.config_file) as f:
        env_config = json.load(f)

    obstacle_df = pd.read_csv(env_config['scaleObstaclePath'], index_col=0)

    result_file = args.path_result_file

    if (result_file is None) or (result_file == ""):
        return

    result_pipes: Dict = np.load(result_file, allow_pickle=True).item()
    groupIdxs = list(result_pipes.keys())

    vis = VisulizerVista()
    random_colors = np.random.uniform(0.0, 1.0, size=(len(groupIdxs), 3))

    for groupIdx in groupIdxs:
        pathInfos = result_pipes[groupIdx]

        for pathInfo in pathInfos:
            path_xyzrl = pathInfo['path_xyzrl']
            radius = pathInfo['radius']

            tube_mesh = VisulizerVista.create_tube(path_xyzrl[:, :3], radius=radius)
            line_mesh = VisulizerVista.create_line(path_xyzrl[:, :3])
            vis.plot(tube_mesh, color=tuple(random_colors[groupIdx]), opacity=0.65)
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
