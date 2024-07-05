import numpy as np
import pandas as pd
import sys
import os
import json
import open3d as o3d
import open3d.visualization.gui as gui
from functools import partial
import argparse
import subprocess
import h5py
import matplotlib.pyplot as plt

from scripts_py.version_9.mapf_pkg.app_utils import FragmentOpen3d
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils
from scripts_py.version_9.mapf_pkg.app_smooth import SmoothApp
from scripts_py.version_9.mapf_pkg.pcd2mesh_utils import Pcd2MeshConverter


class MeshApp(SmoothApp):
    def __init__(self, config):
        super().__init__(config=config)

    def _path2pcd(self):
        algo_file = os.path.join(self.proj_dir, 'algorithm_setup.json')
        if not os.path.exists(algo_file):
            self.console_label.text = "    [ERROR]: No algorithm_setup.json in the project"
            return

        res_file = os.path.join(self.proj_dir, 'smooth_result.hdf5')
        if not os.path.exists(res_file):
            self.console_label.text = f"    [ERROR]: smooth result isn't exist."
            return

        with open(algo_file, 'r') as f:
            algo_json = json.load(f)
        pipe_cfg = algo_json['pipes']

        for _, group_cell in h5py.File(res_file).items():
            group_idx = group_cell['group_idx'][0]
            group_pcds = []

            for i, (name, xyzr) in enumerate(group_cell['spline'].items()):
                xyzr = np.array(xyzr)
                for pipe_name, pipe_info in pipe_cfg.items():
                    pipe_xyz = np.array(pipe_info['discrete_position'])
                    pipe_direction = np.array(pipe_info['direction'])
                    is_input = pipe_info['is_input']

                    if np.all(np.isclose(xyzr[0, :3] - pipe_xyz, 0.0)):
                        if is_input:
                            xyzr = np.concatenate([(xyzr[0, :3] - pipe_direction).reshape((1, -1)), xyzr], axis=0)
                        else:
                            xyzr = np.concatenate([(xyzr[0, :3] + pipe_direction).reshape((1, -1)), xyzr], axis=0)

                    if np.all(np.isclose(xyzr[-1, :3] - pipe_xyz, 0.0)):
                        if is_input:
                            xyzr = np.concatenate([xyzr, (xyzr[0, :3] - pipe_direction).reshape((1, -1))], axis=0)
                        else:
                            xyzr = np.concatenate([xyzr, (xyzr[0, :3] + pipe_direction).reshape((1, -1))], axis=0)

                pcd = Pcd2MeshConverter.path2pcd(xyzr[:, :3], xyzr[0, -1], reso=0.1, deplicate_format=2)
                group_pcds.append(pcd)
            group_pcds = np.concatenate(group_pcds, axis=0)

            for i, (name, xyzr) in enumerate(group_cell['spline'].items()):
                xyzr = np.array(xyzr)
                path = xyzr[:, :3]
                group_pcds = Pcd2MeshConverter.filter_inner_pcd(group_pcds, path, radius=xyzr[0, -1])

            pass

    def init_path_smooth_widget(self):
        mesh_layout = gui.CollapsableVert("Path Mesh Setup", self.spacing, self.blank_margins)
        self.panel.add_child(mesh_layout)


def parse_args():
    parser = argparse.ArgumentParser(description="Fluid Tool")
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--proj_dir', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    # o3d.visualization.webrtc_server.enable_webrtc()

    args = parse_args()
    assert args.proj_dir is not None
    config = {'width': args.width, 'height': args.height, 'proj_dir': args.proj_dir}

    app = gui.Application.instance
    app.initialize()
    window = MeshApp(config=config)

    app.run()

    sys.exit("CopyRight From Qu")


if __name__ == '__main__':
    main()
