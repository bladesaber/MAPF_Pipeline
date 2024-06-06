import numpy as np
import open3d.io
import pandas as pd
import sys
import os
import json
import open3d as o3d
import open3d.visualization.gui as gui
from functools import partial
import argparse
import subprocess
import pyvista

from scripts_py.version_9.mapf_pkg.app_env import EnvironmentApp
from scripts_py.version_9.mapf_pkg.app_utils import FragmentOpen3d
from scripts_py.version_9.mapf_pkg.networker_block import SimpleBlocker
from scripts_py.version_9.mapf_pkg.algo_init_utils import create_search_init_cfg, create_grid_init_cfg, \
    create_obs_pcd, discrete_pipe_position


class PathApp(EnvironmentApp):
    def __init__(self, config):
        super().__init__(config=config)
        self.init_path_search_widget()

    def _init_setup_file_on_click(self):
        setup_file = os.path.join(self.proj_dir, 'algorithm_setup.json')
        if not os.path.exists(setup_file):
            pipes_cfg = {}
            for name in self.geoMap.keys():
                if self.geoMap[name]['param']['type'] != 'pipe':
                    continue
                pipe_info = {
                    'radius': self.geoMap[name]['param']['radius'],
                    'position': self.geoMap[name]['param']['position'],
                    'direction': self.geoMap[name]['param']['direction'],
                    'group_idx': self.geoMap[name]['param']['group_idx'],
                    'is_input': self.geoMap[name]['param']['is_input']
                }
                pipes_cfg[name] = pipe_info

            block_priority = create_search_init_cfg(pipes_cfg)
            grid_env = create_grid_init_cfg(
                self.global_widget['grid_min'].vector_value,
                self.global_widget['grid_max'].vector_value
            )

            setup_json = {
                'grid_env': grid_env,
                'search_tree': {
                    'method': 'simple',
                    'block_sequence': block_priority
                },
                'pipes': pipes_cfg
            }

            with open(setup_file, 'w') as f:
                json.dump(setup_json, f, indent=4)

        subprocess.run(f"gedit {setup_file};", shell=True)

    def _create_search_network_on_click(self):
        setup_file = os.path.join(self.proj_dir, 'algorithm_setup.json')
        if not os.path.exists(setup_file):
            self.console_label.text = "    [ERROR]: No algorithm_setup.json in the project"
            return

        with open(setup_file, 'r') as f:
            setup_json = json.load(f)

        pipes_cfg = setup_json['pipes']
        blocks_seq = setup_json['search_tree']['block_sequence']
        if setup_json['search_tree']['method'] == 'simple':
            SimpleBlocker.create_block_network(pipes_cfg, blocks_seq)

        with open(setup_file, 'w') as f:
            json.dump(setup_json, f, indent=4)

        subprocess.run(f"gedit {setup_file};", shell=True)

    def _create_grid_env_on_click(self):
        setup_file = os.path.join(self.proj_dir, 'algorithm_setup.json')
        if not os.path.exists(setup_file):
            self.console_label.text = "    [ERROR]: No algorithm_setup.json in the project"
            return

        with open(setup_file, 'r') as f:
            setup_json = json.load(f)

        grid_env = setup_json['grid_env']
        if (grid_env['num_of_x'] is None) or (grid_env['num_of_y'] is None) or (grid_env['num_of_z'] is None):
            self.console_label.text = "    [Error]: grid num is 0."
            return
        if np.any(np.array(grid_env['grid_max']) - np.array(grid_env['grid_min']) <= 0):
            self.console_label.text = "    [Error]: grid max is smaller than grid small."
            return

        pipes_cfg = setup_json['pipes']

        obs_cfg = {}
        for name in self.geoMap.keys():
            if self.geoMap[name]['param']['type'] == 'obstacle':
                obs_cfg[name] = {'pcd': np.asarray(self.geoMap[name]['geometry'].points)}

        # ------ step 1 make pipe position discrete
        pipes_cfg = discrete_pipe_position(
            pipes_cfg=pipes_cfg, grid_min=np.array(grid_env['grid_min']), grid_max=np.array(grid_env['grid_max']),
            num_of_x=grid_env['num_of_x'], num_of_y=grid_env['num_of_y'], num_of_z=grid_env['num_of_z']
        )

        # ------ step 2 create obstacle
        obs_pcd = create_obs_pcd(pipes_cfg, obs_cfg)
        pcd_o3d = FragmentOpen3d.create_pcd(obs_pcd)
        self.add_point_cloud(
            'grid_obstacle', pcd_o3d, is_visible=False,
            param={
                'type': 'obstacle', 'shape': 'grid_obstacle', 'rgb': [0.2, 0.4, 0.6]
            }
        )
        grid_env['obstacle_tag'] = 'grid_obstacle'

        with open(setup_file, 'w') as f:
            json.dump(setup_json, f, indent=4)

        subprocess.run(f"gedit {setup_file};", shell=True)

    def _define_search_setting_on_click(self):
        setup_file = os.path.join(self.proj_dir, 'algorithm_setup.json')
        if not os.path.exists(setup_file):
            self.console_label.text = "    [ERROR]: No algorithm_setup.json in the project"
            return

        with open(setup_file, 'r') as f:
            setup_json = json.load(f)

        for block_info in setup_json['search_tree']['block_sequence']:
            for group_info in block_info['groups']:
                for seq_info in group_info['sequence']:
                    if 'x_scale' not in seq_info[2].keys():
                        seq_info[2]['x_scale'] = None
                    if 'y_scale' not in seq_info[2].keys():
                        seq_info[2]['y_scale'] = None
                    if 'z_scale' not in seq_info[2].keys():
                        seq_info[2]['z_scale'] = None

        with open(setup_file, 'w') as f:
            json.dump(setup_json, f, indent=4)

        subprocess.run(f"gedit {setup_file};", shell=True)

    def _step_search_on_click(self):
        pcd = np.random.random((10000, 3))
        mesh = pyvista.PointSet(pcd)
        mesh.plot()

    def init_path_search_widget(self):
        algo_layout = gui.CollapsableVert("Path Search Setup", self.spacing, self.blank_margins)
        self.panel.add_child(algo_layout)

        init_setup_file_btn = FragmentOpen3d.get_widget('button', {'name': 'edit setup file'})
        init_setup_file_btn.set_on_clicked(self._init_setup_file_on_click)

        create_search_network_btn = FragmentOpen3d.get_widget('button', {'name': 'create search network'})
        create_search_network_btn.set_on_clicked(self._create_search_network_on_click)

        create_grid_environment_btn = FragmentOpen3d.get_widget('button', {'name': 'create grid environment'})
        create_grid_environment_btn.set_on_clicked(self._create_grid_env_on_click)

        define_search_setting_btn = FragmentOpen3d.get_widget('button', {'name': 'define search setting'})
        define_search_setting_btn.set_on_clicked(self._define_search_setting_on_click)

        step_search_btn = FragmentOpen3d.get_widget('button', {'name': 'step search'})
        step_search_btn.set_on_clicked(self._step_search_on_click)
        auto_search_btn = FragmentOpen3d.get_widget('button', {'name': 'auto search'})

        algo_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', [
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('step1:'), init_setup_file_btn]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('step2:'), create_search_network_btn]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('step3:'), create_grid_environment_btn]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('step4:'), define_search_setting_btn]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('step5:'), step_search_btn, auto_search_btn]),
            ], 3, self.vert_margins)
        )


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
    window = PathApp(config=config)

    app.run()

    sys.exit("CopyRight From Qu")


if __name__ == '__main__':
    main()
