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
from scripts_py.version_9.mapf_pkg.algo_utils import create_search_init_cfg, \
    create_obs_pcd, discrete_pipe_position
from scripts_py.version_9.mapf_pkg.cbs_utils import CbsSolver
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils


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
            grid_env = {
                'grid_min': self.global_widget['grid_min'].vector_value.tolist(),
                'grid_max': self.global_widget['grid_max'].vector_value.tolist(),
                'size_of_x': None, 'size_of_y': None, 'size_of_z': None,
            }

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
        if (grid_env['size_of_x'] is None) or (grid_env['size_of_y'] is None) or (grid_env['size_of_z'] is None):
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
        grid_min_np = np.array(grid_env['grid_min'])
        grid_max_np = np.array(grid_env['grid_max'])
        size_of_xyz = np.array([grid_env['size_of_x'], grid_env['size_of_y'], grid_env['size_of_z']])
        length_of_xyz = (grid_max_np - grid_min_np) / size_of_xyz
        pipes_cfg = discrete_pipe_position(pipes_cfg, grid_min=grid_min_np, length_of_xyz=length_of_xyz)

        # ------ step 2 create obstacle
        obs_pcd_np = create_obs_pcd(pipes_cfg, obs_cfg)
        pcd_o3d = FragmentOpen3d.create_pcd(obs_pcd_np)
        self.add_point_cloud(
            'grid_obstacle', pcd_o3d, is_visible=False,
            param={'type': 'obstacle', 'shape': 'grid_obstacle', 'rgb': [0.2, 0.4, 0.6]}
        )

        obs_df = pd.DataFrame(obs_pcd_np, columns=['x', 'y', 'z'])
        obs_df['radius'] = 0.0

        obstacle_save_file = os.path.join(self.proj_dir, 'static_obstacle.csv')
        obs_df.to_csv(obstacle_save_file)
        grid_env['obstacle_tag'] = 'grid_obstacle'
        grid_env['obstacle_file'] = obstacle_save_file

        grid_env['x_grid_length'] = length_of_xyz[0]
        grid_env['y_grid_length'] = length_of_xyz[1]
        grid_env['z_grid_length'] = length_of_xyz[2]

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
                    task_info: dict = seq_info[2]
                    task_info.setdefault('task_name', None)
                    task_info.setdefault('step_scale', None)
                    task_info.setdefault('shrink_distance', None)
                    task_info.setdefault('shrink_scale', None)
                    task_info.setdefault('expand_grid_method', None)
                    task_info.setdefault('with_curvature_cost', None)
                    task_info.setdefault('curvature_cost_weight', None)
                    task_info.setdefault('use_constraint_avoid_table', None)
                    task_info.setdefault('with_theta_star', None)

        with open(setup_file, 'w') as f:
            json.dump(setup_json, f, indent=4)

        subprocess.run(f"gedit {setup_file};", shell=True)

    def _step_search_on_click(self, debug_checkbox: gui.Checkbox):
        setup_file = os.path.join(self.proj_dir, 'algorithm_setup.json')
        if not os.path.exists(setup_file):
            self.console_label.text = "    [ERROR]: No algorithm_setup.json in the project"
            return

        with open(setup_file, 'r') as f:
            setup_json = json.load(f)

        grid_cfg = setup_json['grid_env']
        pipe_cfg = setup_json['pipes']

        block_res_list = []
        last_leaf_info = {}
        for block_info in setup_json['search_tree']['block_sequence']:
            solver = CbsSolver(grid_cfg, pipe_cfg)
            root = solver.init_block_root(block_info['groups'], last_leafs_info=last_leaf_info)
            group_idxs = list(solver.task_infos.keys())
            is_success, res_node = solver.solve_block(root, group_idxs, max_iter=200)

            if is_success:
                if debug_checkbox.checked:
                    CbsSolver.draw_node_3D(
                        res_node, group_idxs=group_idxs, task_infos=solver.task_infos, obstacle_df=solver.obstacle_df
                    )

                last_leaf_info = solver.convert_node_to_leaf_info(res_node, group_idxs)
                block_res_list.append(solver.save_path(res_node, group_idxs, solver.task_infos))

            else:
                print(f"[INFO]: False at solving {block_info['block_id']}")
                break

        # ------
        vis = VisUtils()

        obs_df = pd.read_csv(setup_json['grid_env']['obstacle_file'], index_col=0)
        mesh = VisUtils.create_point_cloud(obs_df[['x', 'y', 'z']].values)
        vis.plot(mesh, color=(0.3, 0.3, 0.3), opacity=1.0)

        for block_res in block_res_list:
            for group_idx, group_res in block_res.items():
                color = np.random.uniform(0.0, 1.0, size=(3,))
                for task_name, xyzr in group_res.items():
                    line_set = np.arange(0, xyzr.shape[0], 1)
                    line_set = np.insert(line_set, 0, line_set.shape[0])
                    radius = np.unique(xyzr[:, -1])[0]
                    tube_mesh = VisUtils.create_tube(xyzr[:, :3], radius, line_set)
                    line_mesh = VisUtils.create_line(xyzr[:, :3], line_set)
                    vis.plot(tube_mesh, color=color, opacity=0.6)
                    vis.plot(line_mesh, (0., 0., 0.), opacity=1.0)

        vis.show()

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

        debug_checkbox = FragmentOpen3d.get_widget('checkbox', {'name': 'DebugMode'})
        search_btn = FragmentOpen3d.get_widget('button', {'name': 'search'})
        search_btn.set_on_clicked(partial(self._step_search_on_click, debug_checkbox=debug_checkbox))

        algo_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', [
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('step1:'), init_setup_file_btn]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('step2:'), create_search_network_btn]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('step3:'), create_grid_environment_btn]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('step4:'), define_search_setting_btn]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('step5:'), search_btn, debug_checkbox]),
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
