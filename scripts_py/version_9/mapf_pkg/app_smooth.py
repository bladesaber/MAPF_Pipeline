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
import h5py
from typing import Dict
import matplotlib.pyplot as plt

from scripts_py.version_9.mapf_pkg.app_path import PathApp
from scripts_py.version_9.mapf_pkg.app_utils import FragmentOpen3d
from scripts_py.version_9.mapf_pkg.smooth_optimizer import PathOptimizer
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils


class SmoothApp(PathApp):
    def __init__(self, config):
        super().__init__(config=config)

        line_material = o3d.visualization.rendering.MaterialRecord()
        line_material.shader = 'unlitLine'
        line_material.line_width = 10
        self.materials.update({'line_material': line_material})

        self.init_path_smooth_widget()
        self.smooth_opt: PathOptimizer = None
        self.smooth_tmp = {
            'revert': False,
            'last_step_data': {}
        }

    def _vis_search_result_on_click(
            self, with_line_checkbox: gui.Checkbox, with_tube_checkbox: gui.Checkbox,
            with_obstacle_checkbox: gui.Checkbox, line_opacity_txt: gui.NumberEdit,
            tube_opacity_txt: gui.NumberEdit
    ):
        algo_file = os.path.join(self.proj_dir, 'algorithm_setup.json')
        with open(algo_file, 'r') as f:
            algo_json = json.load(f)

        result_file = os.path.join(self.proj_dir, 'search_result.hdf5')
        if not os.path.exists(result_file):
            self.console_label.text = f"    [ERROR]: there isn't search result."

        res = h5py.File(result_file)
        vis = VisUtils()

        for _, res_cell in res.items():
            group_idx = res_cell['group_idx'][0]
            color = np.random.uniform(0.0, 1.0, size=(3,))
            for name, xyzr in res_cell['path_result'].items():
                line_set = np.arange(0, xyzr.shape[0], 1)
                line_set = np.insert(line_set, 0, line_set.shape[0])

                if with_line_checkbox.checked:
                    line_mesh = VisUtils.create_line(xyzr[:, :3], line_set)
                    vis.plot(line_mesh, (0., 0., 0.), opacity=line_opacity_txt.double_value)

                if with_tube_checkbox.checked:
                    radius = xyzr[:, -1][0]
                    tube_mesh = VisUtils.create_tube(xyzr[:, :3], radius, line_set)
                    vis.plot(tube_mesh, color, opacity=tube_opacity_txt.double_value)

        if with_obstacle_checkbox.checked:
            obs_df = pd.read_csv(algo_json['grid_env']['obstacle_file'], index_col=0)
            mesh = VisUtils.create_point_cloud(obs_df[['x', 'y', 'z']].values)
            vis.plot(mesh, color=(0.3, 0.3, 0.3), opacity=1.0)

        vis.show()

    def _define_path_on_click(self):
        res_file = os.path.join(self.proj_dir, 'search_result.hdf5')
        if not os.path.exists(res_file):
            self.console_label.text = f"    [INFO]: no search file exist, please generate search result first."
            return

        smooth_file = os.path.join(self.proj_dir, 'smooth_setup.json')
        if not os.path.exists(smooth_file):
            smooth_json = {
                'path_list': [
                    {
                        'name': None,
                        'begin_name': None,
                        'end_name': None
                    }
                ],
                'conflict_setting': {
                    'obstacle_weight': 100.0,
                    'path_conflict_weight': 100.0
                },
                'setting': []
            }

            for _, res_cell in h5py.File(res_file).items():
                group_idx = res_cell['group_idx'][0]
                smooth_json['setting'].append({
                    'group_idx': int(group_idx),
                    'segment_degree': None,
                    'segments': {},
                    'paths_cost': {}
                })

            with open(smooth_file, 'w') as f:
                json.dump(smooth_json, f, indent=4)
        subprocess.run(f"gedit {smooth_file};", shell=True)

    def _setup_smooth_runner_on_click(self):
        algo_file = os.path.join(self.proj_dir, 'algorithm_setup.json')
        if not os.path.exists(algo_file):
            self.console_label.text = "    [ERROR]: No algorithm_setup.json in the project"
            return

        smooth_file = os.path.join(self.proj_dir, 'smooth_setup.json')
        if not os.path.exists(smooth_file):
            self.console_label.text = "    [ERROR]: No smooth_setup.json in the project"
            return

        res_file = os.path.join(self.proj_dir, 'search_result.hdf5')
        if not os.path.exists(res_file):
            self.console_label.text = f"    [INFO]: no search file exist, please generate search result first."
            return

        with open(smooth_file, 'r') as f:
            smooth_json = json.load(f)
        with open(algo_file, 'r') as f:
            algo_json = json.load(f)

        path_res = {}
        for _, res_cell in h5py.File(res_file).items():
            group_idx = int(res_cell['group_idx'][0])
            path_res[group_idx] = {}
            for name, xyzr in res_cell['path_result'].items():
                path_res[group_idx][name] = xyzr

        grid_cfg = algo_json['grid_env']
        pipe_cfg = algo_json['pipes']
        obs_xyzr = pd.read_csv(grid_cfg['obstacle_file'], index_col=0)[['x', 'y', 'z', 'radius']].values

        opt = PathOptimizer(grid_cfg, path_res, obs_xyzr, smooth_json['conflict_setting'])

        # ------ Step1: add path
        for path_info in smooth_json['path_list']:
            begin_name, end_name = path_info['begin_name'], path_info['end_name']
            if (begin_name is None) or (end_name is None):
                self.console_label.text = f"    [ERROR]: Non valid {begin_name} or {end_name}."
                return

            if pipe_cfg[begin_name]['group_idx'] != pipe_cfg[end_name]['group_idx']:
                self.console_label.text = f"    [ERROR]: {begin_name}'s group idx is not same as {end_name}."
                return

            group_idx = pipe_cfg[begin_name]['group_idx']
            opt.network_cells[group_idx].add_path(
                name=path_info['name'],
                begin_xyz=np.array(pipe_cfg[begin_name]['discrete_position']),
                begin_vec=np.array(pipe_cfg[begin_name]['direction']),
                end_xyz=np.array(pipe_cfg[end_name]['discrete_position']),
                end_vec=np.array(pipe_cfg[end_name]['direction'])
            )

        # ------ Step2: refit graph
        for setting in smooth_json['setting']:
            opt.network_cells[setting['group_idx']].refit_graph(setting['segment_degree'])

        # ------ Just for human edit
        for setting in smooth_json['setting']:
            group_idx = setting['group_idx']

            for seg_idx, seg_cell in opt.network_cells[group_idx].segments.items():
                if str(seg_idx) in setting['segments'].keys():
                    continue
                setting['segments'][seg_idx] = {
                    'bspline_degree': 3,
                    'bspline_num': 40,
                    'seg_idx': seg_idx,
                    'costs': [
                        {
                            'method': 'control_square_length_cost',
                            'type': 'auto',
                            'weight': 1.0
                        }
                    ],
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
                            'radius_weight': 1.0,
                            'cos_threshold': 0.95,
                            'cos_exponent': 1.5,
                            'cos_weight': 1.0
                        },
                        {
                            'method': 'connector_control_cos_cost',
                            'weight': 0.5
                        }
                    ])

        with open(smooth_file, 'w') as f:
            json.dump(smooth_json, f, indent=4)
        subprocess.run(f"gedit {smooth_file};", shell=True)

        with open(smooth_file, 'r') as f:
            smooth_json = json.load(f)

        # ------ Step3: update optimization info
        for setting in smooth_json['setting']:
            group_idx = setting['group_idx']

            for key in list(setting['segments'].keys()):
                cell = setting['segments'].pop(key)
                setting['segments'][cell['seg_idx']] = cell

            opt.network_cells[group_idx].update_optimization_info(
                segment_infos=setting['segments'],
                path_infos=setting['paths_cost']
            )
        self.smooth_opt = opt

    def _update_segments_on_click(self):
        if self.smooth_opt is None:
            self.console_label.text = f"    [ERROR]: smoother runner hasn't initiation"
            return

        for group_idx, net_cell in self.smooth_opt.network_cells.items():
            for seg_idx, seg_cell in net_cell.segments.items():
                xyzr = seg_cell.get_bspline_xyzr_np(net_cell.control_xyzr_np)
                mesh = FragmentOpen3d.create_line(xyzr[:, :3], color=seg_cell.color)

                self.add_point_cloud(
                    f"{group_idx}_seg{seg_idx}", mesh,
                    param={'type': 'path_line', 'shape': 'lineSet'}, is_visible=True,
                    material_type='line_material'
                )

    def _vis_spline_on_click(
            self, with_control_checkbox: gui.Checkbox, with_spline_checkbox: gui.Checkbox,
            with_obstacle_checkbox: gui.Checkbox
    ):
        algo_file = os.path.join(self.proj_dir, 'algorithm_setup.json')
        with open(algo_file, 'r') as f:
            algo_json = json.load(f)

        vis = VisUtils()
        for group_idx, net_cell in self.smooth_opt.network_cells.items():
            for seg_idx, cell in net_cell.segments.items():
                color = np.random.uniform(0.0, 1.0, size=(3,))
                if with_control_checkbox.checked:
                    control_pcd = net_cell.control_xyzr_np[cell.pcd_idxs, :3]
                    line_set = VisUtils.create_line_set(np.arange(0, cell.pcd_idxs.shape[0], 1))
                    mesh = VisUtils.create_line(control_pcd, line_set)
                    vis.plot(mesh, color=color, point_size=4)

                if with_spline_checkbox.checked:
                    xyzr = cell.get_bspline_xyzr_np(net_cell.control_xyzr_np)
                    radius = xyzr[:, -1][0]
                    line_set = VisUtils.create_line_set(np.arange(0, xyzr.shape[0], 1))
                    tube_mesh = VisUtils.create_tube(xyzr[:, :3], radius, line_set)
                    vis.plot(tube_mesh, color, opacity=1.0)

        if with_obstacle_checkbox.checked:
            obs_df = pd.read_csv(algo_json['grid_env']['obstacle_file'], index_col=0)
            mesh = VisUtils.create_point_cloud(obs_df[['x', 'y', 'z']].values)
            vis.plot(mesh, color=(0.3, 0.3, 0.3), opacity=1.0)

        vis.show()

    def _run_smooth(self, run_times_txt: gui.NumberEdit, lr_txt: gui.NumberEdit):
        if self.smooth_opt is None:
            self.console_label.text = f"    [ERROR]: smooth runner hasn't initiation"
            return

        # ------ restore current information
        self.smooth_tmp['revert'] = False
        self.smooth_tmp['last_step_data'].clear()
        for group_idx, net_cell in self.smooth_opt.network_cells.items():
            self.smooth_tmp['last_step_data'][group_idx] = net_cell.control_xyzr_np.copy()

        # ------ prepare tensor
        for group_idx, net_cell in self.smooth_opt.network_cells.items():
            net_cell.prepare_tensor()

        # ------ start running
        loss_res = self.smooth_opt.run(max_iter=int(run_times_txt.double_value), lr=lr_txt.double_value)
        plt.plot(loss_res)
        plt.show()

    def _revert_smooth(self):
        if self.smooth_tmp['revert']:
            self.console_label.text = f"    [ERROR]: already revert one time."
            return

        for group_idx, net_cell in self.smooth_opt.network_cells.items():
            net_cell.control_xyzr_np = self.smooth_tmp['last_step_data'][group_idx]

        self.smooth_tmp['revert'] = True
        self.console_label.text = f"    [INFO]: revert finish."

    def init_path_smooth_widget(self):
        smoother_layout = gui.CollapsableVert("Path Smooth Setup", self.spacing, self.blank_margins)
        self.panel.add_child(smoother_layout)

        with_line_checkbox = FragmentOpen3d.get_widget('checkbox', widget_info={'name': 'with_line'})
        with_line_checkbox.checked = True
        with_tube_checkbox = FragmentOpen3d.get_widget('checkbox', widget_info={'name': 'with_tube'})
        with_tube_checkbox.checked = True
        with_obstacle_checkbox = FragmentOpen3d.get_widget('checkbox', widget_info={'name': 'with_obstacle'})
        with_obstacle_checkbox.checked = False
        line_opacity_txt = FragmentOpen3d.get_widget('number', widget_info={'style': 'double', 'init_value': 1.0})
        tube_opacity_txt = FragmentOpen3d.get_widget('number', widget_info={'style': 'double', 'init_value': 0.6})
        vis_search_result_btn = FragmentOpen3d.get_widget('button', {'name': 'visualize search result'})
        vis_search_result_btn.set_on_clicked(partial(
            self._vis_search_result_on_click, with_line_checkbox=with_line_checkbox,
            with_tube_checkbox=with_tube_checkbox, with_obstacle_checkbox=with_obstacle_checkbox,
            line_opacity_txt=line_opacity_txt, tube_opacity_txt=tube_opacity_txt
        ))

        define_path_btn = FragmentOpen3d.get_widget('button', {'name': 'define path'})
        define_path_btn.set_on_clicked(self._define_path_on_click)

        setup_smooth_runner_btn = FragmentOpen3d.get_widget('button', {'name': 'setup smooth runner'})
        setup_smooth_runner_btn.set_on_clicked(self._setup_smooth_runner_on_click)

        update_path_visual_btn = FragmentOpen3d.get_widget('button', {'name': 'update path visualization'})
        update_path_visual_btn.set_on_clicked(self._update_segments_on_click)

        run_times_txt = FragmentOpen3d.get_widget(
            'number', {'style': 'double', 'decimal_precision': 0, 'init_value': 10}
        )
        lr_txt = FragmentOpen3d.get_widget(
            'number', {'style': 'double', 'decimal_precision': 6, 'init_value': 0.1, 'preferred_width': 80}
        )
        run_smooth_btn = FragmentOpen3d.get_widget('button', {'name': 'run smooth'})
        run_smooth_btn.set_on_clicked(partial(self._run_smooth, run_times_txt=run_times_txt, lr_txt=lr_txt))
        revert_smooth_btn = FragmentOpen3d.get_widget('button', {'name': 'revert smooth'})
        revert_smooth_btn.set_on_clicked(self._revert_smooth)

        with_control_checkbox = FragmentOpen3d.get_widget('checkbox', widget_info={'name': 'with_control'})
        with_control_checkbox.checked = True
        with_spline_checkbox = FragmentOpen3d.get_widget('checkbox', widget_info={'name': 'with_spline'})
        with_spline_checkbox.checked = True
        vis_spline_result_btn = FragmentOpen3d.get_widget('button', {'name': 'visualize spline result'})
        vis_spline_result_btn.set_on_clicked(partial(
            self._vis_spline_on_click,
            with_control_checkbox=with_control_checkbox, with_spline_checkbox=with_spline_checkbox,
            with_obstacle_checkbox=with_obstacle_checkbox
        ))

        # todo save result

        smoother_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', [
                FragmentOpen3d.get_layout_widget('horiz', [with_line_checkbox, with_tube_checkbox]),
                FragmentOpen3d.get_layout_widget('horiz', [
                    gui.Label('line_opacity:'), line_opacity_txt, gui.Label('tube_opacity:'), tube_opacity_txt
                ]),
                FragmentOpen3d.get_layout_widget('horiz', [with_control_checkbox, with_spline_checkbox]),
                with_obstacle_checkbox,
                vis_search_result_btn,
                vis_spline_result_btn,
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('step1:'), define_path_btn]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('step2:'), setup_smooth_runner_btn]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('check_path:'), update_path_visual_btn]),
                gui.Label('step3:'),
                FragmentOpen3d.get_layout_widget('horiz', [
                    gui.Label('run_times:'), run_times_txt, gui.Label('lr:'), lr_txt
                ], 4),
                FragmentOpen3d.get_layout_widget('horiz', [run_smooth_btn, revert_smooth_btn], 4),
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
    window = SmoothApp(config=config)

    app.run()

    sys.exit("CopyRight From Qu")


if __name__ == '__main__':
    main()
