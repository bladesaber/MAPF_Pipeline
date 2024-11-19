import numpy as np
import pandas as pd
import sys
import os
import shutil
import json
import open3d as o3d
import open3d.visualization.gui as gui
from functools import partial
import argparse
import subprocess
import h5py
import matplotlib.pyplot as plt

from scripts_py.version_9.mapf_pkg.app_path import PathApp
from scripts_py.version_9.mapf_pkg.app_utils import FragmentOpen3d
from scripts_py.version_9.mapf_pkg.smooth_optimizer import PathOptimizer
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils
from scripts_py.version_9.mapf_pkg.shape_utils import ShapeUtils


class SmoothApp(PathApp):
    def __init__(self, config):
        super().__init__(config=config)

        line_material = o3d.visualization.rendering.MaterialRecord()
        line_material.shader = 'unlitLine'
        line_material.line_width = 10
        self.materials.update({'line_material': line_material})

        self.init_path_smooth_widget()
        self.smooth_opt: PathOptimizer = None
        self.smooth_tmp = {'revert_count': 0, 'revert_data': []}

    def push_data(self):
        if self.smooth_opt is None:
            return
        save_term = {}
        for group_idx, group_cell in self.smooth_opt.group_cells.items():
            save_term[group_idx] = group_cell.c_table[['x', 'y', 'z', 'radius']].values
        self.smooth_tmp['revert_data'].append(save_term)
        self.smooth_tmp['revert_data'] = self.smooth_tmp['revert_data'][-10:]
        self.smooth_tmp['revert_count'] += 1

    def pop_data(self):
        if (self.smooth_opt is None) or (self.smooth_tmp['revert_count'] <= 0):
            return
        save_term = self.smooth_tmp['revert_data'].pop()
        for group_idx, xyzr in save_term.items():
            self.smooth_opt.group_cells[group_idx].c_table[['x', 'y', 'z', 'radius']] = xyzr
        self.smooth_tmp['revert_count'] -= 1

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
                    vis.plot(line_mesh, color, opacity=line_opacity_txt.double_value)

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
        algo_file = os.path.join(self.proj_dir, 'algorithm_setup.json')
        if not os.path.exists(algo_file):
            self.console_label.text = "    [ERROR]: No algorithm_setup.json in the project"
            return

        search_file = os.path.join(self.proj_dir, 'search_result.hdf5')
        if not os.path.exists(search_file):
            self.console_label.text = f"    [INFO]: no search file exist, please generate search result first."
            return

        # ------ adding path information
        with open(algo_file, 'r') as f:
            algo_json = json.load(f)

        pipe_cfg = algo_json['pipes']
        for pipe_name in pipe_cfg.keys():
            pipe_cfg[pipe_name].update({'radius_scale': 1.0})

        smooth_file = os.path.join(self.proj_dir, 'smooth_setup.json')
        if not os.path.exists(smooth_file):
            smooth_json = {
                'cfg': {
                    'path_result_hdf5': search_file,
                    'orig_obstacle_file': algo_json['grid_env']['obstacle_file'],
                    'pipe_cfg': pipe_cfg,
                },
                'conflict_setting': {
                    'obstacle_conflict_weight': 1000.0,
                    'path_conflict_weight': 100.0,
                    'self_conflict_weight': 100.0
                },
                'setting': [],
                'auto_run_cfg': {
                    'lr': 0.0025,
                    'factor': 0.25,
                    'patience': 500,
                    'cooldown': 100,
                    'min_lr': 0.001,
                    'max_iter': 20000,
                    'outer_loop': 10
                }
            }

            # segment_degree_min尽量与segment_degree_max相近
            for _, res_cell in h5py.File(search_file).items():
                group_idx = res_cell['group_idx'][0]
                smooth_json['setting'].append({
                    'group_idx': int(group_idx),
                    'aid_path_info': [
                        {
                            'name': None,
                            'from_name': None,
                            'to_name': None
                        }
                    ],
                    'graph_cfg': {
                        'seg_node_min': 15,
                        'seg_node_max': None,
                    },
                    'segments': [],
                    'connectors': []
                })
            with open(smooth_file, 'w') as f:
                json.dump(smooth_json, f, indent=4)
        subprocess.run(f"gedit {smooth_file};", shell=True)

    def _create_smooth_obstacle_on_click(self):
        smooth_file = os.path.join(self.proj_dir, 'smooth_setup.json')
        if not os.path.exists(smooth_file):
            self.console_label.text = "    [ERROR]: No smooth_setup.json in the project"
            return

        with open(smooth_file, 'r') as f:
            smooth_json = json.load(f)

        pipe_cfg = smooth_json['cfg']['pipe_cfg']
        obs_xyz = pd.read_csv(
            smooth_json['cfg']['orig_obstacle_file'], index_col=0
        )[['x', 'y', 'z', 'radius']].values[:, :3]
        obs_xyz_list = [obs_xyz]

        # ------ create shell
        min_reso = np.inf
        for pipe_name in pipe_cfg.keys():
            min_reso = min(pipe_cfg[pipe_name]['radius'] * pipe_cfg[pipe_name]['radius_scale'], min_reso)

        for pipe_name in pipe_cfg.keys():
            shell_pcd = ShapeUtils.create_sphere_pcd(
                center=pipe_cfg[pipe_name]['discrete_position'],
                vector=pipe_cfg[pipe_name]['direction'],
                radius=pipe_cfg[pipe_name]['radius'] * pipe_cfg[pipe_name]['radius_scale'] * 1.015,
                reso=min_reso * 0.3
                # reso=0.3
            )
            obs_xyz_list.append(shell_pcd)

        obs_xyz = np.concatenate(obs_xyz_list, axis=0)

        # ------ remove inner pcd
        for pipe_name in pipe_cfg.keys():
            pipe_info = pipe_cfg[pipe_name]
            obs_xyz = ShapeUtils.remove_obstacle_point_in_pipe(
                obs_pcd=obs_xyz,
                center=pipe_info['discrete_position'],
                direction=pipe_info['direction'],
                radius=pipe_info['radius'] * pipe_info['radius_scale'],
                tol_scale=1.025,
                is_input=pipe_info['is_input']
            )

        obs_df = pd.DataFrame(obs_xyz, columns=['x', 'y', 'z'])
        obs_df['radius'] = 0.0
        smooth_obstacle_file = os.path.join(self.proj_dir, 'smooth_obstacle.csv')
        obs_df.to_csv(smooth_obstacle_file)
        smooth_json['cfg'].update({'smooth_obstacle_file': smooth_obstacle_file})

        with open(smooth_file, 'w') as f:
            json.dump(smooth_json, f, indent=4)

        pcd_o3d = FragmentOpen3d.create_pcd(obs_xyz)
        self.add_point_cloud(
            'smooth_obstacle', pcd_o3d, is_visible=False,
            param={'type': 'obstacle', 'shape': 'grid_obstacle', 'rgb': [0.2, 0.4, 0.6]}, no_save=True
        )

    def _setup_smooth_runner_on_click(self):
        smooth_file = os.path.join(self.proj_dir, 'smooth_setup.json')
        if not os.path.exists(smooth_file):
            self.console_label.text = "    [ERROR]: No smooth_setup.json in the project"
            return

        with open(smooth_file, 'r') as f:
            smooth_json = json.load(f)

        # ------ Step 1 load search result
        pipes_cfg = smooth_json['cfg']['pipe_cfg']
        obs_xyzr = pd.read_csv(
            smooth_json['cfg']['smooth_obstacle_file'], index_col=0
        )[['x', 'y', 'z', 'radius']].values

        res_data = h5py.File(smooth_json['cfg']['path_result_hdf5'])

        init_cfg = {}
        for group_setting in smooth_json['setting']:
            group_idx = group_setting['group_idx']
            init_cfg[group_idx] = {
                'path_dict': {},
                'assemble_cfg': {
                    'seg_node_min': group_setting['graph_cfg']['seg_node_min'],
                    'seg_node_max': group_setting['graph_cfg']['seg_node_max'],
                },
                'aid_path_info': [],
            }
            for info in group_setting['aid_path_info']:
                info_copy = info.copy()
                info_copy.update({
                    'from_position': pipes_cfg[info['from_name']]['discrete_position'],
                    'from_vector': pipes_cfg[info['from_name']]['direction'],
                    'from_is_input': pipes_cfg[info['from_name']]['is_input'],
                    'to_position': pipes_cfg[info['to_name']]['discrete_position'],
                    'to_vector': pipes_cfg[info['to_name']]['direction'],
                    'to_is_input': pipes_cfg[info['to_name']]['is_input'],
                })
                init_cfg[group_idx]['aid_path_info'].append(info_copy)
            for name, xyzr in res_data[str(group_idx)]['path_result'].items():
                init_cfg[group_idx]['path_dict'][name] = np.array(xyzr)

        # ------ Step 2: add path
        opt = PathOptimizer()
        opt.init_cfg(init_cfg, obs_xyzr, smooth_json['conflict_setting'])

        # # ------ debug for visualization
        # for group_idx, group_cell in opt.group_cells.items():
        #     # group_cell.draw_control_points(with_seg=True, with_connector=False, with_overlap=False, with_fix=False)
        #     group_cell.draw_control_points(with_seg=True, with_connector=False, with_overlap=True, with_fix=True)
        # # ------

        # ------ Just for human edit
        # todo be careful bspline_density is very important
        for group_cfg in smooth_json['setting']:
            group_idx = group_cfg['group_idx']

            if len(group_cfg['segments']) == 0:
                for seg_idx, seg_cell in opt.group_cells[group_idx].seg_dict.items():
                    group_cfg['segments'].append({
                        'seg_idx': seg_idx,
                        'bspline_degree': 3,
                        'bspline_density': 0.25,
                        'left_overlap_ratio': 0.25,
                        'right_overlap_ratio': 0.25,
                        'left_intercept_ratio': 0.5,
                        'right_intercept_ratio': 0.5,
                        'segment_scale': 1.0,
                        'with_analytical_curvature_computed': True,
                        'cost_setting': {
                            'path_thickness': 0.01,
                            'obstacle_thickness': 0.01,
                            'cost_cfgs': [
                                {
                                    'cost_type': 'length',
                                    'data_type': 'control_points',
                                    'method': 'square_sum',
                                    'weight': 1.0
                                },
                                {
                                    'cost_type': 'length_distribution',
                                    'data_type': 'control_points',
                                    'method': 'length_variance',
                                    'weight': 100.0
                                },
                                {
                                    "cost_type": "curvature",
                                    "curvature_method": "analytical_curvature",
                                    "method": "sigmoid",
                                    "curvature_scale": 3.0,
                                    "weight": 2.5
                                },
                                {
                                    "cost_type": "curvature_distribution",
                                    "curvature_method": "analytical_curvature",
                                    "method": "value_variance",
                                    "weight": 1.0
                                }
                            ]
                        }
                    })

            if len(group_cfg['connectors']) == 0:
                for con_idx, con_cell in opt.group_cells[group_idx].connector_dict.items():
                    group_cfg['connectors'].append({
                        'con_idx': con_idx,
                        'bspline_degree': 3,
                        'bspline_density': 0.2,
                        'left_seg_idx': con_cell.left_seg_idx,
                        'right_seg_idx': con_cell.right_seg_idx,
                        'with_analytical_curvature_computed': True,
                        'cost_setting': {
                            'path_thickness': 0.01,
                            'obstacle_thickness': 0.01,
                            'cost_cfgs': [
                                {
                                    "cost_type": "curvature",
                                    "curvature_method": "analytical_curvature",
                                    "method": "sigmoid",
                                    "curvature_scale": 3.0,
                                    "weight": 2.5
                                },
                                {
                                    "cost_type": "curvature_distribution",
                                    "curvature_method": "analytical_curvature",
                                    "method": "value_variance",
                                    "weight": 1.0
                                }
                            ]
                        }
                    })

        with open(smooth_file, 'w') as f:
            json.dump(smooth_json, f, indent=4)
        subprocess.run(f"gedit {smooth_file};", shell=True)

        with open(smooth_file, 'r') as f:
            smooth_json = json.load(f)

        # ------ Step3: update optimization info
        for group_cfg in smooth_json['setting']:
            group_idx = group_cfg['group_idx']
            opt.init_optimize_cfg(group_idx, {
                'segments': group_cfg['segments'],
                'connectors': group_cfg['connectors']
            })
        self.smooth_opt = opt

        # ------ Step4: prepare spline mat
        for group_idx in opt.group_cells.keys():
            opt.update_spline_NpCfg(group_idx)

        # # ------ debug for visualization
        # opt.prepare_tensor()
        # for group_idx, group_cell in opt.group_cells.items():
        #     # group_cell.draw_control_points(with_seg=True, with_connector=True, with_overlap=False, with_fix=False)
        #     # group_cell.draw_spline(is_tube=False)
        #     group_cell.draw_tensor()
        # # ------

    def _update_cost_cfg(self, path: str):
        with open(path, 'r') as f:
            smooth_json = json.load(f)

        # ------ aggressive update
        # for group_cfg in smooth_json['setting']:
        #     group_idx = group_cfg['group_idx']
        #     self.smooth_opt.init_optimize_cfg(group_idx, {
        #         'segments': group_cfg['segments'],
        #         'connectors': group_cfg['connectors']
        #     })
        # for group_idx in self.smooth_opt.group_cells.keys():
        #     self.smooth_opt.update_spline_NpCfg(group_idx)

        # ------ lazy update
        for group_setting in smooth_json['setting']:
            group_idx = group_setting['group_idx']
            group_cell = self.smooth_opt.group_cells[group_idx]

            for seg_cfg in group_setting['segments']:
                group_cell.seg_dict[seg_cfg['seg_idx']].update_cost_cfg(seg_cfg['cost_setting'])
            for con_cfg in group_setting['connectors']:
                group_cell.connector_dict[con_cfg['con_idx']].update_cost_cfg(con_cfg['cost_setting'])

        self.smooth_opt.conflict_cfg = smooth_json['conflict_setting']

    def _update_segments_on_click(self):
        if self.smooth_opt is None:
            self.console_label.text = f"    [ERROR]: smoother runner hasn't initiation"
            return

        for group_idx, group_cell in self.smooth_opt.group_cells.items():
            table_xyzr = group_cell.c_table[['x', 'y', 'z', 'radius']].values
            color = np.random.uniform(0.0, 1.0, size=3)
            for seg_idx, seg_cell in group_cell.seg_dict.items():
                xyzr = seg_cell.compute_spline_np(table_xyzr)
                mesh = FragmentOpen3d.create_line(xyzr[:, :3], color=color)
                self.add_point_cloud(
                    f"{group_idx}/seg_{seg_idx}", mesh,
                    param={'type': 'path_line', 'shape': 'lineSet'}, is_visible=True,
                    material_type='line_material'
                )

            for con_idx, con_cell in group_cell.connector_dict.items():
                xyzr = con_cell.compute_spline_np(table_xyzr)
                mesh = FragmentOpen3d.create_line(xyzr[:, :3], color=(0.0, 1.0, 0.0))
                self.add_point_cloud(
                    f"{group_idx}/con_{con_idx}", mesh,
                    param={'type': 'path_line', 'shape': 'lineSet'}, is_visible=True,
                    material_type='line_material'
                )

    def _vis_spline_on_click(
            self, with_control_checkbox: gui.Checkbox, with_spline_checkbox: gui.Checkbox,
            with_obstacle_checkbox: gui.Checkbox, with_tube_checkbox: gui.Checkbox
    ):
        smooth_file = os.path.join(self.proj_dir, 'smooth_setup.json')
        with open(smooth_file, 'r') as f:
            smooth_json = json.load(f)

        vis = VisUtils()
        for group_idx, group_cell in self.smooth_opt.group_cells.items():
            table_xyzr = group_cell.c_table[['x', 'y', 'z', 'radius']].values
            group_color = np.random.uniform(0.0, 1.0, size=(3,))

            if with_control_checkbox.checked:
                for seg_idx, seg_cell in group_cell.seg_dict.items():
                    control_xyz = table_xyzr[seg_cell.c_idxs, :3]
                    line_set = VisUtils.create_line_set(np.arange(0, control_xyz.shape[0], 1))
                    mesh = VisUtils.create_line(control_xyz, line_set)
                    vis.plot(mesh, color=group_color, point_size=4)

            for seg_idx, seg_cell in group_cell.seg_dict.items():
                unique_color = np.random.uniform(0.0, 1.0, size=(3,))
                spline_xyzr = seg_cell.compute_spline_np(table_xyzr)
                line_set = VisUtils.create_line_set(np.arange(0, spline_xyzr.shape[0], 1))

                if with_spline_checkbox.checked:
                    line_mesh = VisUtils.create_line(spline_xyzr[:, :3], line_set)
                    vis.plot(line_mesh, unique_color, opacity=1.0)

                if with_tube_checkbox.checked:
                    tube_mesh = VisUtils.create_deformation_tube(
                        spline_xyzr[:, :3], line_set, radius=None, scalars=spline_xyzr[:, -1]
                    )
                    vis.plot(tube_mesh, unique_color, opacity=1.0)

            for con_idx, con_cell in group_cell.connector_dict.items():
                spline_xyzr = con_cell.compute_spline_np(table_xyzr)
                line_set = VisUtils.create_line_set(np.arange(0, spline_xyzr.shape[0], 1))

                if with_spline_checkbox.checked:
                    line_mesh = VisUtils.create_line(spline_xyzr[:, :3], line_set)
                    vis.plot(line_mesh, (0.0, 1.0, 0.0), opacity=1.0)

                if with_tube_checkbox.checked:
                    tube_mesh = VisUtils.create_deformation_tube(
                        spline_xyzr[:, :3], line_set, radius=None, scalars=spline_xyzr[:, -1]
                    )
                    vis.plot(tube_mesh, (0.0, 1.0, 0.0), opacity=1.0)

        if with_obstacle_checkbox.checked:
            obs_df = pd.read_csv(smooth_json['cfg']['smooth_obstacle_file'], index_col=0)
            mesh = VisUtils.create_point_cloud(obs_df[['x', 'y', 'z']].values)
            vis.plot(mesh, color=(0.3, 0.3, 0.3), opacity=1.0)

        vis.show()

    def _run_smooth(self, run_times_txt: gui.NumberEdit, lr_txt: gui.NumberEdit):
        if self.smooth_opt is None:
            self.console_label.text = f"    [ERROR]: smooth runner hasn't initiation"
            return

        # ------ restore current information
        self.push_data()

        # ------ prepare tensor
        self.smooth_opt.prepare_tensor()

        # # ------ visualization debug
        # self.smooth_opt.draw_constraints(
        #     lr_txt.double_value, with_control=False, with_spline=True,
        #     with_path_conflict=True, with_obstacle_conflict=True, with_obstacle=False,
        #     with_self_conflict=False
        # )
        # # ------

        # ------ start running
        loss_record, loss_info = self.smooth_opt.run_by_hand(
            max_iter=int(run_times_txt.double_value), lr=lr_txt.double_value
        )

        plt.title('loss record')
        plt.plot(loss_record)
        plt.show()

    def _auto_smooth(self):
        if self.smooth_opt is None:
            self.console_label.text = f"    [ERROR]: smooth runner hasn't initiation"
            return

        smooth_file = os.path.join(self.proj_dir, 'smooth_setup.json')
        with open(smooth_file, 'r') as f:
            smooth_json = json.load(f)

        # ------ restore current information
        self.push_data()

        # ------ prepare tensor
        self.smooth_opt.prepare_tensor()

        # ------ start running
        loss_record, loss_info = self.smooth_opt.auto_run(smooth_json['auto_run_cfg'])

        plt.title('loss record')
        plt.plot(loss_record)
        plt.show()

    def _revert_smooth(self):
        self.pop_data()

    def _save_result(self):
        if self.smooth_opt is None:
            return

        save_file = os.path.join(self.proj_dir, 'smooth_result.hdf5')
        self.smooth_opt.save(file=save_file)

    def loss_check_on_click(self, lr_txt: gui.NumberEdit):
        if self.smooth_opt is None:
            return

        self.smooth_opt.prepare_tensor()
        loss_info = self.smooth_opt.loss_check(lr_txt.double_value)
        for name, value in loss_info.items():
            print(f"{name}: {value}")

        plt.figure()
        plt.title('loss fragment')
        plt.barh(list(loss_info.keys()), loss_info.values())
        for i, (key, value) in enumerate(loss_info.items()):
            plt.text(value, i, f"{value:.3f}")
        plt.show()

        self.smooth_opt.draw_constraints(
            lr_txt.double_value, with_path_conflict=True, with_obstacle_conflict=True, with_real_conflict=True
        )

    def _draw_curvature_on_click(self):
        if self.smooth_opt is None:
            return
        self.smooth_opt.draw_curvature()

    def init_path_smooth_widget(self):
        smoother_layout = gui.CollapsableVert("Path Smooth Setup", self.spacing, self.blank_margins)
        self.panel.add_child(smoother_layout)

        with_line_checkbox = FragmentOpen3d.get_widget('checkbox', widget_info={'name': 'with_line'})
        with_line_checkbox.checked = True
        with_tube_checkbox = FragmentOpen3d.get_widget('checkbox', widget_info={'name': 'with_tube'})
        with_tube_checkbox.checked = True
        with_obstacle_checkbox = FragmentOpen3d.get_widget('checkbox', widget_info={'name': 'obstacle'})
        with_obstacle_checkbox.checked = False
        line_opacity_txt = FragmentOpen3d.get_widget('number', widget_info={'style': 'double', 'init_value': 1.0})
        tube_opacity_txt = FragmentOpen3d.get_widget('number', widget_info={'style': 'double', 'init_value': 0.6})
        vis_search_result_btn = FragmentOpen3d.get_widget('button', {'name': 'visSearchResult'})
        vis_search_result_btn.set_on_clicked(partial(
            self._vis_search_result_on_click, with_line_checkbox=with_line_checkbox,
            with_tube_checkbox=with_tube_checkbox, with_obstacle_checkbox=with_obstacle_checkbox,
            line_opacity_txt=line_opacity_txt, tube_opacity_txt=tube_opacity_txt
        ))

        define_path_btn = FragmentOpen3d.get_widget('button', {'name': 'define path'})
        define_path_btn.set_on_clicked(self._define_path_on_click)

        setup_smooth_runner_btn = FragmentOpen3d.get_widget('button', {'name': 'setup smooth runner'})
        setup_smooth_runner_btn.set_on_clicked(self._setup_smooth_runner_on_click)

        update_path_visual_btn = FragmentOpen3d.get_widget('button', {'name': 'visPath'})
        update_path_visual_btn.set_on_clicked(self._update_segments_on_click)

        run_times_txt = FragmentOpen3d.get_widget(
            'number', {'style': 'double', 'decimal_precision': 0, 'init_value': 10}
        )
        lr_txt = FragmentOpen3d.get_widget(
            'number', {'style': 'double', 'decimal_precision': 6, 'init_value': 0.0025, 'preferred_width': 80}
        )
        run_smooth_btn = FragmentOpen3d.get_widget('button', {'name': 'run smooth'})
        run_smooth_btn.set_on_clicked(partial(self._run_smooth, run_times_txt=run_times_txt, lr_txt=lr_txt))
        revert_result_btn = FragmentOpen3d.get_widget('button', {'name': 'revert_result'})
        revert_result_btn.set_on_clicked(self._revert_smooth)
        auto_smooth_btn = FragmentOpen3d.get_widget('button', {'name': 'auto smooth'})
        auto_smooth_btn.set_on_clicked(self._auto_smooth)
        save_result_btn = FragmentOpen3d.get_widget('button', {'name': 'save_result'})
        save_result_btn.set_on_clicked(self._save_result)
        update_control_data_btn = FragmentOpen3d.get_widget('button', {'name': 'adjustControlPcd'})  # todo non-complete

        with_control_checkbox = FragmentOpen3d.get_widget('checkbox', widget_info={'name': 'control'})
        with_control_checkbox.checked = True
        with_spline_checkbox = FragmentOpen3d.get_widget('checkbox', widget_info={'name': 'spline'})
        with_spline_checkbox.checked = True
        vis_spline_result_btn = FragmentOpen3d.get_widget('button', {'name': 'visSplineResult'})
        vis_spline_result_btn.set_on_clicked(partial(
            self._vis_spline_on_click,
            with_control_checkbox=with_control_checkbox, with_spline_checkbox=with_spline_checkbox,
            with_obstacle_checkbox=with_obstacle_checkbox, with_tube_checkbox=with_tube_checkbox
        ))

        loss_check_btn = FragmentOpen3d.get_widget('button', {'name': 'loss_check'})
        loss_check_btn.set_on_clicked(partial(self.loss_check_on_click, lr_txt=lr_txt))
        init_obstacle_btn = FragmentOpen3d.get_widget('button', {'name': 'obstacle init'})
        init_obstacle_btn.set_on_clicked(self._create_smooth_obstacle_on_click)

        # ------
        def _update_cost_cfg_btn_on_click():
            filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file", self.window.theme)
            filedlg.add_filter(".json", "Config File (.json)")
            filedlg.set_on_cancel(self.window.close_dialog)
            filedlg.set_on_done(self._update_cost_cfg)
            self.window.show_dialog(filedlg)

        update_cost_cfg_btn = FragmentOpen3d.get_widget('button', {'name': 'updateCfg'})
        update_cost_cfg_btn.set_on_clicked(_update_cost_cfg_btn_on_click)
        vis_curvature_btn = FragmentOpen3d.get_widget('button', {'name': 'visCurvature'})
        vis_curvature_btn.set_on_clicked(self._draw_curvature_on_click)
        debug_btn = FragmentOpen3d.get_widget('button', {'name': 'debug'})
        # debug_btn.set_on_clicked()

        smoother_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', [
                FragmentOpen3d.get_layout_widget('horiz', [with_line_checkbox, with_tube_checkbox]),
                FragmentOpen3d.get_layout_widget('horiz', [
                    with_control_checkbox, with_spline_checkbox, with_obstacle_checkbox
                ]),
                FragmentOpen3d.get_layout_widget('horiz', [
                    gui.Label('line_opacity:'), line_opacity_txt, gui.Label('tube_opacity:'), tube_opacity_txt
                ]),
                FragmentOpen3d.get_layout_widget('horiz', [vis_search_result_btn, vis_spline_result_btn], space=10),
                FragmentOpen3d.get_layout_widget('horiz', [
                    gui.Label('step1:'), define_path_btn, init_obstacle_btn
                ], space=10),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('step2:'), setup_smooth_runner_btn]),
                FragmentOpen3d.get_layout_widget('horiz', [
                    gui.Label('Tool:'), update_path_visual_btn, update_cost_cfg_btn
                ], space=10),
                FragmentOpen3d.get_layout_widget('horiz', [
                    gui.Label('step3:'), gui.Label('iter:'), run_times_txt, gui.Label('lr:'), lr_txt
                ]),
                FragmentOpen3d.get_layout_widget('horiz', [update_control_data_btn, vis_curvature_btn], 4),
                FragmentOpen3d.get_layout_widget('horiz', [run_smooth_btn, auto_smooth_btn], 4),
                FragmentOpen3d.get_layout_widget('horiz', [save_result_btn], 4),
                FragmentOpen3d.get_layout_widget('horiz', [loss_check_btn, revert_result_btn], 4),
                FragmentOpen3d.get_layout_widget('horiz', [debug_btn], 4),
            ], 3, self.vert_margins)
        )

    @staticmethod
    def hdf52csv(proj_dir):
        data = h5py.File(os.path.join(proj_dir, 'smooth_result.hdf5'))
        save_dir = os.path.join(proj_dir, 'excel')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for group_idx in data.keys():
            group_data = data[group_idx]
            spline_data = group_data['spline']

            group_dir = os.path.join(save_dir, f'group_{group_idx}')
            if os.path.exists(group_dir):
                shutil.rmtree(group_dir)
            os.mkdir(group_dir)

            for name in spline_data:
                path = np.array(spline_data[name])
                path_df = pd.DataFrame(path, columns=['x', 'y', 'z', 'radius'])
                path_df.to_csv(os.path.join(group_dir, f"{name}.txt"), header=False, index=False)


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
