import shutil
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

from scripts_py.version_9.mapf_pkg.app_utils import AppWindow
from scripts_py.version_9.mapf_pkg.app_utils import FragmentOpen3d
from scripts_py.version_9.mapf_pkg.shape_utils import ShapeUtils


class EnvironmentApp(AppWindow):
    MENU_SAVE_JSON = 2
    MENU_LOAD_JSON = 3

    def __init__(self, config):
        super().__init__(config=config)

        self.proj_dir = config['proj_dir']
        self.save_dir = os.path.join(self.proj_dir, 'env_ply')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.global_widget = {}
        self.init_operate_widget()

    def init_menu(self):
        super().init_menu()

        self.file_menu.add_item("SaveJson", EnvironmentApp.MENU_SAVE_JSON)
        self.file_menu.add_item("LoadJson", EnvironmentApp.MENU_LOAD_JSON)

        def save_json():
            dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose path to output", self.window.theme)
            dlg.add_filter(".json", "Json files (.json)")
            dlg.set_on_cancel(self.window.close_dialog)
            dlg.set_on_done(self.on_json_export)
            self.window.show_dialog(dlg)

        def load_json():
            dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose path to Load", self.window.theme)
            dlg.add_filter(".json", "Json files (.json)")
            dlg.set_on_cancel(self.window.close_dialog)
            dlg.set_on_done(self.on_json_load)
            self.window.show_dialog(dlg)

        self.window.set_on_menu_item_activated(EnvironmentApp.MENU_SAVE_JSON, save_json)
        self.window.set_on_menu_item_activated(EnvironmentApp.MENU_LOAD_JSON, load_json)

    def _create_box_btn_on_click(
            self,
            box_name_txt: gui.TextEdit, box_min_vec: gui.VectorEdit, box_max_vec: gui.VectorEdit,
            box_color_vec: gui.VectorEdit, is_box_solid: gui.Checkbox, box_reso_txt: gui.NumberEdit,
            console_label: gui.Label
    ):
        name = box_name_txt.text_value
        min_xyz = box_min_vec.vector_value
        max_xyz = box_max_vec.vector_value
        rgb = box_color_vec.vector_value
        is_solid = is_box_solid.checked
        reso = box_reso_txt.double_value

        if reso == 0.0:
            console_label.text = f"    [ERROR]: Non valid resolution"
            return

        if np.sum(rgb) <= 0:
            rgb = np.array([0.3, 0.3, 0.3])

        pcd = ShapeUtils.create_box_pcd(
            min_xyz[0], min_xyz[1], min_xyz[2], max_xyz[0], max_xyz[1], max_xyz[2], reso, is_solid
        )
        pcd_o3d = FragmentOpen3d.create_pcd(pcd, rgb)
        self.add_point_cloud(
            name, pcd_o3d, is_visible=True,
            param={
                'type': 'obstacle', 'shape': 'box', 'min_xyz': min_xyz.tolist(), 'max_xyz': max_xyz.tolist(),
                'reso': reso, 'rgb': rgb.tolist(), 'is_solid': is_solid
            }
        )
        self.adjust_center_camera()

    def _create_cylinder_btn_on_click(
            self,
            cylinder_name_txt: gui.TextEdit, cylinder_center_vec: gui.VectorEdit,
            cylinder_angles_vec: gui.VectorEdit, cylinder_radius_txt: gui.NumberEdit,
            cylinder_height_txt: gui.NumberEdit, cylinder_reso_txt: gui.NumberEdit,
            cylinder_rgb_vec: gui.VectorEdit, is_cylinder_solid: gui.Checkbox, console_label: gui.Label
    ):
        name = cylinder_name_txt.text_value
        xyz = cylinder_center_vec.vector_value
        angles = cylinder_angles_vec.vector_value
        height = cylinder_height_txt.double_value
        radius = cylinder_radius_txt.double_value
        reso = cylinder_reso_txt.double_value
        rgb = np.array(cylinder_rgb_vec.vector_value)
        is_solid = is_cylinder_solid.checked

        if reso == 0.0:
            console_label.text = f"    [ERROR]: Non valid resolution"
            return

        if np.sum(rgb) <= 0:
            rgb = np.array([0.3, 0.3, 0.3])

        pcd = ShapeUtils.create_cylinder_pcd(xyz, radius, height, angles, reso, is_solid)
        pcd_o3d = FragmentOpen3d.create_pcd(pcd, colors=rgb)
        self.add_point_cloud(
            name, pcd_o3d, is_visible=True,
            param={
                'type': 'obstacle', 'shape': 'cylinder',
                'center': xyz.tolist(), 'radius': radius, 'height': height, 'angles': angles.tolist(),
                'reso': reso, 'rgb': rgb.tolist(), 'is_solid': is_solid
            }
        )
        self.adjust_center_camera()

    def _import_obstacle_on_click(self, path: str, sample_num_of_points: gui.NumberEdit, console_label: gui.Label):
        name = os.path.basename(path)
        if path.endswith('.ply') or path.endswith('.obj'):
            self.add_point_cloud_from_file(
                name, path, is_visible=True,
                param={
                    'type': 'obstacle', 'shape': 'point_cloud_file', 'file': path
                }
            )
        else:
            if sample_num_of_points.int_value == 0:
                console_label.text = f"[ERROR]: Wrong sample_num_of_points:{sample_num_of_points}"
            else:
                mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_model(path)
                pcd_o3d = mesh.sample_points_poisson_disk(sample_num_of_points.int_value)
                self.add_point_cloud(
                    name, pcd_o3d, is_visible=True,
                    param={
                        'type': 'obstacle', 'shape': 'point_cloud_file', 'file': path
                    }
                )
        self.window.close_dialog()

    def _create_pipe_btn_on_click(
            self,
            pipe_name_txt: gui.TextEdit, pipe_center_vec: gui.VectorEdit, pipe_direction_combox: gui.Combobox,
            pipe_radius_txt: gui.NumberEdit, pipe_group_int: gui.NumberEdit, pipe_is_input: gui.Checkbox,
            pipe_rgb_vec: gui.VectorEdit, console_label: gui.Label
    ):
        name = pipe_name_txt.text_value
        center = np.array(pipe_center_vec.vector_value)
        direction = pipe_direction_combox.selected_text
        group_idx = pipe_group_int.int_value
        radius = pipe_radius_txt.double_value
        rgb = pipe_rgb_vec.vector_value
        if np.sum(rgb) <= 0:
            rgb = np.array([0.3, 0.3, 0.3])

        if radius == 0:
            console_label.text = "    [Error]: Non valid radius"
            return
        elif group_idx == -1:
            console_label.text = "    [Error]: Non valid group_idx"
            return

        if direction == '+x':
            direction_np = np.array([1., 0., 0.])
        elif direction == '+y':
            direction_np = np.array([0., 1., 0.])
        elif direction == '+z':
            direction_np = np.array([0., 0., 1.])
        elif direction == '-x':
            direction_np = np.array([-1., 0., 0.])
        elif direction == '-y':
            direction_np = np.array([0., -1., 0.])
        else:
            direction_np = np.array([0., 0., -1.])
        mesh_o3d = FragmentOpen3d.create_arrow(center, direction_np, radius, rgb)

        path = os.path.join(self.save_dir, f"{name}.ply")
        # o3d.io.write_triangle_mesh(path, mesh_o3d)
        # self.add_mesh_from_file(
        #     name, path, is_visible=True,
        #     param={
        #         'type': 'pipe',
        #         'position': center.tolist(), 'direction': direction, 'group_idx': group_idx,
        #         'radius': radius, 'is_input': pipe_is_input.checked,
        #         'rgb': rgb.tolist(), 'save_path': path
        #     }
        # )
        self.add_point_cloud(
            name, mesh_o3d, is_visible=True,
            param={
                'type': 'pipe',
                'position': center.tolist(), 'direction': direction_np.tolist(), 'group_idx': group_idx,
                'radius': radius, 'is_input': pipe_is_input.checked, 'rgb': rgb.tolist(), 'save_path': path
            }
        )

        self.adjust_center_camera()

    def init_operate_widget(self):
        operate_layout = gui.CollapsableVert("Environment Operation", self.spacing, self.blank_margins)
        self.panel.add_child(operate_layout)

        self.global_widget['grid_min'] = FragmentOpen3d.get_widget('vector', {})
        self.global_widget['grid_max'] = FragmentOpen3d.get_widget('vector', {})
        grid_block = FragmentOpen3d.get_layout_widget('vert', [
            FragmentOpen3d.get_layout_widget('horiz', [
                FragmentOpen3d.get_widget('label', {'name': 'grid min'}), self.global_widget['grid_min']
            ]),
            FragmentOpen3d.get_layout_widget('horiz', [
                FragmentOpen3d.get_widget('label', {'name': 'grid max'}), self.global_widget['grid_max']
            ])
        ], 1, self.vert_margins)
        operate_layout.add_child(grid_block)

        box_name_txt = FragmentOpen3d.get_widget('text', {})
        box_min_vec = FragmentOpen3d.get_widget('vector', {})
        box_max_vec = FragmentOpen3d.get_widget('vector', {})
        box_rgb_vec = FragmentOpen3d.get_widget('vector', {})
        is_box_solid = FragmentOpen3d.get_widget('checkbox', {'name': 'is Solid:'})
        box_reso_txt = FragmentOpen3d.get_widget('number', {'style': 'double'})
        box_create_btn = FragmentOpen3d.get_widget('button', {'name': 'create box obstacle'})
        box_create_btn.set_on_clicked(partial(
            self._create_box_btn_on_click,
            box_name_txt=box_name_txt, box_min_vec=box_min_vec, box_max_vec=box_max_vec,
            box_color_vec=box_rgb_vec, is_box_solid=is_box_solid, box_reso_txt=box_reso_txt,
            console_label=self.console_label
        ))
        operate_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', [
                gui.Label('create box obstacle:'),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('name:'), box_name_txt]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('min xyz:'), box_min_vec]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('max xyz:'), box_max_vec]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('rgb:'), box_rgb_vec]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('reso:'), box_reso_txt, is_box_solid], 4),
                box_create_btn
            ], 1, self.vert_margins)
        )

        cylinder_name_txt = FragmentOpen3d.get_widget('text', {})
        cylinder_center_vec = FragmentOpen3d.get_widget('vector', {})
        cylinder_angles_vec = FragmentOpen3d.get_widget('vector', {})
        cylinder_radius_txt = FragmentOpen3d.get_widget('number', {'style': 'double'})
        cylinder_height_txt = FragmentOpen3d.get_widget('number', {'style': 'double'})
        cylinder_reso_txt = FragmentOpen3d.get_widget('number', {'style': 'double'})
        cylinder_rgb_vec = FragmentOpen3d.get_widget('vector', {})
        is_cylinder_solid = FragmentOpen3d.get_widget('checkbox', {'name': 'is Solid:'})
        cylinder_create_btn = FragmentOpen3d.get_widget('button', {'name': 'create cylinder obstacle'})
        cylinder_create_btn.set_on_clicked(partial(
            self._create_cylinder_btn_on_click,
            cylinder_name_txt=cylinder_name_txt, cylinder_center_vec=cylinder_center_vec,
            cylinder_angles_vec=cylinder_angles_vec, cylinder_radius_txt=cylinder_radius_txt,
            cylinder_height_txt=cylinder_height_txt, cylinder_reso_txt=cylinder_reso_txt,
            cylinder_rgb_vec=cylinder_rgb_vec, is_cylinder_solid=is_cylinder_solid, console_label=self.console_label
        ))
        operate_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', [
                gui.Label('create cylinder obstacle:'),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('name:'), cylinder_name_txt]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('center:'), cylinder_center_vec]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('angles:'), cylinder_angles_vec]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('rgb:'), cylinder_rgb_vec]),
                FragmentOpen3d.get_layout_widget('horiz', [
                    gui.Label('radius:'), cylinder_radius_txt, gui.Label('height:'), cylinder_height_txt
                ], 4),
                FragmentOpen3d.get_layout_widget('horiz', [
                    gui.Label('reso:'), cylinder_reso_txt, is_cylinder_solid
                ], 4),
                cylinder_create_btn
            ], 1, self.vert_margins)
        )

        file_dlg_button = FragmentOpen3d.get_widget('button', {'name': 'import stl/ply/obj'})
        sample_num_of_points = FragmentOpen3d.get_widget('number', {'style': 'int', 'preferred_width': 120})

        dlg_done_fun = partial(
            self._import_obstacle_on_click,
            sample_num_of_points=sample_num_of_points,
            console_label=self.console_label
        )

        def _file_dlg_btn_on_click():
            filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file", self.window.theme)
            filedlg.add_filter(".obj .ply .stl", "Triangle mesh (.obj, .ply, .stl)")
            filedlg.set_on_cancel(self.window.close_dialog)
            filedlg.set_on_done(dlg_done_fun)
            self.window.show_dialog(filedlg)

        file_dlg_button.set_on_clicked(_file_dlg_btn_on_click)
        operate_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', [
                gui.Label('import obstacle:'),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('sample num:'), sample_num_of_points], 4),
                file_dlg_button
            ], 1, self.vert_margins)
        )

        pipe_name_txt = FragmentOpen3d.get_widget('text', {})
        pipe_center_vec = FragmentOpen3d.get_widget('vector', {})
        pipe_direction_combox = FragmentOpen3d.get_widget('combobox', {'items': ['+x', '+y', '+z', '-x', '-y', '-z']})
        pipe_radius_txt = FragmentOpen3d.get_widget('number', {'style': 'double'})
        pipe_group_int = FragmentOpen3d.get_widget('number', {'style': int, 'init_value': -1, 'preferred_width': 120})
        pipe_is_input = FragmentOpen3d.get_widget('checkbox', {'name': 'is input'})
        pipe_create_btn = FragmentOpen3d.get_widget('button', {'name': 'create pipe'})
        pipe_rgb_vec = FragmentOpen3d.get_widget('vector', {})
        pipe_create_btn.set_on_clicked(partial(
            self._create_pipe_btn_on_click,
            pipe_name_txt=pipe_name_txt,
            pipe_center_vec=pipe_center_vec,
            pipe_direction_combox=pipe_direction_combox,
            pipe_radius_txt=pipe_radius_txt,
            pipe_group_int=pipe_group_int,
            pipe_is_input=pipe_is_input,
            pipe_rgb_vec=pipe_rgb_vec,
            console_label=self.console_label,
        ))
        operate_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', [
                gui.Label('create pipe inlet/outlet:'),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('name:'), pipe_name_txt]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('center:'), pipe_center_vec]),
                FragmentOpen3d.get_layout_widget('horiz', [
                    gui.Label('direction:'), pipe_direction_combox, gui.Label('radius:'), pipe_radius_txt
                ], 4),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('rgb:'), pipe_rgb_vec]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label('group:'), pipe_group_int, pipe_is_input], 4),
                pipe_create_btn
            ], 1, self.vert_margins, with_stretch=True)
        )

    def on_json_export(self, filename: str):
        if not filename.endswith('.json'):
            self.console_label.text = f"[ERROR]: Non valid filename {filename}"
            return

        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.mkdir(self.save_dir)

        obs_cfgs, pipe_cfgs = [], []
        obs_dfs = []
        for name in self.geoMap.keys():
            clean_item = {}
            for key in self.geoMap[name].keys():
                if key == 'geometry':
                    continue
                clean_item[key] = self.geoMap[name][key]

            if clean_item['param']['type'] == 'obstacle' and clean_item['style'] == 'PointCloud':
                obs_cfgs.append(clean_item)
                pcd = np.asarray(self.geoMap[name]['geometry'].points)
                sub_obs_df = pd.DataFrame(data={'x': pcd[:, 0], 'y': pcd[:, 1], 'z': pcd[:, 2]})
                sub_obs_df['radius'] = 0.0
                sub_obs_df['tag'] = clean_item['name']
                obs_dfs.append(sub_obs_df)

            elif clean_item['param']['type'] == 'pipe':
                pipe_cfgs.append(clean_item)
                o3d.io.write_triangle_mesh(clean_item['param']['save_path'], self.geoMap[name]['geometry'])

            else:
                self.console_label.text = f"[Error]: Non Valid type {clean_item['param']['type']}"
                return

        if len(obs_dfs) > 0:
            obs_path = os.path.join(self.save_dir, 'obstacle.csv')
            obs_dfs = pd.concat(obs_dfs, axis=0, ignore_index=True)
            obs_dfs.to_csv(obs_path)

        save_cfg = {
            'proj_dir': self.proj_dir,
            'grid_min': self.global_widget['grid_min'].vector_value.tolist(),
            'grid_max': self.global_widget['grid_max'].vector_value.tolist(),
            'obs_cfgs': obs_cfgs,
            'pipe_cfgs': pipe_cfgs,
            'env_save_dir': self.save_dir
        }

        with open(filename, 'w') as f:
            json.dump(save_cfg, f, indent=4)

    def on_json_load(self, filename: str):
        with open(filename, 'r') as f:
            save_cfg = json.load(f)
        self.proj_dir = save_cfg['proj_dir']
        self.save_dir = save_cfg['env_save_dir']
        self.global_widget['grid_min'].vector_value = save_cfg['grid_min']
        self.global_widget['grid_max'].vector_value = save_cfg['grid_max']
        obs_df = pd.read_csv(os.path.join(self.save_dir, 'obstacle.csv'), index_col=0)

        for obs_cfg in save_cfg['obs_cfgs']:
            pcd = obs_df[obs_df['tag'] == obs_cfg['name']][['x', 'y', 'z']].values
            pcd_o3d = FragmentOpen3d.create_pcd(pcd, np.array(obs_cfg['param']['rgb']))
            self.add_point_cloud(
                obs_cfg['name'], pcd_o3d, is_visible=obs_cfg['is_visible'], param=obs_cfg['param']
            )

        for pipe_cfg in save_cfg['pipe_cfgs']:
            self.add_mesh_from_file(
                pipe_cfg['name'], pipe_cfg['param']['save_path'], is_visible=pipe_cfg['is_visible'],
                param=pipe_cfg['param']
            )

        self.adjust_center_camera()


def parse_args():
    parser = argparse.ArgumentParser(description="Fluid Tool")
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--proj_dir', type=str, default='/home/admin123456/Desktop/work/path_examples')
    args = parser.parse_args()
    return args


def main():
    # o3d.visualization.webrtc_server.enable_webrtc()

    app = gui.Application.instance
    app.initialize()

    args = parse_args()
    if not os.path.exists(args.proj_dir):
        os.mkdir(args.proj_dir)

    config = {
        'width': args.width, 'height': args.height, 'proj_dir': args.proj_dir
    }
    window = EnvironmentApp(config=config)

    app.run()

    sys.exit("CopyRight From Qu")


if __name__ == '__main__':
    main()
