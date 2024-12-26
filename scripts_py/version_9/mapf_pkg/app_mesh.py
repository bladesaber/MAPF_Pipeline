import shutil
import numpy as np
import sys
import os
import json
import open3d.visualization.gui as gui
import argparse
import subprocess
import h5py
import pyvista
import matplotlib.pyplot as plt
from functools import partial

from scripts_py.version_9.mapf_pkg.app_utils import FragmentOpen3d
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils
from scripts_py.version_9.mapf_pkg.app_smooth import SmoothApp
from scripts_py.version_9.mapf_pkg.point_cloud_utils import PathSurfaceGenerator, Pcd2MeshUtils
from scripts_py.version_9.mapf_pkg.postprocess_utils import GroupRadiusAdjuster, RadiusOptimizer, StepExporter


class MeshApp(SmoothApp):
    def __init__(self, config):
        super().__init__(config=config)
        self.init_path_mesh_widget()
        self.radius_opt: RadiusOptimizer = None

    def _export_point_cloud_on_click(self):
        res_file = os.path.join(self.proj_dir, 'radius_adjust_result.hdf5')
        if not os.path.exists(res_file):
            self.console_label.text = f"    [ERROR]: finial result isn't exist."
            return

        mesh_dir = os.path.join(self.proj_dir, 'mesh')
        if os.path.exists(mesh_dir):
            shutil.rmtree(mesh_dir)
        os.mkdir(mesh_dir)

        for _, group_cell in h5py.File(res_file).items():
            group_idx = int(group_cell['group_idx'][0])

            cells = []
            for name, xyzr in group_cell['segment'].items():
                cells.append((name, xyzr))
            for name, xyzr in group_cell['connector'].items():
                cells.append((name, xyzr))

            group_gen = {}
            for seg_name, xyzr in cells:
                xyzr = np.array(xyzr)
                generator = PathSurfaceGenerator(xyzr[:, :3], xyzr[:, -1])
                generator.generate_pcd({
                    'length_reso': 0.2,  # line segment resolution
                    'sphere_reso': 0.15,  # sphere resolution
                    'relax_factor': 0.01,  # remove resolution,
                    'deplicate_format': 3,
                    'left_clamp_reso': 0.2,
                    'right_clamp_reso': 0.2,
                    'left_clamp_relax_factor': 0.01,
                    'right_clamp_relax_factor': 0.01,
                    'with_left_clamp': False,
                    'with_right_clamp': False
                })
                group_gen[seg_name] = generator

            for name_i, gen_i in group_gen.items():
                for name_j, gen_j in group_gen.items():
                    gen_i.pcd_data = PathSurfaceGenerator.remove_inner_sphere(
                        gen_i.pcd_data, gen_j.sphere_centers, gen_j.sphere_radius, relax_factor=1e-4,
                    )

            group_pcds = []
            for name, gen in group_gen.items():
                group_pcds.append(gen.merge_pcd())
            group_pcds = np.concatenate(group_pcds, axis=0)

            pcd_ply = pyvista.PolyData(group_pcds)
            # pcd_ply.plot()
            Pcd2MeshUtils.save_ply(pcd_ply, file=os.path.join(mesh_dir, f"group_{group_idx}.ply"))

    def _setup_radius_adjuster_on_click(
            self,
            is_segment_line_checkbox: gui.Checkbox,
            is_connector_line_checkbox: gui.Checkbox,
    ):
        res_file = os.path.join(self.proj_dir, 'smooth_result.hdf5')
        if not os.path.exists(res_file):
            self.console_label.text = f"    [ERROR]: smooth result isn't exist."
            return

        postprocess_file = os.path.join(self.proj_dir, 'postprocess.json')
        if not os.path.exists(postprocess_file):
            postprocess_json = {
                "run_cfg": {
                    "lr": 0.01,
                    "factor": 0.5,
                    "patience": 1000,
                    "cooldown": 100,
                    "min_lr": 0.001,
                    "max_iter": 30000
                },
                "cost_cfg": {
                    "bound_weight": 1000.0,
                    "scale_weight": 100.0,
                    "balance_weight": 1.0,
                    "keep_weight": 1.0
                },
                'setting': {}
            }

            for group_name, group_layer in h5py.File(res_file).items():
                select_ratio_dict = {}
                for seg_name, seg_cell_layer in group_layer['segment'].items():
                    num_ts = np.array(seg_cell_layer['ts']).reshape(-1).shape[0]
                    select_idxs = np.arange(0, num_ts, 1)[::2].tolist()
                    if select_idxs[-1] != num_ts - 1:
                        select_idxs.append(num_ts - 1)
                    select_ratio_dict[seg_name] = {
                        'select_idxs': select_idxs
                    }
                for con_name, con_cell_layer in group_layer['connector'].items():
                    num_ts = np.array(con_cell_layer['ts']).reshape(-1).shape[0]
                    select_idxs = np.arange(0, num_ts, 1)[::2].tolist()
                    if select_idxs[-1] != num_ts - 1:
                        select_idxs.append(num_ts - 1)
                    select_ratio_dict[con_name] = {
                        'select_idxs': select_idxs
                    }
                postprocess_json['setting'][group_name] = select_ratio_dict

            with open(postprocess_file, 'w') as f:
                json.dump(postprocess_json, f, indent=4)
        subprocess.run(f"gedit {postprocess_file};", shell=True)

        # ------
        with open(postprocess_file, 'r') as f:
            postprocess_json = json.load(f)

        self.radius_opt = RadiusOptimizer()
        for group_name, group_layer in h5py.File(res_file).items():
            group_idx = group_layer['group_idx'][0]
            c_xyzr = np.array(group_layer['c_xyzr'])
            group_cfg = postprocess_json['setting'][group_name]

            group_adjuster = GroupRadiusAdjuster(group_idx, table_xyz=c_xyzr[:, :3], table_radius=c_xyzr[:, -1])
            self.radius_opt.add_cell(group_adjuster)

            for seg_name, seg_cell_layer in group_layer['segment'].items():
                cell_idx = seg_cell_layer['idx'][0]
                cell_type = str(seg_cell_layer['cell_type'][0], 'utf-8')
                ts = np.array(seg_cell_layer['ts']).reshape(-1)
                degree = int(seg_cell_layer['degree'][0])
                knot = np.array(seg_cell_layer['knot']).reshape(-1)
                fill_last_one = bool(seg_cell_layer['fill_last_one'][0])
                c_idxs = np.array(seg_cell_layer['c_idxs']).reshape(-1)
                c_radius = np.array(seg_cell_layer['c_radius']).reshape(-1)
                left_clamp = bool(seg_cell_layer['left_clamp'][0])
                right_clamp = bool(seg_cell_layer['right_clamp'][0])
                left_vector = np.array(seg_cell_layer['left_vector']).reshape(-1)
                right_vector = np.array(seg_cell_layer['right_vector']).reshape(-1)
                select_idxs = np.array(group_cfg[seg_name]['select_idxs'])
                group_adjuster.add_cell(
                    cell_idx, group_idx, cell_type, ts[select_idxs], degree, knot, c_idxs, fill_last_one,
                    c_radius, left_clamp, right_clamp, left_vector, right_vector,
                    is_line=is_segment_line_checkbox.checked
                )

            for con_name, con_cell_layer in group_layer['connector'].items():
                cell_idx = con_cell_layer['idx'][0]
                cell_type = str(con_cell_layer['cell_type'][0], 'utf-8')
                ts = np.array(con_cell_layer['ts']).reshape(-1)
                degree = int(con_cell_layer['degree'][0])
                knot = np.array(con_cell_layer['knot']).reshape(-1)
                fill_last_one = bool(con_cell_layer['fill_last_one'][0])
                c_idxs = np.array(con_cell_layer['c_idxs']).reshape(-1)
                c_radius = np.array(con_cell_layer['c_radius']).reshape(-1)
                left_clamp = bool(con_cell_layer['left_clamp'][0])
                right_clamp = bool(con_cell_layer['right_clamp'][0])
                select_idxs = np.array(group_cfg[con_name]['select_idxs'])
                group_adjuster.add_cell(
                    cell_idx, group_idx, cell_type, ts[select_idxs], degree, knot, c_idxs, fill_last_one,
                    c_radius, left_clamp, right_clamp, left_vector=None, right_vector=None,
                    is_line=is_connector_line_checkbox.checked
                )

    def _run_radius_adjust_on_click(self):
        postprocess_file = os.path.join(self.proj_dir, 'postprocess.json')
        if not os.path.exists(postprocess_file):
            self.console_label.text = f"    [ERROR]: postprocess.json isn't exist."
            return
        with open(postprocess_file, 'r') as f:
            postprocess_json = json.load(f)

        loss_record, loss_info = self.radius_opt.optimize(postprocess_json['run_cfg'], postprocess_json['cost_cfg'])
        for name, value in loss_info.items():
            print(f"{name}: {value}")

        # plt.title('loss record')
        # plt.plot(loss_record)
        # plt.show()

    def _radius_check_on_click(self):
        if self.radius_opt is None:
            return
        vis = VisUtils()
        self.radius_opt.draw_wireframe(vis)
        vis.show()

        # self.radius_opt.draw_wireframe()

    def _export_cad_on_click(self, is_linear_link_checkbox: gui.Checkbox):
        if self.radius_opt is None:
            return

        cad_dir = os.path.join(self.proj_dir, 'CAD')
        if os.path.exists(cad_dir):
            shutil.rmtree(cad_dir)
        os.mkdir(cad_dir)

        for group_idx, group_cell in self.radius_opt.group_dict.items():
            StepExporter.to_step(
                group_cell, cad_dir, is_dry_mode=True, with_vis=False, with_fuse=False,
                with_ruled=is_linear_link_checkbox.checked
            )

        self.radius_opt.save_cad_info(file=os.path.join(cad_dir, 'cad_info.json'))

    def init_path_mesh_widget(self):
        mesh_layout = gui.CollapsableVert("Path Mesh Setup", self.spacing, self.blank_margins)
        self.panel.add_child(mesh_layout)

        is_segment_line_checkbox = FragmentOpen3d.get_widget('checkbox', widget_info={'name': 'is segment line'})
        is_segment_line_checkbox.checked = False
        is_connector_line_checkbox = FragmentOpen3d.get_widget('checkbox', widget_info={'name': 'is connector line'})
        is_connector_line_checkbox.checked = True

        setup_radius_adjuster_btn = FragmentOpen3d.get_widget('button', {'name': 'setupAdjuster'})
        setup_radius_adjuster_btn.set_on_clicked(partial(
            self._setup_radius_adjuster_on_click,
            is_segment_line_checkbox=is_segment_line_checkbox,
            is_connector_line_checkbox=is_connector_line_checkbox
        ))

        run_adjust_btn = FragmentOpen3d.get_widget('button', {'name': 'runAdjust'})
        run_adjust_btn.set_on_clicked(self._run_radius_adjust_on_click)

        radius_check_btn = FragmentOpen3d.get_widget('button', {'name': 'radiusCheck'})
        radius_check_btn.set_on_clicked(self._radius_check_on_click)

        is_linear_link_checkbox = FragmentOpen3d.get_widget('checkbox', widget_info={'name': 'is linear link'})
        is_linear_link_checkbox.checked = True

        export_cad_btn = FragmentOpen3d.get_widget('button', {'name': 'exportCAD'})
        export_cad_btn.set_on_clicked(partial(
            self._export_cad_on_click,
            is_linear_link_checkbox=is_linear_link_checkbox
        ))
        export_pcd_btn = FragmentOpen3d.get_widget('button', {'name': 'exportPcd'})
        export_pcd_btn.set_on_clicked(self._export_point_cloud_on_click)

        mesh_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', [
                FragmentOpen3d.get_layout_widget('horiz', [is_segment_line_checkbox, is_connector_line_checkbox]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label("step1:"), setup_radius_adjuster_btn], 5),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label("step2:"), run_adjust_btn], 5),
                FragmentOpen3d.get_layout_widget('horiz', [is_linear_link_checkbox, export_cad_btn], 3),
                FragmentOpen3d.get_layout_widget('horiz', [export_pcd_btn, radius_check_btn], 5),
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
    window = MeshApp(config=config)

    app.run()

    sys.exit("CopyRight From Qu")


if __name__ == '__main__':
    main()
