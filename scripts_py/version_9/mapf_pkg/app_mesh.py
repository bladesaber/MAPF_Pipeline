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

from scripts_py.version_9.mapf_pkg.app_utils import FragmentOpen3d
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils
from scripts_py.version_9.mapf_pkg.app_smooth import SmoothApp
from scripts_py.version_9.mapf_pkg.point_cloud_utils import PathSurfaceGenerator, Pcd2MeshUtils
from scripts_py.version_9.mapf_pkg.postprocess_utils import GroupRadiusAdjuster


class MeshApp(SmoothApp):
    def __init__(self, config):
        super().__init__(config=config)
        self.init_path_mesh_widget()

        self.groupRadiusAdjuster = GroupRadiusAdjuster()

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

    def _setup_radius_adjuster_on_click(self):
        res_file = os.path.join(self.proj_dir, 'smooth_result.hdf5')
        if not os.path.exists(res_file):
            self.console_label.text = f"    [ERROR]: smooth result isn't exist."
            return

        postprocess_file = os.path.join(self.proj_dir, 'postprocess.json')
        if not os.path.exists(postprocess_file):
            postprocess_json = {
                "run_cfg": {
                    "lr": 0.002,
                    "factor": 0.25,
                    "patience": 1000,
                    "cooldown": 100,
                    "min_lr": 0.001,
                },
                "cost_cfg": {
                    "scale_weight": 100.0,
                    "balance_weight": 1.0,
                    "keep_weight": 10.0
                }
            }
            with open(postprocess_file, 'w') as f:
                json.dump(postprocess_json, f, indent=4)
        subprocess.run(f"gedit {postprocess_file};", shell=True)

        for group_idx, group_layer in h5py.File(res_file).items():
            for _, cell_layer in group_layer['data'].items():
                cell_type = str(cell_layer['cell_type'][()])
                cell_idx = int(cell_layer['idx'][()])
                cell_flag = f"{group_idx}/{cell_idx}_{cell_type}"
                ts = np.array(cell_layer['ts'])
                degree = int(cell_layer['degree'][()])
                knot = np.array(cell_layer['knot'])
                c_xyz = np.array(cell_layer['c_xyz'])
                fill_last_one = bool(cell_layer['fill_last_one'][()])
                c_radius = np.array(cell_layer['c_radius'])
                self.groupRadiusAdjuster.add_cell(
                    cell_flag, ts=ts, degree=degree, knot=knot,
                    c_xyz=c_xyz, fill_last_one=fill_last_one, c_radius=c_radius
                )

    def _run_radius_adjust_on_click(self):
        postprocess_file = os.path.join(self.proj_dir, 'postprocess.json')
        if not os.path.exists(postprocess_file):
            self.console_label.text = f"    [ERROR]: postprocess.json isn't exist."
            return
        with open(postprocess_file, 'r') as f:
            postprocess_json = json.load(f)

        loss_record, loss_info = self.groupRadiusAdjuster.optimize(
            postprocess_json['run_cfg'], postprocess_json['cost_cfg']
        )

        plt.title('loss record')
        plt.plot(loss_record)
        plt.show()

    def _radius_check_on_click(self):
        self.groupRadiusAdjuster.draw_error_check()

    def _export_step_on_click(self):
        raise NotImplementedError

    def init_path_mesh_widget(self):
        mesh_layout = gui.CollapsableVert("Path Mesh Setup", self.spacing, self.blank_margins)
        self.panel.add_child(mesh_layout)

        setup_radius_adjuster_btn = FragmentOpen3d.get_widget('button', {'name': 'setupAdjuster'})
        setup_radius_adjuster_btn.set_on_clicked(self._setup_radius_adjuster_on_click)

        run_adjust_btn = FragmentOpen3d.get_widget('button', {'name': 'runAdjust'})
        run_adjust_btn.set_on_clicked(self._run_radius_adjust_on_click)

        radius_check_btn = FragmentOpen3d.get_widget('button', {'name': 'radiusCheck'})
        radius_check_btn.set_on_clicked(self._radius_check_on_click)

        export_step_btn = FragmentOpen3d.get_widget('button', {'name': 'exportSTEP'})
        export_step_btn.set_on_clicked(self._export_step_on_click)
        export_pcd_btn = FragmentOpen3d.get_widget('button', {'name': 'exportPcd'})
        export_pcd_btn.set_on_clicked(self._export_point_cloud_on_click)

        mesh_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', [
                FragmentOpen3d.get_layout_widget('horiz', [
                    gui.Label("step1&2:"), setup_radius_adjuster_btn, run_adjust_btn
                ], 10),
                FragmentOpen3d.get_layout_widget('horiz', [radius_check_btn]),
                FragmentOpen3d.get_layout_widget('horiz', [export_step_btn, export_pcd_btn], 10),
            ])
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
