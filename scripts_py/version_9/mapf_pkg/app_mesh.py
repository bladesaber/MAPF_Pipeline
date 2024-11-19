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

    def _setup_radius_adjuster_on_click(self):
        res_file = os.path.join(self.proj_dir, 'smooth_result.hdf5')
        if not os.path.exists(res_file):
            self.console_label.text = f"    [ERROR]: smooth result isn't exist."
            return

        postprocess_file = os.path.join(self.proj_dir, 'postprocess.json')
        if not os.path.exists(postprocess_file):
            postprocess_json = {
                "run_cfg": {
                    "lr": 0.01,
                    "factor": 0.25,
                    "patience": 1000,
                    "cooldown": 100,
                    "min_lr": 0.001,
                },
                "cost_cfg": {
                    "bound_weight": 1000.0,
                    "scale_weight": 100.0,
                    "balance_weight": 1.0,
                    "keep_weight": 10.0,
                    "max_iter": 30000
                }
            }
            with open(postprocess_file, 'w') as f:
                json.dump(postprocess_json, f, indent=4)
        subprocess.run(f"gedit {postprocess_file};", shell=True)

        self.radius_opt = RadiusOptimizer()
        for group_idx, group_layer in h5py.File(res_file).items():
            group_idx = group_layer['group_idx'][0]
            c_xyzr = np.array(group_layer['c_xyzr'])

            group_adjuster = GroupRadiusAdjuster(group_idx, table_xyz=c_xyzr[:, :3], table_radius=c_xyzr[:, -1])
            self.radius_opt.add_cell(group_adjuster)

            for _, cell_layer in group_layer['data'].items():
                cell_idx = cell_layer['idx'][0]
                cell_type = str(cell_layer['cell_type'][0], 'utf-8')
                ts = np.array(cell_layer['ts']).reshape(-1)
                degree = int(cell_layer['degree'][0])
                knot = np.array(cell_layer['knot']).reshape(-1)
                fill_last_one = bool(cell_layer['fill_last_one'][0])
                c_idxs = np.array(cell_layer['c_idxs']).reshape(-1)
                c_radius = np.array(cell_layer['c_radius']).reshape(-1)
                left_clamp = bool(cell_layer['left_clamp'][0])
                right_clamp = bool(cell_layer['right_clamp'][0])

                if cell_type == 'segment':
                    left_vector = np.array(cell_layer['left_vector']).reshape(-1)
                    right_vector = np.array(cell_layer['right_vector']).reshape(-1)
                    group_adjuster.add_cell(
                        cell_idx, group_idx, cell_type, ts, degree, knot, c_idxs, fill_last_one,
                        c_radius, left_clamp, right_clamp, left_vector, right_vector
                    )
                elif cell_type == 'connector':
                    group_adjuster.add_cell(
                        cell_idx, group_idx, cell_type, ts, degree, knot, c_idxs, fill_last_one,
                        c_radius, left_clamp, right_clamp, left_vector=None, right_vector=None
                    )

    def _run_radius_adjust_on_click(self):
        postprocess_file = os.path.join(self.proj_dir, 'postprocess.json')
        if not os.path.exists(postprocess_file):
            self.console_label.text = f"    [ERROR]: postprocess.json isn't exist."
            return
        with open(postprocess_file, 'r') as f:
            postprocess_json = json.load(f)

        loss_record, loss_info = self.radius_opt.optimize(
            postprocess_json['run_cfg'], postprocess_json['cost_cfg']
        )
        for name, value in loss_info.items():
            print(f"{name}: {value}")

        plt.title('loss record')
        plt.plot(loss_record)
        plt.show()

    def _radius_check_on_click(self):
        if self.radius_opt is None:
            return
        self.radius_opt.draw_wireframe()

    def _export_step_on_click(self):
        if self.radius_opt is None:
            return

        cad_dir = os.path.join(self.proj_dir, 'STEP')
        if os.path.exists(cad_dir):
            shutil.rmtree(cad_dir)
        os.mkdir(cad_dir)

        for group_idx, group_cell in self.radius_opt.group_dict.items():
            StepExporter.to_step(group_cell, cad_dir)

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
                FragmentOpen3d.get_layout_widget('horiz', [export_step_btn, export_pcd_btn], 10),
                FragmentOpen3d.get_layout_widget('horiz', [radius_check_btn]),
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
