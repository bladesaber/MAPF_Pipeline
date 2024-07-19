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

from scripts_py.version_9.mapf_pkg.app_utils import FragmentOpen3d
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils
from scripts_py.version_9.mapf_pkg.app_smooth import SmoothApp
from scripts_py.version_9.mapf_pkg.pcd2mesh_utils import Pcd2MeshConverter


class MeshApp(SmoothApp):
    def __init__(self, config):
        super().__init__(config=config)
        self.init_path_mesh_widget()

    def _path2pcd_on_click(self):
        algo_file = os.path.join(self.proj_dir, 'algorithm_setup.json')
        if not os.path.exists(algo_file):
            self.console_label.text = "    [ERROR]: No algorithm_setup.json in the project"
            return

        res_file = os.path.join(self.proj_dir, 'smooth_result.hdf5')
        if not os.path.exists(res_file):
            self.console_label.text = f"    [ERROR]: smooth result isn't exist."
            return

        pcd2mesh_file = os.path.join(self.proj_dir, 'pcd2mesh.json')
        if not os.path.exists(pcd2mesh_file):
            pcd2mesh_json = {}
            for _, group_cell in h5py.File(res_file).items():
                group_idx = int(group_cell['group_idx'][0])
                pcd2mesh_json[group_idx] = {}
                for i, (name, xyzr) in enumerate(group_cell['spline'].items()):
                    pcd2mesh_json[group_idx][name] = {
                        'length_reso': 0.2,  # line segment resolution
                        'sphere_reso': 0.15,  # sphere resolution
                        'relax_factor': 1e-3  # remove resolution
                    }
            with open(pcd2mesh_file, 'w') as f:
                json.dump(pcd2mesh_json, f, indent=4)
        subprocess.run(f"gedit {pcd2mesh_file};", shell=True)

        with open(pcd2mesh_file, 'r') as f:
            pcd2mesh_json = json.load(f)
            for key in list(pcd2mesh_json.keys()):
                pcd2mesh_json[int(key)] = pcd2mesh_json.pop(key)

        with open(algo_file, 'r') as f:
            algo_json = json.load(f)
        pipe_cfg = algo_json['pipes']

        mesh_dir = os.path.join(self.proj_dir, 'mesh')
        if os.path.exists(mesh_dir):
            shutil.rmtree(mesh_dir)
        os.mkdir(mesh_dir)

        group_mesher = {}
        for _, group_cell in h5py.File(res_file).items():
            group_idx = int(group_cell['group_idx'][0])
            group_mesher[group_idx] = Pcd2MeshConverter()

            for i, (seg_name, xyzr) in enumerate(group_cell['spline'].items()):
                xyzr = np.array(xyzr)
                path = xyzr[:, :3]
                radius = xyzr[0, -1]
                left_direction, right_direction = None, None
                with_left_clamp, with_right_clamp = False, False

                for pipe_name, pipe_info in pipe_cfg.items():
                    pipe_xyz = np.array(pipe_info['discrete_position'])
                    pipe_direction = np.array(pipe_info['direction'])

                    if np.all(np.isclose(path[0] - pipe_xyz, 0.0)):
                        if pipe_info['is_input']:
                            path = np.concatenate([(path[0] - pipe_direction).reshape((1, -1)), path], axis=0)
                        else:
                            path = np.concatenate([(path[0] + pipe_direction).reshape((1, -1)), path], axis=0)
                        left_direction = pipe_direction
                        with_left_clamp = True

                    if np.all(np.isclose(path[-1] - pipe_xyz, 0.0)):
                        if pipe_info['is_input']:
                            path = np.concatenate([path, (path[-1] - pipe_direction).reshape((1, -1))], axis=0)
                        else:
                            path = np.concatenate([path, (path[-1] + pipe_direction).reshape((1, -1))], axis=0)
                        right_direction = pipe_direction
                        with_right_clamp = True

                group_mesher[group_idx].add_segment(
                    seg_name, path, radius,
                    left_direction=left_direction, right_direction=right_direction,
                    with_left_clamp=with_left_clamp, with_right_clamp=with_right_clamp,
                    reso_info=pcd2mesh_json[group_idx][seg_name]
                )

            # # ------ debug vis
            # for seg_name, info in group_mesher[group_idx].segment_cell.items():
            #     info['surface_cell'].generate_pcd_by_sphere(
            #         length_reso=info['reso_info']['length_reso'],
            #         sphere_reso=info['reso_info']['sphere_reso'],
            #         relax_factor=info['reso_info']['relax_factor']
            #     )
            #     info['surface_cell'].draw()
            # # ------

            group_mesher[group_idx].generate_pcd_data()
            group_mesher[group_idx].remove_inner_pcd()
            pcd_data = group_mesher[group_idx].get_pcd_data()

            pcd_ply = pyvista.PolyData(pcd_data)
            pcd_ply.plot()
            Pcd2MeshConverter.save_ply(pcd_ply, file=os.path.join(mesh_dir, f"group_{group_idx}.ply"))

    def init_path_mesh_widget(self):
        mesh_layout = gui.CollapsableVert("Path Mesh Setup", self.spacing, self.blank_margins)
        self.panel.add_child(mesh_layout)

        run_pcd2mesh_btn = FragmentOpen3d.get_widget('button', {'name': 'run Pcd to Mesh'})
        run_pcd2mesh_btn.set_on_clicked(self._path2pcd_on_click)

        mesh_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', [
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label("step1:"), run_pcd2mesh_btn]),
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
