import shutil
import numpy as np
import sys
import os
import json
import open3d.visualization.gui as gui
import argparse
import subprocess
import pyvista
import matplotlib.pyplot as plt
from functools import partial
from typing import Dict

from scripts_py.version_9.mapf_pkg.app_utils import FragmentOpen3d
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils
from scripts_py.version_9.mapf_pkg.app_mesh import MeshApp

from scripts_py.version_9.simulator.kepsilon_openfoam import KEpsilonSimulator


class SimulateApp(MeshApp):
    def __init__(self, config):
        super().__init__(config=config)
        self.init_path_simulate_widget()

    def _generate_stl_json(self, stl_file: str, scale_txt: gui.NumberEdit, task_name_txt: gui.TextEdit):
        task_name = task_name_txt.text_value
        scale = scale_txt.double_value
        if scale <= 0 or task_name == 'None':
            return

        simulate_dir = os.path.join(self.proj_dir, 'simulate')
        if not os.path.exists(simulate_dir):
            os.mkdir(simulate_dir)

        simulate_dir = os.path.join(simulate_dir, task_name)
        simulate_json_file = os.path.join(simulate_dir, f'run_{task_name}.json')

        if not os.path.exists(simulate_dir):
            os.mkdir(simulate_dir)

            # ------ step 1 convert scale mesh
            mesh: pyvista.PolyData = pyvista.read(stl_file)
            mesh.points = mesh.points * scale
            scale_stl_file = os.path.join(simulate_dir, f"scale_{os.path.basename(stl_file).split('.')[0]}.stl")
            mesh.save(scale_stl_file, binary=False)

            mesh: pyvista.PolyData = pyvista.read(scale_stl_file)
            xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
            range_xyz = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
            range_tol = np.min(range_xyz)

            # ------ step 2 init json
            cad_info_file = os.path.join(os.path.join(self.proj_dir, 'CAD'), 'cad_info.json')
            if not os.path.exists(cad_info_file):
                cad_info_file = None

            run_json = {
                'proj_dir': os.path.join(simulate_dir, 'run_openfoam'),
                'mesh_file': scale_stl_file,
                'scale': scale,
                'group_idx': None,
                'cad_info_file': cad_info_file,
                'gmsh_mesh_cfg': {
                    'meshSizeMin': 0.001,
                    'meshSizeMax': 0.001
                },
                'openfoam_mesh_cfg': {
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                    'zmin': zmin,
                    'zmax': zmax,
                    'x_grid': 50,
                    'y_grid': 50,
                    'z_grid': 50,
                    'tol': max(range_tol * 0.1, 1e-3),
                    'convertToMeters': 1.0,
                },
                'controlDict': {
                    'application': 'simpleFoam',
                    'endTime': 5.0,
                    'deltaT': 0.005,
                    'writeInterval': 100
                },
                'transportProperties': {
                    'nu': 0.01
                },
                'boundary_definition': {},
                'simulation_cfg': {},
                'background_cfg': {}
            }

            with open(simulate_json_file, 'w') as f:
                json.dump(run_json, f, indent=4)
            subprocess.run(f"gedit {simulate_json_file};", shell=True)

            with open(simulate_json_file, 'r') as f:
                run_json = json.load(f)

            if (run_json['cad_info_file'] is not None) and (run_json['group_idx'] is not None):
                with open(run_json['cad_info_file'], 'r') as f:
                    cad_info_dict: Dict = json.load(f)

                for group_idx_str, group_info in cad_info_dict.items():
                    group_idx = int(group_idx_str)
                    if group_idx != run_json['group_idx']:
                        continue

                    scale = run_json['scale']
                    run_json['openfoam_mesh_cfg'].update({
                        'inside_x': group_info['inside_point'][0] * scale,
                        'inside_y': group_info['inside_point'][1] * scale,
                        'inside_z': group_info['inside_point'][2] * scale,
                    })

                    for pipe_name, pipe_info in group_info['pipe_cfgs'].items():
                        run_json['boundary_definition'][pipe_name] = {
                            'method': 'cylinder',
                            'center': (np.array(pipe_info['center']) * run_json['scale']).tolist(),
                            'direction': pipe_info['direction'],
                            'tol1': range_tol * 0.01,
                            'tol2': range_tol * 0.01,
                            'radius': pipe_info['radius'] * 0.975 * run_json['scale'],
                            'source': 'boundary'
                        }

                        if pipe_info['pipe_type'] == 'Inlet':
                            run_json['simulation_cfg'][pipe_name] = {
                                'U': {
                                    'type': 'fixedValue',
                                    'value': None
                                },
                                'p': {'type': 'zeroGradient'},
                                'nut': {'type': 'calculated', 'value': 0},
                                'k': {
                                    'type': 'fixedValue',
                                    'value': None
                                },
                                'epsilon': {
                                    'type': 'fixedValue',
                                    'value': None
                                },
                            }
                        elif pipe_info['pipe_type'] == 'Outlet':
                            run_json['simulation_cfg'][pipe_name] = {
                                'U': {'type': 'zeroGradient'},
                                'p': {
                                    'type': 'fixedValue',
                                    'value': None
                                },
                                'nut': {
                                    'type': 'calculated',
                                    'value': 0
                                },
                                'k': {'type': 'zeroGradient'},
                                'epsilon': {'type': 'zeroGradient'}
                            }
                        else:
                            run_json['simulation_cfg'][pipe_name] = {}

                run_json['simulation_cfg']['boundary'] = {
                    'U': {'type': 'noSlip'},
                    'p': {'type': 'zeroGradient'},
                    'nut': {
                        'type': 'nutkWallFunction',
                        'value': 0
                    },
                    'k': {
                        'type': 'kqRWallFunction',
                        'value': None
                    },
                    'epsilon': {
                        'type': 'epsilonWallFunction',
                        'value': None
                    }
                }

                run_json['background_cfg'].update({
                    'U': {'internalField': [0.0, 0.0, 0.0]},
                    'p': {'internalField': 0.0},
                    'nut': {'internalField': 0.0},
                    'k': {'internalField': None},
                    'epsilon': {'internalField': None},
                })

            with open(simulate_json_file, 'w') as f:
                json.dump(run_json, f, indent=4)

        subprocess.run(f"gedit {simulate_json_file};", shell=True)

    @staticmethod
    def _setup_openfoam_project(json_file: str):
        assert json_file.endswith('.json')
        with open(json_file, 'r') as f:
            run_json = json.load(f)
        KEpsilonSimulator(run_json, by_method='openfoam')

    @staticmethod
    def _run_simulate(json_file: str):
        assert json_file.endswith('.json')
        with open(json_file, 'r') as f:
            run_cfg = json.load(f)
        KEpsilonSimulator.run(run_cfg)

    def init_path_simulate_widget(self):
        simulate_layout = gui.CollapsableVert("Simulate Setup", self.spacing, self.blank_margins)
        self.panel.add_child(simulate_layout)

        task_name_txt = FragmentOpen3d.get_widget('text', {'text_value': 'None'})
        scale_txt = FragmentOpen3d.get_widget(
            'number', {'style': 'double', 'decimal_precision': 4, 'init_value': -1.0, 'preferred_width': 50}
        )

        def _gen_simulate_cfg_btn_on_click():
            filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file", self.window.theme)
            filedlg.add_filter(".stl .STL", "Triangle mesh (.stl .STL)")
            filedlg.set_on_cancel(self.window.close_dialog)
            filedlg.set_on_done(partial(
                self._generate_stl_json, scale_txt=scale_txt, task_name_txt=task_name_txt
            ))
            self.window.show_dialog(filedlg)

        gen_json_dlg_btn = FragmentOpen3d.get_widget('button', {'name': 'importStl'})
        gen_json_dlg_btn.set_on_clicked(_gen_simulate_cfg_btn_on_click)

        # ------------------------------------------
        def _setup_openfoam_proj_btn_on_click():
            filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file", self.window.theme)
            filedlg.add_filter(".json", "Triangle mesh (.json)")
            filedlg.set_on_cancel(self.window.close_dialog)
            filedlg.set_on_done(SimulateApp._setup_openfoam_project)
            self.window.show_dialog(filedlg)

        setup_openfoam_btn = FragmentOpen3d.get_widget('button', {'name': 'setupOpenfoamProject'})
        setup_openfoam_btn.set_on_clicked(_setup_openfoam_proj_btn_on_click)

        # ------------------------------------------
        def _run_simulate_btn_on_click():
            filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file", self.window.theme)
            filedlg.add_filter(".json", "Triangle mesh (.json)")
            filedlg.set_on_cancel(self.window.close_dialog)
            filedlg.set_on_done(SimulateApp._run_simulate)
            self.window.show_dialog(filedlg)

        run_simulate_btn = FragmentOpen3d.get_widget('button', {'name': 'runSim'})
        run_simulate_btn.set_on_clicked(_run_simulate_btn_on_click)

        # ------------------------------------------
        simulate_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', [
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label("step1:taskName"), task_name_txt], 5),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label("scale:"), scale_txt, gen_json_dlg_btn], 5),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label("step2:"), setup_openfoam_btn]),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label("step3:"), run_simulate_btn]),
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
    window = SimulateApp(config=config)
    app.run()

    sys.exit("CopyRight From Qu")


if __name__ == '__main__':
    main()
