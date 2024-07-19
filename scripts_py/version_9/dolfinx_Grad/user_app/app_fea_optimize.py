import json
import shutil
import numpy as np
import open3d.visualization.gui as gui
import argparse
import sys
import os
from typing import Dict, List, Union
from functools import partial
import subprocess
import pyvista
import open3d as o3d

from scripts_py.version_9.mapf_pkg.app_mesh import MeshApp
from scripts_py.version_9.mapf_pkg.app_utils import FragmentOpen3d
from scripts_py.version_9.dolfinx_Grad.user_app.step1_create_project import create_project
from scripts_py.version_9.dolfinx_Grad.user_app.step2_simulate import openfoam_simulate, MeshUtils
from scripts_py.version_9.dolfinx_Grad.user_app.step3_shape_optimize import load_obstacle, load_base_model
from scripts_py.version_9.dolfinx_Grad.fluid_tools.fluid_shape_opt import FluidConditionalModel, \
    FluidShapeRecombineLayer

"""
1.仿真
2.单体优化
3.多体优化
4.多工况优化
5.多目标优化
"""


class FeaApp(MeshApp):
    def __init__(self, config):
        super().__init__(config=config)

        self.fea_dir = os.path.join(self.proj_dir, 'FEA')
        if not os.path.exists(self.fea_dir):
            os.mkdir(self.fea_dir)

        self.mesh_selected_combox = FragmentOpen3d.get_widget('combobox', {})
        self.mesh_info = {}

        self.init_fea_widget()

    def _init_fea_projects_on_click(
            self, file: str, work_condition_num_txt: gui.NumberEdit,
            rescale_txt: gui.NumberEdit, stl_rgb_vec: gui.VectorEdit
    ):
        if not file.endswith('.stl'):
            self.console_label.text = f"    [Info]: Non valid mesh format."
            return
        base_name = os.path.basename(file).replace('.stl', '').replace('.STL', '')

        mesh_dir = os.path.join(self.fea_dir, base_name)
        model_stl = os.path.join(mesh_dir, f"{base_name}.stl")
        rgb = stl_rgb_vec.vector_value

        if not os.path.exists(mesh_dir):
            os.mkdir(mesh_dir)
            create_project(
                proj_dir=mesh_dir, simulate_methods=['navier_stoke', 'openfoam'], with_recombine_cfg=True
            )
            with open(os.path.join(mesh_dir, 'mesh_reclassify.geo'), 'w') as f:
                f.write(f"// Please use Gmsh Tool(Reclassify 2D) to split mesh and save msh for next step.\n")
                f.write(f'Merge "{model_stl}";\n')
            with open(os.path.join(mesh_dir, 'model.geo'), 'w') as f:
                f.write(f'Merge "{os.path.join(mesh_dir, "mesh_reclassify.msh")}";\n')

            recombine_cfgs = ['fea_cfg.json']
            if work_condition_num_txt.int_value > 1:
                shutil.move(
                    src=os.path.join(mesh_dir, 'fea_cfg.json'),
                    dst=os.path.join(mesh_dir, 'fea_cfg_0.json')
                )
                recombine_cfgs = ['fea_cfg_0.json']
                for cond_id in range(1, work_condition_num_txt.int_value, 1):
                    shutil.copy(
                        src=os.path.join(mesh_dir, 'fea_cfg_0.json'),
                        dst=os.path.join(mesh_dir, f"fea_cfg_{cond_id}.json")
                    )
                    recombine_cfgs.append(f"fea_cfg_{cond_id}.json")

            with open(os.path.join(mesh_dir, 'recombine_cfg.json'), 'r') as f:
                recombine_cfg = json.load(f)
            recombine_cfg['recombine_cfgs'] = recombine_cfgs
            with open(os.path.join(mesh_dir, 'recombine_cfg.json'), 'w') as f:
                json.dump(recombine_cfg, f, indent=4)

        if not os.path.exists(model_stl):
            mesh: pyvista.PolyData = pyvista.read(file)
            mesh.points = mesh.points * rescale_txt.double_value
            mesh.save(os.path.join(mesh_dir, f"{base_name}.stl"), binary=True, recompute_normals=True)

            with open(os.path.join(mesh_dir, 'recombine_cfg.json'), 'r') as f:
                recombine_cfg = json.load(f)
            recombine_cfg['rescale'] = rescale_txt.double_value
            with open(os.path.join(mesh_dir, 'recombine_cfg.json'), 'w') as f:
                json.dump(recombine_cfg, f, indent=4)

        if base_name not in self.mesh_info.keys():
            self.mesh_selected_combox.add_item(base_name)

        with open(os.path.join(mesh_dir, 'recombine_cfg.json'), 'r') as f:
            recombine_cfg = json.load(f)
        self.mesh_info[base_name] = {'proj_dir': mesh_dir, 'rescale': recombine_cfg['rescale'], 'rgb': rgb}

        self.update_mesh(base_name, model_stl, is_visible=True, rgb=rgb)

    def _reclassify_mesh_on_click(self):
        if self.mesh_selected_combox.selected_text not in self.mesh_info.keys():
            self.console_label.text = f"    [Error]: Non valid mesh name."
            return
        mesh_dir = self.mesh_info[self.mesh_selected_combox.selected_text]['proj_dir']
        if not os.path.exists(os.path.join(mesh_dir, 'mesh_reclassify.geo')):
            self.console_label.text = f"    [Error]: mesh_reclassify.geo doesn't exist."
            return
        subprocess.run(f"gmsh {os.path.join(mesh_dir, 'mesh_reclassify.geo')};", shell=True)

    def _update_mesh_element_on_click(self):
        if self.mesh_selected_combox.selected_text not in self.mesh_info.keys():
            self.console_label.text = f"    [Error]: Non valid mesh name."
            return
        mesh_dir = self.mesh_info[self.mesh_selected_combox.selected_text]['proj_dir']
        if not os.path.exists(os.path.join(mesh_dir, 'mesh_reclassify.msh')):
            self.console_label.text = f"    [Error]: mesh_reclassify.msh doesn't exist."
            return
        subprocess.run(f"gmsh {os.path.join(mesh_dir, 'model.geo')};", shell=True)

    def _update_fea_setting_on_click(self):
        if self.mesh_selected_combox.selected_text not in self.mesh_info.keys():
            self.console_label.text = f"    [Error]: Non valid mesh name."
            return
        mesh_dir = self.mesh_info[self.mesh_selected_combox.selected_text]['proj_dir']

        # step1: update inlet/outlet function
        subprocess.run(f"gedit {os.path.join(mesh_dir, 'condition.py')};", shell=True)

        # step2: update working condition
        recombine_file = os.path.join(mesh_dir, 'recombine_cfg.json')
        subprocess.run(f"gedit {recombine_file};", shell=True)
        with open(recombine_file, 'r') as f:
            recombine_cfg = json.load(f)

        # step2: update conditional simulation setting
        for cfg_name in recombine_cfg['recombine_cfgs']:
            cfg_file = os.path.join(mesh_dir, cfg_name)
            subprocess.run(f"gedit {cfg_file};", shell=True)

    def _generate_mesh_element_on_click(self):
        if self.mesh_selected_combox.selected_text not in self.mesh_info.keys():
            self.console_label.text = f"    [Error]: Non valid mesh name."
            return
        mesh_dir = self.mesh_info[self.mesh_selected_combox.selected_text]['proj_dir']

        recombine_file = os.path.join(mesh_dir, 'recombine_cfg.json')
        with open(recombine_file, 'r') as f:
            recombine_cfg = json.load(f)

        cfg_names = recombine_cfg['recombine_cfgs']
        if len(cfg_names) == 0:
            self.console_label.text = f"    [Error]: Zero Working Condition."
            return

        with open(os.path.join(mesh_dir, cfg_names[0]), 'r') as f:
            run_cfg = json.load(f)

        MeshUtils.msh_to_XDMF(
            name='model', dim=run_cfg['dim'],
            msh_file=os.path.join(run_cfg['proj_dir'], run_cfg['msh_file']),
            output_file=os.path.join(run_cfg['proj_dir'], run_cfg['xdmf_file'])
        )

    def _simulate_run_on_click(self):
        if self.mesh_selected_combox.selected_text not in self.mesh_info.keys():
            self.console_label.text = f"    [Error]: Non valid mesh name."
            return
        mesh_dir = self.mesh_info[self.mesh_selected_combox.selected_text]['proj_dir']

        recombine_file = os.path.join(mesh_dir, 'recombine_cfg.json')
        with open(recombine_file, 'r') as f:
            recombine_cfg = json.load(f)

        openfoam_simulate(
            cfg=recombine_cfg, init_mesh=False, msh_tag='msh_file', xdmf_tag='xdmf_file',
            openfoam_remesh=False, geo_tag='geo_file',
        )

    def _shape_optimize_on_click(self):
        models: List[Union[FluidConditionalModel, FluidShapeRecombineLayer]] = []
        log_list, check_names = [], []

        for name, info in self.mesh_info.items():
            mesh_dir = info['proj_dir']

            with open(os.path.join(mesh_dir, 'recombine_cfg.json'), 'r') as f:
                recombine_cfg: dict = json.load(f)
            cfg_files = recombine_cfg['recombine_cfgs']
            if len(cfg_files) == 0:
                continue

            elif len(cfg_files) == 1:
                with open(os.path.join(mesh_dir, cfg_files[0]), 'r') as f:
                    run_cfg: dict = json.load(f)
                model, log_dict = load_base_model(run_cfg, args, tag_name=None)
                log_list.append(log_dict)

            else:
                model = FluidShapeRecombineLayer(tag_name=recombine_cfg['tag_name'])
                for sub_cfg_name in cfg_files:
                    with open(os.path.join(run_cfg['proj_dir'], sub_cfg_name), 'r') as f:
                        run_cfg = json.load(f)
                    sub_model, log_dict = load_base_model(run_cfg, args, tag_name=model.tag_name)
                    model.add_condition_opt(sub_model)
                    log_list.append(log_dict)



    def init_fea_widget(self):
        fea_layout = gui.CollapsableVert("FEA Setup", self.spacing, self.blank_margins)
        self.panel.add_child(fea_layout)

        # ------ step 1
        rescale_txt = FragmentOpen3d.get_widget('number', {
            'style': 'double', 'init_value': 1.0, 'preferred_width': 60.0, 'decimal_precision': 4
        })
        work_condition_num_txt = FragmentOpen3d.get_widget('number', {'style': 'int', 'init_value': 1})
        stl_rgb_vec = FragmentOpen3d.get_widget('vector', {'init_value': [0.3, 0.3, 0.3]})

        def _file_dlg_btn_on_click():
            filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file", self.window.theme)
            filedlg.add_filter(".obj .ply .stl", "Triangle mesh (.stl)")
            filedlg.set_on_cancel(self.window.close_dialog)
            filedlg.set_on_done(partial(
                self._init_fea_projects_on_click,
                work_condition_num_txt=work_condition_num_txt,
                rescale_txt=rescale_txt,
                stl_rgb_vec=stl_rgb_vec
            ))
            self.window.show_dialog(filedlg)

        init_project_btn = FragmentOpen3d.get_widget('button', {'name': 'init FEA project'})
        init_project_btn.set_on_clicked(_file_dlg_btn_on_click)

        # ------ step 2
        reclassify_mesh_btn = FragmentOpen3d.get_widget('button', {'name': 'reclassify mesh'})
        reclassify_mesh_btn.set_on_clicked(self._reclassify_mesh_on_click)
        update_mesh_element_btn = FragmentOpen3d.get_widget('button', {'name': 'update mesh element'})
        update_mesh_element_btn.set_on_clicked(self._update_mesh_element_on_click)
        generate_mesh_element_btn = FragmentOpen3d.get_widget('button', {'name': 'generate mesh element'})
        generate_mesh_element_btn.set_on_clicked(self._generate_mesh_element_on_click)

        # ------ step 3
        update_fea_setting_btn = FragmentOpen3d.get_widget('button', {'name': 'update fea setting'})
        update_fea_setting_btn.set_on_clicked(self._update_fea_setting_on_click)
        simulate_btn = FragmentOpen3d.get_widget('button', {'name': 'fluid simulate'})
        simulate_btn.set_on_clicked(self._simulate_run_on_click)

        paraview_btn = FragmentOpen3d.get_widget('button', {'name': 'visualization tool'})
        paraview_btn.set_on_clicked(lambda: subprocess.run(f"paraview;", shell=True))

        fea_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', [
                gui.Label("step1:"),
                FragmentOpen3d.get_layout_widget('horiz', [gui.Label("rgb:"), stl_rgb_vec]),
                FragmentOpen3d.get_layout_widget('horiz', [
                    gui.Label("rescale:"), rescale_txt, init_project_btn
                ], space=2),
                FragmentOpen3d.get_layout_widget('vert', [
                    gui.Label("step2:"),
                    FragmentOpen3d.get_layout_widget('horiz', [gui.Label("Selected Mesh:"), self.mesh_selected_combox]),
                    FragmentOpen3d.get_layout_widget('horiz', [
                        gui.Label("step 2.1:"), reclassify_mesh_btn
                    ]),
                    FragmentOpen3d.get_layout_widget('horiz', [
                        gui.Label("step 2.2:"), update_mesh_element_btn
                    ]),
                    FragmentOpen3d.get_layout_widget('horiz', [
                        gui.Label("step 2.3:"), generate_mesh_element_btn
                    ])
                ]),
                FragmentOpen3d.get_layout_widget('horiz', [
                    gui.Label("step 3:"), update_fea_setting_btn
                ]),
                FragmentOpen3d.get_layout_widget('horiz', [
                    gui.Label("step 4:"), simulate_btn
                ]),
                paraview_btn
            ]),
        )

    def update_mesh(self, base_name, file, is_visible: bool, rgb: np.ndarray):
        mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(file)
        vertices = np.asarray(mesh_o3d.vertices) * (1.0 / self.mesh_info[base_name]['rescale'])
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(np.tile(rgb.reshape((1, -1)), [vertices.shape[0], 1]))
        self.add_point_cloud(
            base_name, mesh_o3d, param={'type': 'pipe_stl'}, is_visible=is_visible, no_save=True
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
    window = FeaApp(config=config)

    app.run()
    sys.exit("CopyRight From Qu")


if __name__ == '__main__':
    main()
