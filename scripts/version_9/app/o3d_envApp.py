import numpy as np
import pandas as pd
import math
import sys
import os
import json
import open3d as o3d
from tempfile import TemporaryDirectory

import open3d.visualization.gui as gui

from o3d_utils import AppWindow
from o3d_utils import O3D_Utils
from env_utils import GridEnv_Utils, Shape_Utils

from scripts.version_9.mapf_pipeline_py.spanTree_TaskAllocator import SizeTreeTaskRunner

class CustomApp(AppWindow):
    MENU_SAVEJSON = 2
    MENU_LOADJSON = 3

    def __init__(self, config):
        super().__init__(config=config)

        self.init_operateWidget()
        self.init_infoWidget()

        self.groupColors = np.random.uniform(0.0, 1.0, size=(20, 3))

        self.config_params = {
            'envX': -1, 'envY': -1, 'envZ': -1,
        }

    def init_menu(self):
        super().init_menu()

        self.file_menu.add_item("SaveJson", CustomApp.MENU_SAVEJSON)
        self.file_menu.add_item("LoadJson", CustomApp.MENU_LOADJSON)

        self.window.set_on_menu_item_activated(CustomApp.MENU_SAVEJSON, self._on_menu_saveJson)
        self.window.set_on_menu_item_activated(CustomApp.MENU_LOADJSON, self._on_menu_loadJson)

    def init_operateWidget(self):
        collapsedLayout = gui.CollapsableVert("EnvOperate", self.spacing, self.margins)

        self.envX_txt = O3D_Utils.getInputWidget(style='double')
        self.envY_txt = O3D_Utils.getInputWidget(style='double')
        self.envZ_txt = O3D_Utils.getInputWidget(style='double')
        vlayout = O3D_Utils.createVertCell([
            gui.Label("Environment Block"),
            O3D_Utils.createHorizCell([
                gui.Label("X:"), self.envX_txt,
                gui.Label("Y:"), self.envY_txt,
                gui.Label("Z:"), self.envZ_txt,
            ])
        ])
        collapsedLayout.add_child(vlayout)
        collapsedLayout.add_fixed(10)

        self.boxName_txt = O3D_Utils.getInputWidget(style='text')
        self.boxXmin_txt = O3D_Utils.getInputWidget(style='double')
        self.boxXmax_txt = O3D_Utils.getInputWidget(style='double')
        self.boxYmin_txt = O3D_Utils.getInputWidget(style='double')
        self.boxYmax_txt = O3D_Utils.getInputWidget(style='double')
        self.boxZmin_txt = O3D_Utils.getInputWidget(style='double')
        self.boxZmax_txt = O3D_Utils.getInputWidget(style='double')
        self.boxReso_txt = O3D_Utils.getInputWidget(style='double')
        self.boxR_txt = O3D_Utils.getInputWidget(style='double')
        self.boxG_txt = O3D_Utils.getInputWidget(style='double')
        self.boxB_txt = O3D_Utils.getInputWidget(style='double')
        self.boxCreate_btn = gui.Button("createBox_Obstacle")
        self.boxCreate_btn.set_on_clicked(self.creatBoxBtn_on_click)
        vlayout = O3D_Utils.createVertCell([
            gui.Label("Create Box"),
            O3D_Utils.createHorizCell([gui.Label("Name:"), self.boxName_txt]),
            O3D_Utils.createHorizCell([
                O3D_Utils.createVertCell([
                    O3D_Utils.createHorizCell([gui.Label("xmin:"), self.boxXmin_txt]),
                    O3D_Utils.createHorizCell([gui.Label("xmax:"), self.boxXmax_txt]),
                ]),
                O3D_Utils.createVertCell([
                    O3D_Utils.createHorizCell([gui.Label("ymin:"), self.boxYmin_txt]),
                    O3D_Utils.createHorizCell([gui.Label("ymax:"), self.boxYmax_txt]),
                ]),
                O3D_Utils.createVertCell([
                    O3D_Utils.createHorizCell([gui.Label("zmin:"), self.boxZmin_txt]),
                    O3D_Utils.createHorizCell([gui.Label("zmax:"), self.boxZmax_txt]),
                ]),
            ]),
            O3D_Utils.createHorizCell([
                gui.Label("R:"), self.boxR_txt, gui.Label("G:"), self.boxG_txt, gui.Label("B:"), self.boxB_txt
            ]),
            O3D_Utils.createHorizCell([
                gui.Label("Reso:"), self.boxReso_txt,
            ]),
            self.boxCreate_btn,
        ])
        collapsedLayout.add_child(vlayout)
        collapsedLayout.add_fixed(10)

        self.cylinderName_txt = O3D_Utils.getInputWidget(style='text')
        self.cylinderX_txt = O3D_Utils.getInputWidget(style='double')
        self.cylinderY_txt = O3D_Utils.getInputWidget(style='double')
        self.cylinderZ_txt = O3D_Utils.getInputWidget(style='double')
        self.cylinderDireX_txt = O3D_Utils.getInputWidget(style='double')
        self.cylinderDireY_txt = O3D_Utils.getInputWidget(style='double')
        self.cylinderDireZ_txt = O3D_Utils.getInputWidget(style='double')
        self.cylinderRadius_txt = O3D_Utils.getInputWidget(style='double')
        self.cylinderHeight_txt = O3D_Utils.getInputWidget(style='double')
        self.cylinderReso_txt = O3D_Utils.getInputWidget(style='double')
        self.cylinderR_txt = O3D_Utils.getInputWidget(style='double')
        self.cylinderG_txt = O3D_Utils.getInputWidget(style='double')
        self.cylinderB_txt = O3D_Utils.getInputWidget(style='double')
        self.cylinderCreate_btn = gui.Button("createCylinder_Obstacle")
        self.cylinderCreate_btn.set_on_clicked(self.creatCylinderBtn_on_click)
        vlayout = O3D_Utils.createVertCell([
            gui.Label("Create Cylinder"),
            O3D_Utils.createHorizCell([gui.Label("Name:"), self.cylinderName_txt]),
            O3D_Utils.createHorizCell([
                gui.Label("x:"), self.cylinderX_txt, gui.Label("xDire:"), self.cylinderDireX_txt
            ]),
            O3D_Utils.createHorizCell([
                gui.Label("y:"), self.cylinderY_txt, gui.Label("yDire:"), self.cylinderDireY_txt
            ]),
            O3D_Utils.createHorizCell([
                gui.Label("z:"), self.cylinderZ_txt, gui.Label("zDire:"), self.cylinderDireZ_txt
            ]),
            O3D_Utils.createHorizCell([
                gui.Label("Radius:"), self.cylinderRadius_txt,
                gui.Label("Height:"), self.cylinderHeight_txt,
                gui.Label("Reso:"), self.cylinderReso_txt,
            ]),
            O3D_Utils.createHorizCell([
                gui.Label("R:"), self.cylinderR_txt,
                gui.Label("G:"), self.cylinderG_txt,
                gui.Label("B:"), self.cylinderB_txt
            ]),
            self.cylinderCreate_btn
        ])
        collapsedLayout.add_child(vlayout)
        collapsedLayout.add_fixed(10)

        self.pipeName_txt = O3D_Utils.getInputWidget(style='text')
        self.pipeX_txt = O3D_Utils.getInputWidget(style='double')
        self.pipeY_txt = O3D_Utils.getInputWidget(style='double')
        self.pipeZ_txt = O3D_Utils.getInputWidget(style='double')
        self.pipeDireX_txt = O3D_Utils.getInputWidget(style='double')
        self.pipeDireY_txt = O3D_Utils.getInputWidget(style='double')
        self.pipeDireZ_txt = O3D_Utils.getInputWidget(style='double')
        self.pipeGroup_txt = O3D_Utils.getInputWidget(style='int', init_value=-1)
        self.pipeRadius_txt = O3D_Utils.getInputWidget(style='double')
        self.pipeCreate_btn = gui.Button("createPipe")
        self.pipeCreate_btn.set_on_clicked(self.creatPipeBtn_on_click)
        vlayout = O3D_Utils.createVertCell([
            gui.Label("Create Pipe"),
            O3D_Utils.createHorizCell([gui.Label("Name:"), self.pipeName_txt]),
            O3D_Utils.createHorizCell([
                gui.Label("x:"), self.pipeX_txt, gui.Label("xDire:"), self.pipeDireX_txt
            ]),
            O3D_Utils.createHorizCell([
                gui.Label("y:"), self.pipeY_txt, gui.Label("yDire:"), self.pipeDireY_txt
            ]),
            O3D_Utils.createHorizCell([
                gui.Label("z:"), self.pipeZ_txt, gui.Label("zDire:"), self.pipeDireZ_txt
            ]),
            O3D_Utils.createHorizCell([
                gui.Label("group:"), self.pipeGroup_txt
            ]),
            O3D_Utils.createHorizCell([gui.Label("radius:"), self.pipeRadius_txt]),
            self.pipeCreate_btn
        ])
        collapsedLayout.add_child(vlayout)

        self.panel.add_child(collapsedLayout)

    def creatBoxBtn_on_click(self):
        name: str = self.boxName_txt.text_value
        xmin, ymin, zmin = self.boxXmin_txt.double_value, self.boxYmin_txt.double_value, self.boxZmin_txt.double_value
        xmax, ymax, zmax = self.boxXmax_txt.double_value, self.boxYmax_txt.double_value, self.boxZmax_txt.double_value
        reso = self.boxReso_txt.double_value
        rgb = np.array([self.boxR_txt.double_value, self.boxG_txt.double_value, self.boxB_txt.double_value])

        if (
                len(name.strip()) != 1.0
                or reso == 0
                or (xmax - xmin) <= 1.5 * reso or (ymax - ymin) <= 1.5 * reso or (zmax - zmin) <= 1.5 * reso
                or name in self.geoMap.keys()
        ):
            self.info_content.text = "[Warning]: Please Check Valid Parameters for Box Define"
            return
        if np.sum(rgb) <= 0:
            rgb = np.array([0.3, 0.3, 0.3])

        xyzs = Shape_Utils.create_BoxPcd(xmin, ymin, zmin, xmax, ymax, zmax, reso)
        pcd_o3d = O3D_Utils.createPCD(xyzs, colors=rgb)
        self.add_pointCloud(name, pcd_o3d)
        self.adjust_centerCamera()

        self.geoMap[name].update({
            'type': 'obstacle',
            'desc': {
                'shape': 'Box',
                'xmin': xmin, 'xmax': xmax,
                'ymin': ymin, 'ymax': ymax,
                'zmin': zmin, 'zmax': zmax,
                'shape_reso': reso,
                'color': list(rgb)
            },
            'pointCloud': xyzs,
        })

    def creatCylinderBtn_on_click(self):
        name: str = self.cylinderName_txt.text_value
        xyz = np.array([
            self.cylinderX_txt.double_value, self.cylinderY_txt.double_value, self.cylinderZ_txt.double_value
        ])
        direction = np.array([
            self.cylinderDireX_txt.double_value,
            self.cylinderDireY_txt.double_value,
            self.cylinderDireZ_txt.double_value
        ])
        height, radius = self.cylinderHeight_txt.double_value, self.cylinderRadius_txt.double_value
        reso = self.cylinderReso_txt.double_value
        rgb = np.array([
            self.cylinderR_txt.double_value, self.cylinderG_txt.double_value, self.cylinderB_txt.double_value
        ])

        if (
                np.sum(direction) != 1.0
                or height < 1.5 * reso or radius < 1.5 * reso or reso == 0
                or name in self.geoMap.keys()
        ):
            self.info_content.text = "[Warning]: Please Check Valid Parameters for Cylinder Define"
            return
        if np.sum(rgb) <= 0:
            rgb = np.array([0.3, 0.3, 0.3])

        xyzs = Shape_Utils.create_CylinderPcd(xyz, radius, height, direction, reso)
        pcd_o3d = O3D_Utils.createPCD(xyzs, colors=rgb)
        self.add_pointCloud(name, pcd_o3d)
        self.adjust_centerCamera()

        self.geoMap[name].update({
            'type': 'obstacle',
            'desc': {
                'shape': 'Cylinder',
                'position': list(xyz),
                'radius': radius,
                'height': height,
                'direction': list(direction),
                'shape_reso': reso,
                'color': list(rgb)
            },
            'pointCloud': xyzs,
        })

    def creatPipeBtn_on_click(self):
        name: str = self.pipeName_txt.text_value
        xyz = np.array([self.pipeX_txt.double_value, self.pipeY_txt.double_value, self.pipeZ_txt.double_value])
        direction = np.array([
            self.pipeDireX_txt.double_value, self.pipeDireY_txt.double_value, self.pipeDireZ_txt.double_value
        ])
        group_idx = self.pipeGroup_txt.int_value
        radius = self.pipeRadius_txt.double_value

        if (
                name in self.geoMap.keys()
                or radius < 0.5
        ):
            self.info_content.text = "[Warning]: Please Check Valid Parameters for Pipe Define"
            return

        mesh_o3d = O3D_Utils.createArrow(xyz, direction, radius, color=self.groupColors[group_idx, :])
        with TemporaryDirectory() as tempDir:
            path = os.path.join(tempDir, 'temp.ply')
            o3d.io.write_triangle_mesh(path, mesh_o3d)
            self.add_mesh_fromFile(name, path)
        self.adjust_centerCamera()

        self.geoMap[name].update({
            'type': 'pipe',
            'desc': {
                'position': list(xyz),
                'direction': list(direction),
                'groupIdx': group_idx,
                'radius': radius,
            }
        })

    def _on_menu_saveJson(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose path to output", self.window.theme)
        dlg.add_filter(".json", "Json files (.json)")
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self.on_jsonExport_dialog_done)
        self.window.show_dialog(dlg)

    def _on_menu_loadJson(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose path to Load", self.window.theme)
        dlg.add_filter(".json", "Json files (.json)")
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self.on_jsonLoad_dialog_done)
        self.window.show_dialog(dlg)

    def on_jsonExport_dialog_done(self, filename: str):
        if not filename.endswith('.json'):
            return

        self.config_params['envX'] = self.envX_txt.double_value
        self.config_params['envY'] = self.envY_txt.double_value
        self.config_params['envZ'] = self.envZ_txt.double_value

        obstacle_df = pd.DataFrame(columns=['x', 'y', 'z', 'radius', 'tag'])
        obstacles_cfg, pipes_cfg = {}, {}
        for name in self.geoMap.keys():
            geo_info = self.geoMap[name]
            if geo_info['type'] == 'obstacle':
                obstacles_cfg[name] = geo_info["desc"]

                xyzs_df = pd.DataFrame(geo_info['pointCloud'], columns=['x', 'y', 'z'])
                xyzs_df[['tag', 'radius']] = name, 0.0
                obstacle_df = pd.concat([obstacle_df, xyzs_df], axis=0, ignore_index=True)

            elif geo_info['type'] == 'pipe':
                group_idx = int(geo_info['desc']['groupIdx'])
                if group_idx not in pipes_cfg.keys():
                    pipes_cfg[group_idx] = {}
                pipes_cfg[group_idx][name] = geo_info['desc']

        with open(filename, 'w') as f:
            save_setting = {
                "project_dir": os.path.dirname(filename),
                "global_params": self.config_params,
                "pipe_cfgs": pipes_cfg,
                "obstacle_cfgs": obstacles_cfg,
            }

            if obstacle_df.shape[0] > 0:
                obstacle_path = os.path.join(save_setting["project_dir"], 'obstacle.csv')
                obstacle_df[['x', 'y', 'z']] = np.round(obstacle_df[['x', 'y', 'z']], decimals=1)
                obstacle_df.drop_duplicates(subset=['x', 'y', 'z'], inplace=True)
                obstacle_df.to_csv(obstacle_path)
                save_setting["obstacle_path"] = obstacle_path

            json.dump(save_setting, f, indent=4)

    def on_jsonLoad_dialog_done(self, filename: str):
        with open(filename, 'r') as f:
            setting = json.load(f)

        self.config_params = setting["global_params"]
        self.envX_txt.double_value = self.config_params['envX']
        self.envY_txt.double_value = self.config_params['envY']
        self.envZ_txt.double_value = self.config_params['envZ']

        obstacle_df = pd.read_csv(setting['obstacle_path'], index_col=0)
        for name in setting["obstacle_cfgs"].keys():
            info = setting["obstacle_cfgs"][name]

            xyzs = obstacle_df[obstacle_df['tag'] == name][['x', 'y', 'z']].values
            if info['shape'] == 'shell':
                obstacle_type = 'shell'
                pcd_o3d = O3D_Utils.createPCD(xyzs, colors=np.array([1.0, 0.0, 0.0]))
            else:
                obstacle_type = 'obstacle'
                pcd_o3d = O3D_Utils.createPCD(xyzs, colors=np.array(info['color']))
            self.add_pointCloud(name, pcd_o3d)

            update_info = {
                'type': obstacle_type,
                'desc': info,
                'pointCloud': xyzs,
            }
            self.geoMap[name].update(update_info)

        for group_idx_str in setting["pipe_cfgs"].keys():
            group_infos = setting["pipe_cfgs"][group_idx_str]
            for name in group_infos.keys():
                info = group_infos[name]

                mesh_o3d = O3D_Utils.createArrow(
                    np.array(info['position']), np.array(info['direction']),
                    radius=info['radius'], color=self.groupColors[info['groupIdx']]
                )
                with TemporaryDirectory() as tempDir:
                    path = os.path.join(tempDir, 'temp.ply')
                    o3d.io.write_triangle_mesh(path, mesh_o3d)
                    self.add_mesh_fromFile(name, path)

                self.geoMap[name].update({'type': 'pipe', 'desc': info})
        self.adjust_centerCamera()

def main():
    # o3d.visualization.webrtc_server.enable_webrtc()

    app = gui.Application.instance
    app.initialize()

    config = {
        'width': 960,
        'height': 720,
    }
    window = CustomApp(config=config)

    app.run()

    sys.exit("CopyRight From Qu")

if __name__ == '__main__':
    main()
