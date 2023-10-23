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
from gridEnv_utils import GridEnv_Utils

from scripts_py.version_8.mapf_pipeline_py.spanTree_TaskAllocator import SizeTreeTaskRunner

class ObstacleUtils(object):
    @staticmethod
    def create_BoxWallPcd(xmin, ymin, zmin, xmax, ymax, zmax, reso):
        xSteps = math.ceil((xmax - xmin) / reso)
        ySteps = math.ceil((ymax - ymin) / reso)
        zSteps = math.ceil((zmax - zmin) / reso)
        xWall = np.linspace(xmin, xmax, xSteps)
        yWall = np.linspace(ymin, ymax, ySteps)
        zWall = np.linspace(zmin, zmax, zSteps)

        ys, zs = np.meshgrid(yWall, zWall)
        yzs = np.concatenate([ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1).reshape((-1, 2))
        xs, zs = np.meshgrid(xWall, zWall)
        xzs = np.concatenate([xs[..., np.newaxis], zs[..., np.newaxis]], axis=-1).reshape((-1, 2))
        xs, ys = np.meshgrid(xWall, yWall)
        xys = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1).reshape((-1, 2))

        xmin_wall = np.concatenate([np.ones((yzs.shape[0], 1)) * xmin, yzs], axis=1)
        xmax_wall = np.concatenate([np.ones((yzs.shape[0], 1)) * xmax, yzs], axis=1)
        ymin_wall = np.concatenate([xzs[:, :1], np.ones((xzs.shape[0], 1)) * ymin, xzs[:, -1:]], axis=1)
        ymax_wall = np.concatenate([xzs[:, :1], np.ones((xzs.shape[0], 1)) * ymax, xzs[:, -1:]], axis=1)
        zmin_wall = np.concatenate([xys, np.ones((xys.shape[0], 1)) * zmin], axis=1)
        zmax_wall = np.concatenate([xys, np.ones((xys.shape[0], 1)) * zmax], axis=1)

        wall_pcd = np.concatenate([
            xmin_wall, xmax_wall,
            ymin_wall, ymax_wall,
            zmin_wall, zmax_wall
        ], axis=0)
        wall_pcd = pd.DataFrame(wall_pcd).drop_duplicates().values

        return wall_pcd

    @staticmethod
    def create_BoxSolidPcd(xmin, ymin, zmin, xmax, ymax, zmax, reso):
        xSteps = math.ceil((xmax - xmin) / reso)
        ySteps = math.ceil((ymax - ymin) / reso)
        zSteps = math.ceil((zmax - zmin) / reso)
        xs = np.linspace(xmin, xmax, xSteps)
        ys = np.linspace(ymin, ymax, ySteps)
        zs = np.linspace(zmin, zmax, zSteps)

        xs, ys, zs = np.meshgrid(xs, ys, zs)
        xyzs = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1)
        xyzs = xyzs.reshape((-1, 3))
        xyzs = pd.DataFrame(xyzs).drop_duplicates().values

        return xyzs

    @staticmethod
    def create_CylinderSolidPcd(xyz, radius, height, direction, reso):
        assert np.sum(direction) == np.max(direction) == 1.0

        uSteps = math.ceil(radius / reso)
        hSteps = max(math.ceil(height / reso), 2)

        uvs = []
        for cell_radius in np.linspace(0, radius, uSteps):
            length = 2 * cell_radius * np.pi
            num = max(math.ceil(length / reso), 1)

            rads = np.deg2rad(np.linspace(0, 360.0, num))
            uv = np.zeros(shape=(num, 2))
            uv[:, 0] = np.cos(rads) * cell_radius
            uv[:, 1] = np.sin(rads) * cell_radius
            uvs.append(uv)
        uvs = np.concatenate(uvs, axis=0)

        pcds = []
        if direction[0] == 1:
            for h_value in np.linspace(-height/2.0, height/2.0, hSteps):
                pcds.append(np.concatenate([
                    np.ones(shape=(uvs.shape[0], 1)) * h_value,
                    uvs[:, 0:1],
                    uvs[:, 1:2],
                ], axis=1))

        elif direction[1] == 1:
            for h_value in np.linspace(-height/2.0, height/2.0, hSteps):
                pcds.append(np.concatenate([
                    uvs[:, 0:1],
                    np.ones(shape=(uvs.shape[0], 1)) * h_value,
                    uvs[:, 1:2],
                ], axis=1))

        elif direction[2] == 1:
            for h_value in np.linspace(-height/2.0, height/2.0, hSteps):
                pcds.append(np.concatenate([
                    uvs[:, 0:1],
                    uvs[:, 1:2],
                    np.ones(shape=(uvs.shape[0], 1)) * h_value,
                    ], axis=1))
        else:
            raise ValueError

        pcd = np.concatenate(pcds, axis=0)
        pcd = pcd + xyz

        return pcd

    @staticmethod
    def removePointInSphereShell(xyzs, center, radius):
        distance = np.linalg.norm(xyzs - center, ord=2, axis=1)
        xyzs = xyzs[distance > (radius + 0.1)]
        return xyzs

    @staticmethod
    def removePointInBoxShell(xyzs, shellRange):
        xmin, xmax, ymin, ymax, zmin, zmax = shellRange
        invalid = (xyzs[:, 0] >= xmin) & (xyzs[:, 0] <= xmax) & \
                  (xyzs[:, 1] >= ymin) & (xyzs[:, 1] <= ymax) & \
                  (xyzs[:, 2] >= zmin) & (xyzs[:, 2] <= zmax)
        xyzs = xyzs[~invalid]
        return xyzs

    @staticmethod
    def removePointOutBoundary(xyzs:np.array, xmin, xmax, ymin, ymax, zmin, zmax):
        xyzs = xyzs[~(xyzs[:, 0] < xmin)]
        xyzs = xyzs[~(xyzs[:, 0] > xmax)]
        xyzs = xyzs[~(xyzs[:, 1] < ymin)]
        xyzs = xyzs[~(xyzs[:, 1] > ymax)]
        xyzs = xyzs[~(xyzs[:, 2] < zmin)]
        xyzs = xyzs[~(xyzs[:, 2] > zmax)]
        return xyzs

class CustomApp(AppWindow):
    MENU_SAVEJSON = 2
    MENU_LOADJSON = 3

    def __init__(self, config):
        super().__init__(config=config)

        self.init_operateWidget()
        self.init_gridWidget()
        self.init_infoWidget()

        self.groupColors = np.random.uniform(0.0, 1.0, size=(20, 3))

        self.config_params = {
            'envX': -1,
            'envY': -1,
            'envZ': -1,
            'envScaleX': -1,
            'envScaleY': -1,
            'envScaleZ': -1,
            'envGridX': -1,
            'envGridY': -1,
            'envGridZ': -1,
            'grid_scale': -1
        }
        self.planner_params = {}

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
        self.pipeReso_txt = O3D_Utils.getInputWidget(style='double')
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
            O3D_Utils.createHorizCell([
                gui.Label("radius:"), self.pipeRadius_txt,
                gui.Label("Reso:"), self.pipeReso_txt,
            ]),
            self.pipeCreate_btn
        ])
        collapsedLayout.add_child(vlayout)

        self.panel.add_child(collapsedLayout)

    def init_gridWidget(self):
        collapsedLayout = gui.CollapsableVert("GridOperate", self.spacing, self.margins)

        self.visEnvSwitch = gui.ToggleSwitch(":")
        self.visEnvSwitch.is_on = True
        self.visEnvSwitch.set_on_clicked(self.visEnvSwitch_on_click)
        self.envName_Label = gui.Label("RawEnvironemtn")

        self.gridScale_txt = O3D_Utils.getInputWidget(style='double')
        self.gridRun_btn = gui.Button("refitEnvironment")
        self.gridRun_btn.set_on_clicked(self.refitEnv)
        vlayout = O3D_Utils.createVertCell([
            O3D_Utils.createHorizCell([self.envName_Label, self.visEnvSwitch]),
            O3D_Utils.createHorizCell([gui.Label("Scale:"), self.gridScale_txt]),
            self.gridRun_btn
        ])
        collapsedLayout.add_child(vlayout)

        self.panel.add_child(collapsedLayout)

    def creatBoxBtn_on_click(self):
        name: str = self.boxName_txt.text_value
        xmin = self.boxXmin_txt.double_value
        xmax = self.boxXmax_txt.double_value
        ymin = self.boxYmin_txt.double_value
        ymax = self.boxYmax_txt.double_value
        zmin = self.boxZmin_txt.double_value
        zmax = self.boxZmax_txt.double_value
        reso = self.boxReso_txt.double_value
        rgb = np.array([
            self.boxR_txt.double_value,
            self.boxG_txt.double_value,
            self.boxB_txt.double_value
        ])

        if len(name.strip()) == 0:
            return
        if reso == 0:
            return
        if (xmax - xmin) < reso or (ymax - ymin) < reso or (zmax - zmin) < reso:
            return
        if name in self.geoMap.keys():
            return

        if np.sum(rgb) <= 0:
            rgb = np.array([0.0, 1.0, 0.0])
        rgb = rgb / np.sum(rgb)

        xyzs = ObstacleUtils.create_BoxWallPcd(xmin, ymin, zmin, xmax, ymax, zmax, reso)
        pcd_o3d = O3D_Utils.createPCD(xyzs, colors=rgb)
        self.add_pointCloud(name, pcd_o3d)
        self.adjust_centerCamera()

        self.geoMap[name].update({
            'type': 'obstacle',
            'shape': 'Box',
            'desc': {
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
            self.cylinderX_txt.double_value,
            self.cylinderY_txt.double_value,
            self.cylinderZ_txt.double_value
        ])
        direction = np.array([
            self.cylinderDireX_txt.double_value,
            self.cylinderDireY_txt.double_value,
            self.cylinderDireZ_txt.double_value
        ])
        height = self.cylinderHeight_txt.double_value
        radius = self.cylinderRadius_txt.double_value

        reso = self.cylinderReso_txt.double_value
        rgb = np.array([
            self.cylinderR_txt.double_value,
            self.cylinderG_txt.double_value,
            self.cylinderB_txt.double_value
        ])

        if np.sum(direction) == 0:
            return
        if height < reso:
            return
        if radius < reso:
            return
        if reso == 0:
            return
        if name in self.geoMap.keys():
            return

        if np.sum(rgb) <= 0:
            rgb = np.array([0.0, 1.0, 0.0])
        rgb = rgb / np.sum(rgb)

        xyzs = ObstacleUtils.create_CylinderSolidPcd(xyz, radius, height, direction, reso)
        pcd_o3d = O3D_Utils.createPCD(xyzs, colors=rgb)
        self.add_pointCloud(name, pcd_o3d)
        self.adjust_centerCamera()

        self.geoMap[name].update({
            'type': 'obstacle',
            'shape': 'Cylinder',
            'desc': {
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
        xyz = np.array([
            self.pipeX_txt.double_value,
            self.pipeY_txt.double_value,
            self.pipeZ_txt.double_value
        ])
        direction = np.array([
            self.pipeDireX_txt.double_value,
            self.pipeDireY_txt.double_value,
            self.pipeDireZ_txt.double_value
        ])
        groupIdx = self.pipeGroup_txt.int_value
        radius = self.pipeRadius_txt.double_value
        reso = self.pipeReso_txt.double_value

        if name in self.geoMap.keys():
            return
        if radius == 0:
            return
        if radius < reso:
            return
        if reso == 0:
            return

        mesh_o3d = O3D_Utils.createArrow(xyz, direction, radius, color=self.groupColors[groupIdx, :])
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
                'groupIdx': groupIdx,
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
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose path to Load", self.window.theme)
        ### todo unfinish
        # dlg.add_child()
        dlg.add_filter(".json", "Json files (.json)")
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self.on_jsonLoad_dialog_done)
        self.window.show_dialog(dlg)

    def on_jsonExport_dialog_done(self, filename:str):
        if not filename.endswith('.json'):
            return

        self.config_params['envX'] = self.envX_txt.double_value
        self.config_params['envY'] = self.envY_txt.double_value
        self.config_params['envZ'] = self.envZ_txt.double_value
        with open(filename, 'w') as f:
            save_setting = {
                "projectDir": os.path.dirname(filename),
                "global_params": self.config_params,
                "pipeConfig": {},
                "obstacleConfig": {},
                "planner_params": self.planner_params
            }

            obstacle_df = pd.DataFrame(columns=['x', 'y', 'z', 'radius', 'tag'])
            scaleObstacle_df = pd.DataFrame(columns=['x', 'y', 'z', 'radius', 'tag'])

            for name in self.geoMap.keys():
                geoInfo = self.geoMap[name]

                if geoInfo['type'] == 'obstacle':
                    save_setting["obstacleConfig"][name] = {
                        "shape": geoInfo["shape"],
                        'desc': geoInfo["desc"]
                    }

                    if 'scale_desc' in geoInfo.keys():
                        save_setting["obstacleConfig"][name].update({'scale_desc': geoInfo['scale_desc']})

                    if 'pointCloud' in geoInfo.keys():
                        xyzs_df = pd.DataFrame(geoInfo['pointCloud'], columns=['x', 'y', 'z'])
                        xyzs_df[['tag', 'radius']] = name, 0.0
                        obstacle_df = pd.concat([obstacle_df, xyzs_df], axis=0, ignore_index=True)
                        del xyzs_df

                    if 'scale_pointCloud' in geoInfo.keys():
                        xyzs_df = pd.DataFrame(geoInfo['scale_pointCloud'], columns=['x', 'y', 'z'])
                        xyzs_df[['tag', 'radius']] = name, 0.0
                        scaleObstacle_df = pd.concat([scaleObstacle_df, xyzs_df], axis=0, ignore_index=True)
                        del xyzs_df

                elif geoInfo['type'] == 'pipe':
                    groupIdx = int(geoInfo['desc']['groupIdx'])
                    if groupIdx not in save_setting["pipeConfig"].keys():
                        save_setting["pipeConfig"][groupIdx] = {}

                    save_setting["pipeConfig"][groupIdx][name] = {'desc': geoInfo['desc']}

                    if 'scale_desc' in geoInfo.keys():
                        save_setting["pipeConfig"][groupIdx][name].update({'scale_desc': geoInfo['scale_desc']})

            if obstacle_df.shape[0] > 0:
                obstacle_path = os.path.join(save_setting["projectDir"], 'obstacle.csv')
                obstacle_df[['x', 'y', 'z']] = np.round(obstacle_df[['x', 'y', 'z']], decimals=1)
                obstacle_df.drop_duplicates(subset=['x', 'y', 'z'], inplace=True)
                obstacle_df.to_csv(obstacle_path)
                save_setting["obstaclePath"] = obstacle_path

            if scaleObstacle_df.shape[0] > 0:
                scale_obstacle_path = os.path.join(save_setting["projectDir"], 'scale_obstacle.csv')
                scaleObstacle_df[['x', 'y', 'z']] = np.round(scaleObstacle_df[['x', 'y', 'z']], decimals=1)
                scaleObstacle_df.drop_duplicates(subset=['x', 'y', 'z'], inplace=True)
                scaleObstacle_df.to_csv(scale_obstacle_path)
                save_setting["scaleObstaclePath"] = scale_obstacle_path

            json.dump(save_setting, f, indent=4)

    def on_jsonLoad_dialog_done(self, filename: str):
        with open(filename, 'r') as f:
            setting = json.load(f)

        self.config_params = setting["global_params"]
        self.envX_txt.double_value = self.config_params['envX']
        self.envY_txt.double_value = self.config_params['envY']
        self.envZ_txt.double_value = self.config_params['envZ']
        self.gridScale_txt.double_value = self.config_params['grid_scale']

        obstacle_df = pd.read_csv(setting['obstaclePath'], index_col=0)
        find_scaleObstacle_df = False
        if 'scaleObstaclePath' in setting.keys():
            scaleObstacle_df = pd.read_csv(setting['scaleObstaclePath'], index_col=0)
            find_scaleObstacle_df = True

        for name in setting["obstacleConfig"].keys():
            info = setting["obstacleConfig"][name]
            desc = info['desc']

            update_info = {
                'type': 'obstacle',
                'shape': info['shape'],
                'desc': info['desc'],
            }

            xyzs = obstacle_df[obstacle_df['tag'] == name][['x', 'y', 'z']].values
            pcd_o3d = O3D_Utils.createPCD(xyzs, colors=np.array(desc['color']))
            self.add_pointCloud(name, pcd_o3d)

            self.geoMap[name].update(update_info)
            self.geoMap[name].update({'pointCloud': xyzs})
            if 'scale_desc' in info.keys():
                self.geoMap[name].update({'scale_desc': info['scale_desc']})

            if find_scaleObstacle_df:
                xyzs = scaleObstacle_df[scaleObstacle_df['tag'] == name][['x', 'y', 'z']].values
                self.geoMap[name].update({'scale_pointCloud': xyzs})

        for groupIdx in setting["pipeConfig"].keys():
            groupInfo = setting["pipeConfig"][groupIdx]

            for name in groupInfo.keys():
                info = groupInfo[name]
                desc = info['desc']

                mesh_o3d = O3D_Utils.createArrow(
                    np.array(desc['position']),
                    np.array(desc['direction']),
                    desc['radius'],
                    color=self.groupColors[desc['groupIdx'], :]
                )
                with TemporaryDirectory() as tempDir:
                    path = os.path.join(tempDir, 'temp.ply')
                    o3d.io.write_triangle_mesh(path, mesh_o3d)
                    self.add_mesh_fromFile(name, path)

                self.geoMap[name].update({
                    'type': 'pipe',
                    'desc': desc,
                })
                if 'scale_desc' in info.keys():
                    self.geoMap[name].update({'scale_desc': info['scale_desc']})

        self.adjust_centerCamera()

    def refitEnv(self):
        scale = self.gridScale_txt.double_value
        if scale > 1.0 or scale < 0.0:
            return
        self.config_params['grid_scale'] = scale

        minimum_reso = np.inf
        for name in self.geoMap.keys():
            geoInfo = self.geoMap[name]
            if geoInfo['type'] == 'pipe':
                desc = geoInfo['desc']
                minimum_reso = np.minimum(desc['radius'], minimum_reso)
        scale_reso = minimum_reso * scale

        self.config_params['envX'] = self.envX_txt.double_value
        self.config_params['envY'] = self.envY_txt.double_value
        self.config_params['envZ'] = self.envZ_txt.double_value
        self.config_params['envScaleX'] = self.config_params['envX'] * scale
        self.config_params['envScaleY'] = self.config_params['envY'] * scale
        self.config_params['envScaleZ'] = self.config_params['envZ'] * scale
        self.config_params['envGridX'] = math.floor(self.config_params['envScaleX'])
        self.config_params['envGridY'] = math.floor(self.config_params['envScaleY'])
        self.config_params['envGridZ'] = math.floor(self.config_params['envScaleZ'])

        ### ------ 1.1 create scale description
        genMap_names = list(self.geoMap.keys())
        pipeNames = []

        for name in genMap_names:
            geoInfo = self.geoMap[name]
            desc = geoInfo['desc']

            if geoInfo['type'] == 'obstacle':
                if geoInfo['shape'] == 'Box':
                    scale_desc = {
                        'xmin': desc['xmin'] * scale, 'xmax': desc['xmax'] * scale,
                        'ymin': desc['ymin'] * scale, 'ymax': desc['ymax'] * scale,
                        'zmin': desc['zmin'] * scale, 'zmax': desc['zmax'] * scale,
                        'shape_reso': scale_reso
                    }
                    geoInfo['scale_desc'] = scale_desc
                    geoInfo['scale_pointCloud'] = ObstacleUtils.create_BoxWallPcd(
                        scale_desc['xmin'], scale_desc['ymin'], scale_desc['zmin'],
                        scale_desc['xmax'], scale_desc['ymax'], scale_desc['zmax'],
                        scale_desc['shape_reso']
                    )

                    geoInfo['desc'].update({'shape_reso': minimum_reso})
                    geoInfo['pointCloud'] = ObstacleUtils.create_BoxWallPcd(
                        desc['xmin'], desc['ymin'], desc['zmin'], desc['xmax'], desc['ymax'], desc['zmax'],
                        desc['shape_reso']
                    )

                elif geoInfo['shape'] == 'Cylinder':
                    scale_desc = {
                        'position': list(np.array(desc['position']) * scale),
                        'radius': desc['radius'] * scale,
                        'height': desc['height'] * scale,
                        'direction': desc['direction'],
                        'shape_reso': scale_reso,
                    }
                    geoInfo['scale_desc'] = scale_desc
                    geoInfo['scale_pointCloud'] = ObstacleUtils.create_CylinderSolidPcd(
                        np.array(scale_desc['position']), scale_desc['radius'],
                        scale_desc['height'], scale_desc['direction'], scale_desc['shape_reso']
                    )

                    geoInfo['desc'].update({'shape_reso': minimum_reso})
                    geoInfo['pointCloud'] = ObstacleUtils.create_CylinderSolidPcd(
                        np.array(desc['position']), desc['radius'], desc['height'], desc['direction'], desc['shape_reso']
                    )

            elif geoInfo['type'] == 'pipe':
                pipeNames.append(name)

                scale_xyz, xyz_grid = GridEnv_Utils.pipe_convert_grid(
                    np.array(desc['position']), scale,
                    xmin=0, xmax=self.config_params['envGridX'],
                    ymin=0, ymax=self.config_params['envGridY'],
                    zmin=0, zmax=self.config_params['envGridZ'],
                )

                geoInfo['scale_desc'] = {
                    'scale_radius': desc['radius'] * scale,
                    # 'grid_position': list(xyz_grid),
                    'grid_position': [int(xyz_grid[0]), int(xyz_grid[1]), int(xyz_grid[2])],
                    'scale_position': list(scale_xyz),
                }

        ### ------ 1.2 adjust radius
        # self.adjust_pipeRadius_v2(pipeNames)
        self.adjust_pipeRadius_v3(pipeNames)

        ### ------ 1.3 remove shell conflict
        for name in self.geoMap.keys():
            geoInfo = self.geoMap[name]
            if geoInfo['type'] != 'obstacle':
                continue

            xyzs = geoInfo['scale_pointCloud']
            for pipeName in pipeNames:
                scale_desc = self.geoMap[pipeName]['scale_desc']
                xyzs = ObstacleUtils.removePointInSphereShell(
                    xyzs, center=scale_desc["grid_position"], radius=scale_desc["scale_radius"]
                )

            geoInfo['scale_pointCloud'] = ObstacleUtils.removePointOutBoundary(
                xyzs,
                xmin=0, xmax=self.config_params['envScaleX'],
                ymin=0, ymax=self.config_params['envScaleY'],
                zmin=0, zmax=self.config_params['envScaleZ']
            )

        self.showRawEnv()

    def visEnvSwitch_on_click(self, is_on):
        if is_on:
            self.envName_Label = "RawEnvironemtn"
            self.showRawEnv()
        else:
            self.envName_Label = "GridEnvironemtn"
            self.showGridEnv()

    def showGridEnv(self):
        invalid_name = []
        for name in self.geoMap.keys():
            info = self.geoMap[name]

            if info['type'] in 'obstacle':
                if 'scale_pointCloud' not in info.keys():
                    invalid_name.append(name)
                    continue

                pcd_o3d = O3D_Utils.createPCD(info['scale_pointCloud'], colors=np.array(info['desc']['color']))
                self.updateGeo(name, pcd_o3d)

            elif info['type'] == 'pipe':
                mesh_o3d = O3D_Utils.createArrow(
                    np.array(info['scale_desc']['grid_position']),
                    np.array(info['desc']['direction']),
                    info['scale_desc']['scale_radius'],
                    color=self.groupColors[info['desc']['groupIdx'], :]
                )
                with TemporaryDirectory() as tempDir:
                    path = os.path.join(tempDir, 'temp.ply')
                    o3d.io.write_triangle_mesh(path, mesh_o3d)
                    mesh_o3d = o3d.io.read_triangle_model(path)
                self.updateGeo(name, mesh_o3d)

        self.updateVis(excludeName=invalid_name)

    def showRawEnv(self):
        invalid_name = []

        for name in self.geoMap.keys():
            info = self.geoMap[name]

            if info['type'] == 'obstacle':
                if 'pointCloud' not in info.keys():
                    invalid_name.append(name)
                    continue

                pcd_o3d = O3D_Utils.createPCD(info['pointCloud'], colors=np.array(info['desc']['color']))
                self.updateGeo(name, pcd_o3d)

            elif info['type'] == 'pipe':
                mesh_o3d = O3D_Utils.createArrow(
                    np.array(info['desc']['position']),
                    np.array(info['desc']['direction']),
                    info['desc']['radius'],
                    color=self.groupColors[info['desc']['groupIdx'], :]
                )
                with TemporaryDirectory() as tempDir:
                    path = os.path.join(tempDir, 'temp.ply')
                    o3d.io.write_triangle_mesh(path, mesh_o3d)
                    mesh_o3d = o3d.io.read_triangle_model(path)
                self.updateGeo(name, mesh_o3d)

        self.updateVis(excludeName=invalid_name)

    ### --------------------------------------------
    def adjust_pipeRadius_v2(self, pipeNames):
        groupRadius = {}
        for name in pipeNames:
            geoInfo = self.geoMap[name]
            groupIdx = geoInfo['desc']['groupIdx']
            if groupIdx not in groupRadius.keys():
                groupRadius[groupIdx] = []
            groupRadius[groupIdx].append(geoInfo['scale_desc']['scale_radius'])

        for groupIdx in groupRadius.keys():
            groupRadius[groupIdx] = np.mean(groupRadius[groupIdx])

        for name in pipeNames:
            groupIdx = self.geoMap[name]['desc']['groupIdx']
            self.geoMap[name]['scale_desc']['scale_radius'] = groupRadius[groupIdx]

    def adjust_pipeRadius_v3(self, pipeNames):
        groupPipeInfos = {}
        for name in pipeNames:
            geoInfo = self.geoMap[name]
            groupIdx = geoInfo['desc']['groupIdx']
            if groupIdx not in groupPipeInfos.keys():
                groupPipeInfos[groupIdx] = {}
            groupPipeInfos[groupIdx][name] = {
                'desc': geoInfo['desc'],
                'scale_desc': geoInfo['scale_desc']
            }

        taskTrees = {}
        for groupIdx in groupPipeInfos:
            gropPipesInfo = groupPipeInfos[groupIdx]
            allocator = SizeTreeTaskRunner()

            groupPipeNames = list(gropPipesInfo.keys())
            for name in groupPipeNames:
                scaleInfo = gropPipesInfo[name]['scale_desc']
                allocator.add_node(name, pose=scaleInfo['grid_position'], radius=scaleInfo['scale_radius'])

            allocator_res = allocator.getTaskTrees(method='method2')

            res = []
            for name_i, name_j, radius in allocator_res:
                self.geoMap[name_i]['scale_desc']['scale_radius'] = np.maximum(
                    self.geoMap[name_i]['scale_desc']['scale_radius'], radius
                )
                self.geoMap[name_j]['scale_desc']['scale_radius'] = np.maximum(
                    self.geoMap[name_j]['scale_desc']['scale_radius'], radius
                )

                res.append({
                    'name0': name_i, 'name1': name_j, 'radius': radius,
                })

            taskTrees[groupIdx] = res

            # print(groupIdx)
            # for i in res:
            #     print(i)
            # print()

        self.planner_params['taskTree'] = taskTrees

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
