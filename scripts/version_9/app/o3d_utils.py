import pandas as pd
import numpy as np
from typing import Dict, Union
import sys
from functools import partial
import math
from scipy.spatial import transform

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class O3D_Utils(object):
    @staticmethod
    def createPCD(xyzs: np.array, colors: np.array = None) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs)

        if colors is not None:
            colors = np.tile(colors.reshape((1, 3)), (xyzs.shape[0], 1))
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    @staticmethod
    def createMesh(xyzs: np.array, triangles: np.array, color: np.array = None) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(xyzs)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        if color is not None:
            mesh.paint_uniform_color(color)
        return mesh

    @staticmethod
    def dire2Rotmat(vec):
        gamma = math.atan2(vec[1], vec[0])
        Rz = transform.Rotation.from_euler(
            seq='xyz', angles=np.array([0.0, 0.0, gamma]), degrees=False
        ).as_matrix()

        vec = np.linalg.inv(Rz) @ vec.reshape((-1, 1))
        vec = vec.reshape(-1)

        beta = math.atan2(vec[0], vec[2])
        Ry = transform.Rotation.from_euler(
            seq='xyz', angles=np.array([0.0, beta, 0.0]), degrees=False
        ).as_matrix()

        rot_mat = Rz @ Ry

        return rot_mat

    @staticmethod
    def createArrow(xyz, vec, radius, color: np.array = None):
        arrow: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=radius * 0.5,
            cone_radius=radius,
            cylinder_height=radius * 0.7,
            cone_height=radius * 0.3,
            # cylinder_radius=1.5,
            # cone_radius=1.5 * 2.0,
            # cylinder_height=3,
            # cone_height=3 * 0.3,
        )
        rot_mat = O3D_Utils.dire2Rotmat(vec)
        arrow.rotate(rot_mat, center=np.array([0., 0., 0.]))
        arrow.translate(xyz, relative=False)

        if color is not None:
            arrow.paint_uniform_color(color)
        return arrow

    @staticmethod
    def getInputWidget(style: str, init_value=None):
        if style == 'double':
            input_txt = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            input_txt.set_preferred_width(40.0)
            input_txt.decimal_precision = 1
        elif style == 'text':
            input_txt = gui.TextEdit()
        elif style == 'int':
            input_txt = gui.NumberEdit(gui.NumberEdit.INT)
            # input_txt.set_preferred_width(40.0)
            input_txt.set_limits(0, 10)
        else:
            raise ValueError

        if init_value is not None:
            input_txt.int_value = init_value

        return input_txt

    @staticmethod
    def createVertCell(widget_list, space=0, margins=gui.Margins(0)):
        vlayout = gui.Vert(space, margins)
        for widget in widget_list:
            vlayout.add_child(widget)
        return vlayout

    @staticmethod
    def createHorizCell(widget_list, space=0, margins=gui.Margins(0)):
        hlayout = gui.Horiz(space, margins)
        for widget in widget_list:
            hlayout.add_child(widget)
        return hlayout

class AppWindow(object):
    MENU_QUIT = 1
    MENU_ABOUT = 99

    ### material
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    def __init__(self, config: Dict):
        self.config = config

        ### ------ global params
        self.selected_geo = None
        self.geoMap = {}

        self.window = gui.Application.instance.create_window("Reconstruct", self.config['width'], self.config['height'])

        self.init_menu()
        self.init_scence()
        self.init_panel()
        self.init_geoVisWidget()

        self.window.set_on_layout(self._on_layout)
        self.widget3d.scene.show_axes(True)

    def init_panel(self):
        em = self.window.theme.font_size
        self.spacing = int(np.round(0.1 * em))
        vspacing = int(np.round(0.1 * em))
        self.margins = gui.Margins(vspacing)
        self.panel = gui.Vert(self.spacing, self.margins)

        self.window.add_child(self.panel)

    def init_scence(self):
        ### ------ init scence widget
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        ### ------ init material
        self.materials = {
            AppWindow.LIT: rendering.MaterialRecord(),
            AppWindow.UNLIT: rendering.MaterialRecord(),
            AppWindow.NORMALS: rendering.MaterialRecord(),
            AppWindow.DEPTH: rendering.MaterialRecord()
        }
        self.materials[AppWindow.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self.materials[AppWindow.LIT].shader = AppWindow.LIT
        self.materials[AppWindow.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self.materials[AppWindow.UNLIT].shader = AppWindow.UNLIT
        self.materials[AppWindow.NORMALS].shader = AppWindow.NORMALS
        self.materials[AppWindow.DEPTH].shader = AppWindow.DEPTH

        self.window.add_child(self.widget3d)

    def init_menu(self):
        if gui.Application.instance.menubar is None:
            self.file_menu = gui.Menu()
            self.file_menu.add_item("Quit", AppWindow.MENU_QUIT)

            self.help_menu = gui.Menu()
            self.help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("File", self.file_menu)
            menu.add_menu("Help", self.help_menu)
            gui.Application.instance.menubar = menu

        self.window.set_on_menu_item_activated(AppWindow.MENU_QUIT, gui.Application.instance.quit)
        self.window.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)

    def _on_layout(self, layout_context):
        em = layout_context.theme.font_size

        panel_width = 20 * em
        rect = self.window.content_rect

        self.widget3d.frame = gui.Rect(rect.x, rect.y, rect.get_right() - panel_width, rect.height)
        self.panel.frame = gui.Rect(self.widget3d.frame.get_right(), rect.y, panel_width, rect.height)

    def init_geoVisWidget(self):
        collapsedLayout = gui.CollapsableVert("Vis", self.spacing, self.margins)
        self.panel.add_child(collapsedLayout)

        vlayout = gui.Vert()
        collapsedLayout.add_child(vlayout)

        self.geoVis_layout = gui.CollapsableVert("geoVis", self.spacing, self.margins)
        vlayout.add_child(self.geoVis_layout)

        hlayout = gui.Horiz(self.spacing, self.margins)
        hlayout.add_child(gui.Label("Selected:"))
        self.selectGeo_combo = gui.Combobox()
        self.selectGeo_combo.set_on_selection_changed(self.selectGeo_combo_change)
        hlayout.add_child(self.selectGeo_combo)
        vlayout.add_child(hlayout)

        self.geoDelete_btn = gui.Button("Delete")
        self.geoDelete_btn.set_on_clicked(self.geoDelete_on_click)
        vlayout.add_child(self.geoDelete_btn)

    def init_infoWidget(self):
        self.info_layout = gui.CollapsableVert("info", self.spacing, self.margins)
        self.info_content = gui.Label("[info]: Welcome")
        self.panel.add_child(self.info_content)

    # ------------------------------------------------------------------
    def _on_menu_about(self):
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Version 0.1"))

        ok = gui.Button("Cancel")
        ok.set_on_clicked(self.window.close_dialog)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def add_pointCloud_fromFile(self, name: str, file: str, isSilent=False):
        try:
            pcd = o3d.io.read_point_cloud(file)
        except:
            pcd = None

        if pcd is not None:
            if not isSilent:
                self.widget3d.scene.add_geometry(name, pcd, self.materials[AppWindow.LIT])
                self.add_GeoItem(name, checked=True)
            else:
                self.add_GeoItem(name, checked=False)
            self.geoMap[name] = {
                'name': name,
                'style': 'PointCloud',
                'isVisible': not isSilent,
                'geometry': pcd
            }

    def add_pointCloud(self, name: str, pcd: o3d.geometry.PointCloud, isSilent=False):
        if not isSilent:
            self.widget3d.scene.add_geometry(name, pcd, self.materials[AppWindow.LIT])
            self.add_GeoItem(name, checked=True)
        else:
            self.add_GeoItem(name, checked=False)
        self.geoMap[name] = {
            'name': name,
            'style': 'PointCloud',
            'isVisible': not isSilent,
            'geometry': pcd
        }

    def add_mesh_fromFile(self, name: str, file: str, isSilent=False):
        geometry_type = o3d.io.read_file_geometry_type(file)
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            try:
                mesh = o3d.io.read_triangle_model(file)
            except:
                mesh = None

            if mesh is not None:
                if not isSilent:
                    self.widget3d.scene.add_model(name, mesh)
                    self.add_GeoItem(name, True)
                else:
                    self.add_GeoItem(name, False)
                self.geoMap[name] = {
                    'name': name,
                    'style': 'Mesh',
                    'isVisible': not isSilent,
                    'geometry': mesh
                }

    def add_mesh(self, name: str, mesh: o3d.visualization.rendering.TriangleMeshModel, isSilent=False):
        if not isSilent:
            self.widget3d.scene.add_model(name, mesh)
            self.add_GeoItem(name, True)
        else:
            self.add_GeoItem(name, False)
        self.geoMap[name] = {
            'name': name,
            'style': 'Mesh',
            'isVisible': not isSilent,
            'geometry': mesh
        }

    def adjust_centerCamera(self):
        ### adjust camera viewpoint
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60, bounds, bounds.get_center())

    # -------------------------------------------------------------------
    def selectGeo_combo_change(self, val, idx):
        self.selected_geo = val

    def add_GeoItem(self, name: str, checked):
        if len(name.strip()) == 0:
            return

        checkbox = gui.Checkbox(name)
        checkbox.checked = checked
        checkbox.set_on_checked(partial(self.geoCheckBox_on_click, name=name))
        self.geoVis_layout.add_child(checkbox)
        self.selectGeo_combo.add_item(name)

    def delete_GeoItem(self, name: str):
        if name is None:
            return

        if name in self.geoMap.keys():
            isVisible = self.geoMap[name]['isVisible']
            if isVisible:
                self.widget3d.scene.remove_geometry(name)

            del self.geoMap[name]
            self.selectGeo_combo.remove_item(name)
            # self.geoVis_layout.remove_item()

    def updateVis(self, excludeName=None):
        for name in self.geoMap.keys():
            if excludeName is not None:
                if name in excludeName:
                    self.widget3d.scene.remove_geometry(name)
                    continue

            if self.geoMap[name]['isVisible']:
                self.widget3d.scene.remove_geometry(name)

                style = self.geoMap[name]['style']
                if style == 'PointCloud':
                    self.widget3d.scene.add_geometry(name, self.geoMap[name]['geometry'], self.materials[AppWindow.LIT])
                elif style == 'Mesh':
                    self.widget3d.scene.add_model(name, self.geoMap[name]['geometry'])

    def updateGeo(self, name, geometry: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]):
        if name not in self.geoMap.keys():
            return
        self.geoMap[name]['geometry'] = geometry

    def geoDelete_on_click(self):
        self.delete_GeoItem(self.selected_geo)

    def geoCheckBox_on_click(self, is_checked, name):
        if name in self.geoMap.keys():
            isVisible = self.geoMap[name]['isVisible']
            if isVisible != is_checked:
                style = self.geoMap[name]['style']
                if is_checked:
                    if style == 'PointCloud':
                        self.widget3d.scene.add_geometry(
                            name, self.geoMap[name]['geometry'], self.materials[AppWindow.LIT]
                        )
                    elif style == 'Mesh':
                        self.widget3d.scene.add_model(name, self.geoMap[name]['geometry'])
                else:
                    self.widget3d.scene.remove_geometry(name)
                self.geoMap[name]['isVisible'] = is_checked


def main():
    app = gui.Application.instance
    app.initialize()

    config = {
        'width': 1080,
        'height': 720,
    }
    window = AppWindow(config=config)

    app.run()

    # sys.exit("GoodBye")


if __name__ == '__main__':
    main()

    raise ValueError("Why ??")
