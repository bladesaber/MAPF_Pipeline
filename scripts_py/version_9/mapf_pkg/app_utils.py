import numpy as np
from typing import Dict, Union
from functools import partial
from typing import Literal

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from scripts_py.version_9.mapf_pkg.shape_utils import AxisTransform


class FragmentOpen3d(object):
    @staticmethod
    def create_pcd(pcd: np.array, colors: np.array = None) -> o3d.geometry.PointCloud:
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        if colors is not None:
            colors = np.tile(colors.reshape((1, 3)), (pcd.shape[0], 1))
            pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
        return pcd_o3d

    @staticmethod
    def create_mesh(pcd: np.array, triangle_cells: np.array, color: np.array = None) -> o3d.geometry.TriangleMesh:
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(pcd)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(triangle_cells)
        if color is not None:
            mesh_o3d.paint_uniform_color(color)
        return mesh_o3d

    @staticmethod
    def create_arrow(xyz, vec, radius, color: np.array = None):
        arrow: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=radius * 0.5, cone_radius=radius,
            cylinder_height=radius * 0.7, cone_height=radius * 0.3,
        )
        arrow.compute_triangle_normals()
        rot_mat = AxisTransform.vector_to_rotation_mat(vec)
        arrow.rotate(rot_mat, center=np.array([0., 0., 0.]))
        arrow.translate(xyz, relative=False)
        if color is not None:
            arrow.paint_uniform_color(color)
        return arrow

    @staticmethod
    def get_widget(widget_type: str, widget_info: dict):
        if widget_type == 'number':
            if widget_info['style'] == 'double':
                widget = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                widget.set_preferred_width(widget_info.get('preferred_width', 40.0))
                widget.decimal_precision = widget_info.get('decimal_precision', 2)
                widget.double_value = widget_info.get('init_value', 0.0)

            else:
                widget = gui.NumberEdit(gui.NumberEdit.INT)
                widget.set_preferred_width(widget_info.get('preferred_width', 40.0))
                widget.int_value = widget_info.get('init_value', 0)

        elif widget_type == 'text':
            widget = gui.TextEdit()
            widget.text_value = widget_info.get('text_value', '')

        elif widget_type == 'label':
            widget = gui.Label(widget_info.get('name', "Label"))

        elif widget_type == 'button':
            widget = gui.Button(widget_info.get('name', "button"))

        elif widget_type == 'dialog':
            widget = gui.Dialog(widget_info.get('name', "dialog"))

        elif widget_type == 'combobox':
            widget = gui.Combobox()
            for item in widget_info.get('items', []):
                widget.add_item(item)

        elif widget_type == 'vector':
            widget = gui.VectorEdit()
            widget.vector_value = widget_info.get('init_value', [0., 0., 0.])

        elif widget_type == 'checkbox':
            widget = gui.Checkbox(widget_info.get('name', 'checkbox'))

        else:
            raise ValueError("[ERROR]: Non-Valid")

        return widget

    @staticmethod
    def get_layout_widget(
            layout_type: Literal["vert", "horiz"], widgets, space=0, margins=gui.Margins(0), with_stretch=False
    ):
        if layout_type == 'vert':
            layout = gui.Vert(space, margins)
        else:
            layout = gui.Horiz(space, margins)

        for widget in widgets:
            if with_stretch:
                layout.add_stretch()
            layout.add_child(widget)
        return layout


class AppWindow(object):
    MENU_QUIT = 1
    MENU_ABOUT = 99

    # --- material
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"
    materials = {
        "defaultLit": rendering.MaterialRecord(),
        "defaultUnlit": rendering.MaterialRecord(),
        "normals": rendering.MaterialRecord(),
        "depth": rendering.MaterialRecord()
    }

    def __init__(self, config: Dict):
        self.config = config
        self.geoMap = {}

        self.window = gui.Application.instance.create_window("Space", self.config['width'], self.config['height'])
        self.spacing = int(self.window.theme.font_size)
        self.center_margins = gui.Margins(left=self.spacing, top=self.spacing, right=self.spacing, bottom=self.spacing)
        self.vert_margins = gui.Margins(left=self.spacing, top=0, right=self.spacing, bottom=0)
        self.horiz_margins = gui.Margins(left=0, top=self.spacing, right=0, bottom=self.spacing)
        self.blank_margins = gui.Margins(left=0, top=0, right=0, bottom=0)

        self.init_menu()
        self.init_scence()
        self.init_panel()
        self.init_visualize_widget()
        self.init_info_widget()

        self.window.set_on_layout(self._on_layout)
        self.widget3d.scene.show_axes(True)

    def init_menu(self):
        gui.Application.instance.menubar = gui.Menu()

        self.file_menu = gui.Menu()
        self.file_menu.add_item("Quit", AppWindow.MENU_QUIT)
        self.help_menu = gui.Menu()
        self.help_menu.add_item("About", AppWindow.MENU_ABOUT)

        gui.Application.instance.menubar.add_menu("File", self.file_menu)
        gui.Application.instance.menubar.add_menu("Help", self.help_menu)

        self.window.set_on_menu_item_activated(AppWindow.MENU_QUIT, gui.Application.instance.quit)
        self.window.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)

    def init_scence(self):
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        self.materials[AppWindow.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self.materials[AppWindow.LIT].shader = AppWindow.LIT
        self.materials[AppWindow.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self.materials[AppWindow.UNLIT].shader = AppWindow.UNLIT
        self.materials[AppWindow.NORMALS].shader = AppWindow.NORMALS
        self.materials[AppWindow.DEPTH].shader = AppWindow.DEPTH

        self.window.add_child(self.widget3d)

    def init_panel(self):
        self.panel = gui.Vert(self.spacing, self.blank_margins)
        self.window.add_child(self.panel)

    def init_visualize_widget(self):
        vis_layout = gui.CollapsableVert("visualize", self.spacing, self.vert_margins)
        self.panel.add_child(vis_layout)

        self.geo_container = gui.CollapsableVert("geometry", self.spacing, self.vert_margins)

        def combo_change(val, idx):
            self.selected_geo = val

        self.selected_geo = None
        self.geo_selected_combox = FragmentOpen3d.get_widget('combobox', {})
        self.geo_selected_combox.set_on_selection_changed(combo_change)

        def delete_on_click():
            self.delete_geo_item(self.selected_geo)

        self.geo_delete_btn = FragmentOpen3d.get_widget('button', {'name': 'delete'})
        self.geo_delete_btn.set_on_clicked(delete_on_click)

        sub_widget0 = FragmentOpen3d.get_layout_widget(
            'horiz',
            [FragmentOpen3d.get_widget('label', {'name': 'Geometry:'}), self.geo_selected_combox],
        )

        widgets = [self.geo_container, sub_widget0, self.geo_delete_btn]
        vis_layout.add_child(
            FragmentOpen3d.get_layout_widget('vert', widgets, self.spacing)
        )

    def _on_layout(self, layout_context):
        panel_width = 20 * self.spacing
        rect = self.window.content_rect
        self.widget3d.frame = gui.Rect(rect.x, rect.y, rect.get_right() - panel_width, rect.height)
        self.panel.frame = gui.Rect(self.widget3d.frame.get_right(), rect.y, panel_width, rect.height)

    def init_info_widget(self):
        info_layout = gui.CollapsableVert("Log", self.spacing, self.blank_margins)
        self.panel.add_child(info_layout)

        self.console_label = gui.Label("    [info]: Welcome to Use")

        widgets = [FragmentOpen3d.get_widget('label', {'name': 'Console:'}), self.console_label]
        info_layout.add_child(FragmentOpen3d.get_layout_widget('vert', widgets, 0, with_stretch=False))

    def _on_menu_about(self, content="Welcome"):
        dlg = FragmentOpen3d.get_widget('dialog', {'name': 'about'})
        widgets = [
            FragmentOpen3d.get_widget('label', {'name': content}),
            FragmentOpen3d.get_widget('button', {'name': 'cancel'}),
        ]
        widgets[1].set_on_clicked(self.window.close_dialog)
        layout = FragmentOpen3d.get_layout_widget('vert', widgets, self.spacing, self.center_margins)
        dlg.add_child(layout)
        self.window.show_dialog(dlg)

    def add_point_cloud_from_file(self, name: str, file: str, param: dict, is_visible=False):
        try:
            pcd = o3d.io.read_point_cloud(file)
        except Exception as e:
            self.console_label.text = f"[ERROR]: Non valid file: {file}"
            return
        self.add_point_cloud(name, pcd, param, is_visible)

    def add_point_cloud(
            self, name: str, pcd: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh],
            param: dict, is_visible=False
    ):
        if name in self.geoMap.keys():
            if self.geoMap[name]['is_visible']:
                self.widget3d.scene.remove_geometry(name)
            self.geoMap[name].update({'geometry': pcd, 'param': param, 'is_visible': is_visible})
            if is_visible:
                self.widget3d.scene.add_geometry(name, self.geoMap[name]['geometry'], self.materials[AppWindow.LIT])

        else:
            if is_visible:
                self.widget3d.scene.add_geometry(name, pcd, self.materials[AppWindow.LIT])
                self.add_geo_item(name, checked=True)
            else:
                self.add_geo_item(name, checked=False)
            self.geoMap[name] = {
                'name': name, 'style': 'PointCloud', 'is_visible': is_visible, 'geometry': pcd, 'param': param
            }

    def add_mesh_from_file(self, name: str, file: str, param: dict, is_visible=False):
        try:
            # mesh = o3d.io.read_triangle_model(file)  # return triangle model
            mesh = o3d.io.read_triangle_mesh(file)     # return triangle mesh
            mesh.compute_triangle_normals()
        except Exception as e:
            self.console_label.text = f"[ERROR]: Non valid file: {file}"
            return

        # self.add_mesh(name, mesh, param, is_visible)       # when mesh is triangle model
        self.add_point_cloud(name, mesh, param, is_visible)  # when mesh is triangle mesh

    def add_mesh(self, name: str, mesh: o3d.visualization.rendering.TriangleMeshModel, param: dict, is_visible=False):
        if name in self.geoMap.keys():
            if self.geoMap[name]['is_visible']:
                self.widget3d.scene.remove_geometry(name)
            self.geoMap[name].update({'geometry': mesh, 'param': param, 'is_visible': is_visible})
            if is_visible:
                self.widget3d.scene.add_geometry(name, self.geoMap[name]['geometry'])

        else:
            if is_visible:
                self.widget3d.scene.add_model(name, mesh)
                self.add_geo_item(name, checked=True)
            else:
                self.add_geo_item(name, checked=False)
            self.geoMap[name] = {
                'name': name, 'style': 'Mesh', 'is_visible': is_visible, 'geometry': mesh, 'param': param
            }

    def adjust_center_camera(self):
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60, bounds, bounds.get_center())

    def add_geo_item(self, name: str, checked):
        if len(name.strip()) == 0:
            self.console_label.text = "    [Error] Non valid or replicated name."
            return

        def checkbox_on_click(is_checked, box_name):
            if box_name in self.geoMap.keys():
                if self.geoMap[box_name]['is_visible'] != is_checked:
                    if is_checked:
                        if self.geoMap[box_name]['style'] == 'PointCloud':
                            self.widget3d.scene.add_geometry(
                                box_name, self.geoMap[box_name]['geometry'], self.materials[AppWindow.LIT]
                            )
                        elif self.geoMap[box_name]['style'] == 'Mesh':
                            self.widget3d.scene.add_model(box_name, self.geoMap[box_name]['geometry'])
                    else:
                        self.widget3d.scene.remove_geometry(box_name)
                    self.geoMap[box_name]['is_visible'] = is_checked

        checkbox = FragmentOpen3d.get_widget('checkbox', {'name': name})
        checkbox.checked = checked
        checkbox.set_on_checked(partial(checkbox_on_click, box_name=name))
        self.geo_container.add_child(checkbox)
        self.geo_selected_combox.add_item(name)

    def delete_geo_item(self, name: str = None):
        if name is None or len(name.strip()) == 0:
            self.console_label.text = "    [Error] Non valid name."
            return

        if name in self.geoMap.keys():
            is_visible = self.geoMap[name]['is_visible']
            if is_visible:
                self.widget3d.scene.remove_geometry(name)
            del self.geoMap[name]
            self.geo_selected_combox.remove_item(name)


def main():
    app = gui.Application.instance
    app.initialize()

    config = {'width': 1080, 'height': 720}
    window = AppWindow(config=config)

    app.run()
    # sys.exit("GoodBye")


if __name__ == '__main__':
    main()
