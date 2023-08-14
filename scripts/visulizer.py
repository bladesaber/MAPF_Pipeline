import numpy as np
import open3d as o3d
import pyvista
from typing import Callable
from tqdm import tqdm

from scripts.utils import polar2RotMatrix

class VisulizerVista(object):
    def __init__(self):
        self.ploter = pyvista.Plotter()
        # self.ploter.set_background('white', top='black')
        self.ploter.set_background('white')

    @staticmethod
    def create_pointCloud(xyzs:np.array):
        pointCloud_mesh = pyvista.PolyData(xyzs)
        return pointCloud_mesh

    @staticmethod
    def create_tube(xyzs:np.array, radius=0.5):
        pointCloud_mesh = pyvista.PolyData(xyzs)
        the_cell = np.arange(0, xyzs.shape[0], 1)
        the_cell = np.insert(the_cell, 0, xyzs.shape[0])
        pointCloud_mesh.lines = the_cell
        # pointCloud_mesh["scalars"] = np.arange(pointCloud_mesh.n_points)
        tube_mesh = pointCloud_mesh.tube(radius=radius)
        return tube_mesh
    
    @staticmethod
    def create_box(xyz, length=1.0):
        semi_length = length / 2.0
        box = pyvista.Box(bounds=(
            -semi_length, semi_length,
            -semi_length, semi_length,
            -semi_length, semi_length,
        ))
        box.translate(xyz, inplace=True)
        return box

    @staticmethod
    def create_sphere(xyz, radius):
        sphere = pyvista.Sphere(radius, center=xyz)
        return sphere

    @staticmethod
    def create_many_boxs(xyzs:np.array, length=1.0):
        semi_length = length / 2.0

        boxs_mesh = []
        for xyz in tqdm(xyzs):
            box = pyvista.Box(bounds=(
                -semi_length, semi_length,
                -semi_length, semi_length,
                -semi_length, semi_length,
            ))
            box.translate(xyz, inplace=True)
            boxs_mesh.append(box)

        boxs_mesh = pyvista.MultiBlock(boxs_mesh)
        return boxs_mesh

    @staticmethod
    def create_complex_tube(xyzs:np.array, capping, radius, scalars=None):
        pointCloud_mesh = pyvista.PolyData(xyzs)
        the_cell = np.arange(0, xyzs.shape[0], 1)
        the_cell = np.insert(the_cell, 0, xyzs.shape[0])
        pointCloud_mesh.lines = the_cell
        if scalars is not None:
            pointCloud_mesh["scalars"] = scalars
            tube_mesh = pointCloud_mesh.tube(
                radius=radius, capping=capping, scalars='scalars', 
                # radius_factor=2.0,
                absolute=True
            )
        else:
            tube_mesh = pointCloud_mesh.tube(radius=radius, capping=capping)
        return tube_mesh

    @staticmethod
    def create_line(xyzs):
        line_mesh = pyvista.PolyData(xyzs)
        the_cell = np.arange(0, xyzs.shape[0], 1)
        the_cell = np.insert(the_cell, 0, xyzs.shape[0])
        line_mesh.lines = the_cell
        return line_mesh

    def plot(self, mesh, color=(0.5, 0.1, 0.8), opacity=1.0, style=None, show_edges=False):
        self.ploter.add_mesh(mesh, color=color, opacity=opacity, style=style, show_edges=show_edges)

    def show(self):
        self.ploter.show()

    def add_KeyPressEvent(self, key: str, callback: Callable):
        self.ploter.add_key_event(key, callback)

    @staticmethod
    def read_file(path):
        return pyvista.read_meshio(path)

class VisulizerO3D(object):
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(height=720, width=960)

        coodr = self.create_Axis(size=5.0)
        self.vis.add_geometry(coodr, reset_bounding_box=True)

        self.vis.create_window(height=720, width=960)

    def create_Arrow(
            self, x, y, z, alpha, beta,
            cylinder_radius=0.05, cone_radius=0.2, cylinder_height=1.0, cone_height=0.25
        ):
        rot_mat = polar2RotMatrix(alpha, beta)

        arrow: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cylinder_radius, 
            cone_radius=cone_radius,
            cylinder_height=cylinder_height,
            cone_height=cone_height,
        )
        arrow.rotate(rot_mat, center=np.array([0., 0., 0.]))
        arrow.translate(np.array([x, y, z]))
        return arrow

    def create_Axis(self, size=1.0, origin=np.array([0., 0., 0.])):
        coodr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        return coodr

    def create_Sphere(self, x, y, z, radius):
        sphere: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(np.array([x, y, z]))
        return sphere

    def addArrow(self, pos, color):
        arrow = self.create_Arrow(pos[0], pos[1], pos[2], pos[3], pos[4])
        arrow.paint_uniform_color(color)
        self.vis.add_geometry(arrow, reset_bounding_box=False)

    def addConstrain(self, x, y, z, radius):
        sphere = self.create_Sphere(x, y, z, radius)
        sphere.paint_uniform_color(np.array([0.0, 0.0, 0.0]))
        self.vis.add_geometry(sphere, reset_bounding_box=False)

    def addPathArrow(self, xyz_thetas, color):
        for x, y, z, alpha, beta in xyz_thetas:
            arrow: o3d.geometry.TriangleMesh = self.create_Arrow(x, y, z, alpha, beta)
            arrow.paint_uniform_color(color)
            self.vis.add_geometry(arrow, reset_bounding_box=False)

    def addPathPoint(self, xyzs, color):
        dist = xyzs[1:, :] - xyzs[:-1, :]
        xyzs = np.concatenate([
            xyzs,
            xyzs[:-1, :] + 0.25 * dist,
            xyzs[:-1, :] + 0.50 * dist,
            xyzs[:-1, :] + 0.75 * dist
        ], axis=0)

        path_pcd = o3d.geometry.PointCloud()
        path_pcd.points = o3d.utility.Vector3dVector(xyzs)

        color_np = np.tile(color.reshape((1, -1)), [xyzs.shape[0], 1])
        path_pcd.colors = o3d.utility.Vector3dVector(color_np)

        self.vis.add_geometry(path_pcd, reset_bounding_box=False)

    def show(self):
        self.vis.run()
        self.vis.destroy_window()

if __name__ == '__main__':
    pass