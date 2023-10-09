import numpy as np
import math
import pyvista

def get_rotationMat(xyzTag, radian):
    if xyzTag == "X":
        rotMat = np.array([
            [1., 0., 0.],
            [0., np.cos(radian), -np.sin(radian)],
            [0., np.sin(radian), np.cos(radian)],
        ])
    elif xyzTag == "Y":
        rotMat = np.array([
            [np.cos(radian), 0., np.sin(radian)],
            [0., 1., 0.],
            [-np.sin(radian), 0., np.cos(radian)],
        ])
    elif xyzTag == "Z":
        rotMat = np.array([
            [np.cos(radian), -np.sin(radian), 0.],
            [np.sin(radian), np.cos(radian), 0.],
            [0., 0., 1.],
        ])
    else:
        raise NotImplementedError

    return rotMat


class ConnectVisulizer(object):
    def __init__(self):
        self.ploter = pyvista.Plotter()
        # self.ploter.set_background('white', top='black')
        self.ploter.set_background('white')

    def plot_connect(self, xyzs: np.array, color, opacity=1.0):
        line_mesh = pyvista.PolyData(xyzs)
        the_cell = np.arange(0, xyzs.shape[0], 1)
        the_cell = np.insert(the_cell, 0, xyzs.shape[0])
        line_mesh.lines = the_cell
        self.plot(line_mesh, color=color, opacity=opacity)

    def plot_tube(self, xyzs: np.array, radius, color, opacity=0.2):
        pointCloud_mesh = pyvista.PolyData(xyzs)
        the_cell = np.arange(0, xyzs.shape[0], 1)
        the_cell = np.insert(the_cell, 0, xyzs.shape[0])
        pointCloud_mesh.lines = the_cell
        tube_mesh = pointCloud_mesh.tube(radius=radius)
        self.plot(tube_mesh, color=color, opacity=opacity)

    def plot_cell(self, xyz, radius, color, opacity=1.0):
        sphere = pyvista.Sphere(radius, center=xyz)
        self.plot(sphere, color=color, opacity=opacity)

    def plot_structor(self, xyz, radius, shape_xyzs, color, opacity=1.0, with_center=True):
        if with_center:
            sphere = pyvista.Sphere(radius, center=xyz)
            self.plot(sphere, color=color, opacity=opacity)
        shape_mesh = pyvista.PolyData(shape_xyzs)
        self.plot(shape_mesh, color=color, opacity=opacity)

    def plot_bound(self, xmax, ymax, zmax, color, opacity=0.1):
        bound = pyvista.Box((0., xmax, 0., ymax, 0., zmax))
        self.plot(bound, color=color, opacity=opacity)

    def plot(self, mesh, color=(0.5, 0.1, 0.8), opacity=1.0, style=None, show_edges=False):
        self.ploter.add_mesh(mesh, color=color, opacity=opacity, style=style, show_edges=show_edges)

    def show(self):
        self.ploter.show()