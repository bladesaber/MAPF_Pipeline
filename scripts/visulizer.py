import numpy as np
import open3d
import pyvista
from typing import Callable
from tqdm import tqdm

class VisulizerVista(object):
    def __init__(self):
        self.ploter = pyvista.Plotter()

    def plot_pointCloud(self, ploter:pyvista.Plotter, xyzs:np.array, color=(0.1, 0.1, 0.1)):
        pointCloud_mesh = pyvista.PolyData(xyzs)
        ploter.add_mesh(pointCloud_mesh, color=color)
        return pointCloud_mesh

    def plot_tube(self, ploter:pyvista.Plotter, xyzs:np.array, color=(0.1, 0.1, 0.1), radius=0.5):
        pointCloud_mesh = pyvista.PolyData(xyzs)
        the_cell = np.arange(0, xyzs.shape[0], 1)
        the_cell = np.insert(the_cell, 0, xyzs.shape[0])
        pointCloud_mesh.lines = the_cell
        # pointCloud_mesh["scalars"] = np.arange(pointCloud_mesh.n_points)
        tube_mesh = pointCloud_mesh.tube(radius=radius)

        ploter.add_mesh(tube_mesh, color=color)
        return tube_mesh

    def plot_box(self, ploter:pyvista.Plotter, xyz, length=1.0, color=(0.1, 0.1, 0.1)):
        semi_length = length / 2.0
        box = pyvista.Box(bounds=(
            -semi_length, semi_length,
            -semi_length, semi_length,
            -semi_length, semi_length,
        ))
        box.translate(xyz, inplace=True)
        ploter.add_mesh(box, color=color)
        return box

    def plot_many_boxs(self, ploter:pyvista.Plotter, xyzs:np.array, length=1.0, color=(0.1, 0.1, 0.1)):
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
        ploter.add_mesh(boxs_mesh)

        return boxs_mesh

    def add_KeyPressEvent(self, key:str, callback:Callable):
        self.ploter.add_key_event(key, callback)

    def show(self):
        self.ploter.show()
