import numpy as np
import open3d
import pyvista
from typing import Callable
from tqdm import tqdm

class VisulizerVista(object):
    def __init__(self):
        self.ploter = pyvista.Plotter()

    def create_pointCloud(self, xyzs:np.array):
        pointCloud_mesh = pyvista.PolyData(xyzs)
        return pointCloud_mesh

    def create_tube(self, xyzs:np.array, radius=0.5):
        pointCloud_mesh = pyvista.PolyData(xyzs)
        the_cell = np.arange(0, xyzs.shape[0], 1)
        the_cell = np.insert(the_cell, 0, xyzs.shape[0])
        pointCloud_mesh.lines = the_cell
        # pointCloud_mesh["scalars"] = np.arange(pointCloud_mesh.n_points)
        tube_mesh = pointCloud_mesh.tube(radius=radius)
        return tube_mesh

    def create_box(self, xyz, length=1.0):
        semi_length = length / 2.0
        box = pyvista.Box(bounds=(
            -semi_length, semi_length,
            -semi_length, semi_length,
            -semi_length, semi_length,
        ))
        box.translate(xyz, inplace=True)
        return box

    def create_many_boxs(self, xyzs:np.array, length=1.0):
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

    def plot(self, mesh, color=(0.5, 0.1, 0.8)):
        self.ploter.add_mesh(mesh, color=color)

    def show(self):
        self.ploter.show()

    def add_KeyPressEvent(self, key: str, callback: Callable):
        self.ploter.add_key_event(key, callback)

class InterativeVista(VisulizerVista):
    def __init__(self):
        super(InterativeVista, self).__init__()
        self.add_KeyPressEvent('p', self.update_tube)

    def update_tube(self):
        xyzs = np.array(np.random.randint(0, 4, size=(10, 3)))


if __name__ == '__main__':
    vis = InterativeVista()
    xyzs = np.array([
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 0., 2.],
        [0., 0., 3.],
        [0., 1., 3.],
        [0., 2., 3.],
        [0., 3., 3.],
        [1., 3., 3.],
        [2., 3., 3.],
        [3., 3., 3.],
    ])
    tube_mesh = vis.create_tube(xyzs)
    print(tube_mesh)

    # xyzs = np.array(np.random.randint(0, 4, size=(10, 3))).astype(np.float)

    # vis.plot(tube_mesh)
    # vis.show()