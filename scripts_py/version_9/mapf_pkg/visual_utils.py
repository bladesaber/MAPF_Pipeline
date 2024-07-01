import pyvista
import numpy as np
from typing import List, Union


class VisUtils(object):
    def __init__(self):
        self.plotter: pyvista.Plotter = pyvista.Plotter()
        self.plotter.set_background('white')

    def show(self, dim: int = None):
        if dim is not None:
            if dim == 2:
                self.plotter.view_xy()
        self.plotter.show()

    def plot(
            self, mesh, color=(0.5, 0.1, 0.8), opacity=1.0,
            style=None, show_edges=False, show_scalar_bar=False,
            point_size=None, line_width=None
    ):
        self.plotter.add_mesh(
            mesh, color=color, opacity=opacity, style=style,
            show_edges=show_edges, show_scalar_bar=show_scalar_bar,
            point_size=point_size, line_width=line_width
        )

    @staticmethod
    def create_point_cloud(xyz: np.ndarray):
        mesh = pyvista.PolyData(xyz)
        return mesh

    @staticmethod
    def create_box(xyz, length=1.0):
        semi_length = length * 0.5
        mesh = pyvista.Box(bounds=(-semi_length, semi_length, -semi_length, semi_length, -semi_length, semi_length))
        mesh.translate(xyz, inplace=True)
        return mesh

    @staticmethod
    def create_sphere(xyz, radius):
        mesh = pyvista.Sphere(radius, center=xyz)
        return mesh

    @staticmethod
    def create_cylinder(xyz, direction, radius, height):
        mesh = pyvista.Cylinder(xyz, direction, radius, height)
        return mesh

    @staticmethod
    def create_line_set(line_idxs: Union[List, np.ndarray]):
        """
        line_idx: [idx_of_point1, idx_of_point2, ...]
        """
        line_set = []
        for i in range(1, len(line_idxs), 1):
            line_set.append([2, line_idxs[i - 1], line_idxs[i]])  # 2 means that in this line contain 2 points
        line_set = np.array(line_set)
        return line_set

    @staticmethod
    def create_line(pcd: np.ndarray, lines_set: np.ndarray):
        line_mesh = pyvista.PolyData(pcd)
        line_mesh.lines = lines_set
        return line_mesh

    @staticmethod
    def create_sphere_set(pcd: np.ndarray, radius):
        meshes = []
        for xyz in pcd:
            meshes.append(VisUtils.create_sphere(xyz, radius))
        meshes = pyvista.MultiBlock(meshes)
        return meshes

    @staticmethod
    def create_tube(pcd: np.ndarray, radius, lines_set: np.ndarray) -> pyvista.PolyData:
        pcd_mesh = pyvista.PolyData(pcd)
        pcd_mesh.lines = lines_set
        tube_mesh = pcd_mesh.tube(radius=radius)
        return tube_mesh

    @staticmethod
    def create_deformation_tube(pcd: np.ndarray, lines_set: np.ndarray, radius, scalars: np.ndarray, capping=True):
        pcd_mesh = pyvista.PolyData(pcd)
        pcd_mesh.lines = lines_set
        pcd_mesh["scalars"] = scalars
        mesh = pcd_mesh.tube(radius=radius, capping=capping, scalars='scalars', absolute=True)
        return mesh

    @staticmethod
    def multi_tube_example():
        pcd = np.array([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 2., 0.],
            [0., 3., 0.],
            [0., 4., 0.],
            [1., 2., 0.],
            [2., 2., 0.],
            [3., 2., 0.]
        ])
        lines_connect = [
            [0, 1, 2, 3, 4],
            [2, 5, 6, 7]
        ]
        lines_set = []
        for sep_line in lines_connect:
            lines_set.append(VisUtils.create_line_set(sep_line))
        lines_set = np.concatenate(lines_set, axis=0)
        # scalars = np.random.uniform(0.05, 0.45, size=(pcd.shape[0],))
        scalars = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.05, 0.05, 0.05])

        mesh = VisUtils.create_deformation_tube(pcd, radius=0.25, lines_set=lines_set, scalars=scalars)
        mesh.plot()

        # mesh = VisUtils.create_tube(pcd, radius=0.25, lines_set=lines_set)
        # mesh.plot()

    @staticmethod
    def multi_tube_union_example():
        """
        todo Fail example
        """
        pcd = np.array([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 2., 0.],
            [0., 3., 0.],
            [0., 4., 0.],
            [1., 2., 0.],
            [2., 2., 0.],
            [3., 2., 0.]
        ])
        lines_connect = [
            [0, 1, 2, 3, 4],
            [2, 5, 6, 7]
        ]
        radius = [0.25, 0.05]

        meshes = []
        for i in range(len(lines_connect)):
            line_set = VisUtils.create_line_set(lines_connect[i])
            mesh = VisUtils.create_tube(pcd, radius=radius[i], lines_set=line_set)
            meshes.append(mesh)

        vis = VisUtils()

        # for mesh in meshes:
        #     vis.plot(mesh, style='wireframe')

        main_mesh = meshes[0]
        for mesh in meshes[1:]:
            main_mesh.boolean_union(mesh)
        vis.plot(main_mesh, style='wireframe')

        vis.show()

