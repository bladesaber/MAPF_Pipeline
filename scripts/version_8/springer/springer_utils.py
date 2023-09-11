import numpy as np
import pandas as pd
import math
import pyvista

def get_rotateMat(xyzTag, radian):
    if xyzTag == "X":
        rotMat = np.array([
            [1.,             0.,              0.],
            [0., np.cos(radian), -np.sin(radian)],
            [0., np.sin(radian),  np.cos(radian)]
        ])

    elif xyzTag == "Y":
        rotMat = np.array([
            [np.cos(radian), 0., np.sin(radian)],
            [0.,              1.,             0.],
            [-np.sin(radian), 0., np.cos(radian)]
        ])

    elif xyzTag == "Z":
        rotMat = np.array([
            [np.cos(radian), -np.sin(radian), 0.],
            [np.sin(radian), np.cos(radian),  0.],
            [0.,             0.,              1.]
        ])

    else:
        rotMat = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])

    return rotMat


class ShapePcdUtils(object):
    @staticmethod
    def create_PlanePcd(xyzTag, xmin, ymin, zmin, xmax, ymax, zmax, reso):
        xSteps = math.ceil((xmax - xmin) / reso)
        ySteps = math.ceil((ymax - ymin) / reso)
        zSteps = math.ceil((zmax - zmin) / reso)

        if xyzTag == "xmin":
            yWall = np.linspace(ymin, ymax, ySteps)
            zWall = np.linspace(zmin, zmax, zSteps)
            ys, zs = np.meshgrid(yWall, zWall)
            yzs = np.concatenate([ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1).reshape((-1, 2))
            wall = np.concatenate([np.ones((yzs.shape[0], 1)) * xmin, yzs], axis=1)

        elif xyzTag == "xmax":
            yWall = np.linspace(ymin, ymax, ySteps)
            zWall = np.linspace(zmin, zmax, zSteps)
            ys, zs = np.meshgrid(yWall, zWall)
            yzs = np.concatenate([ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1).reshape((-1, 2))
            wall = np.concatenate([np.ones((yzs.shape[0], 1)) * xmax, yzs], axis=1)

        elif xyzTag == "ymin":
            xWall = np.linspace(xmin, xmax, xSteps)
            zWall = np.linspace(zmin, zmax, zSteps)
            xs, zs = np.meshgrid(xWall, zWall)
            xzs = np.concatenate([xs[..., np.newaxis], zs[..., np.newaxis]], axis=-1).reshape((-1, 2))
            wall = np.concatenate([xzs[:, :1], np.ones((xzs.shape[0], 1)) * ymin, xzs[:, -1:]], axis=1)

        elif xyzTag == "ymax":
            xWall = np.linspace(xmin, xmax, xSteps)
            zWall = np.linspace(zmin, zmax, zSteps)
            xs, zs = np.meshgrid(xWall, zWall)
            xzs = np.concatenate([xs[..., np.newaxis], zs[..., np.newaxis]], axis=-1).reshape((-1, 2))
            wall = np.concatenate([xzs[:, :1], np.ones((xzs.shape[0], 1)) * ymax, xzs[:, -1:]], axis=1)

        elif xyzTag == "zmin":
            xWall = np.linspace(xmin, xmax, xSteps)
            yWall = np.linspace(ymin, ymax, ySteps)
            xs, ys = np.meshgrid(xWall, yWall)
            xys = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1).reshape((-1, 2))
            wall = np.concatenate([xys, np.ones((xys.shape[0], 1)) * zmin], axis=1)

        elif xyzTag =='zmax':
            xWall = np.linspace(xmin, xmax, xSteps)
            yWall = np.linspace(ymin, ymax, ySteps)
            xs, ys = np.meshgrid(xWall, yWall)
            xys = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1).reshape((-1, 2))
            wall = np.concatenate([xys, np.ones((xys.shape[0], 1)) * zmax], axis=1)

        else:
            raise NotImplementedError

        return wall

    @staticmethod
    def create_ShapePcd(pose: np.array, radian, xyzTag, shapePcd: np.array):
        rotMat = get_rotateMat(xyzTag, radian)
        shapePcd_w = (rotMat.dot(shapePcd.T)).T + pose
        return shapePcd_w

    @staticmethod
    def create_BoxPcd(xmin, ymin, zmin, xmax, ymax, zmax, reso):
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
    def create_CylinderPcd(xyz, radius, height, direction, reso):
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

        num = max(math.ceil( 2 * radius * np.pi / reso), 1)
        rads = np.deg2rad(np.linspace(0, 360.0, num))
        huv = np.zeros(shape=(num, 2))
        huv[:, 0] = np.cos(rads) * radius
        huv[:, 1] = np.sin(rads) * radius

        if direction[0] == 1:
            pcds = [
                np.concatenate([
                    np.ones(shape=(uvs.shape[0], 1)) * height / 2.0,
                    uvs[:, 0:1],
                    uvs[:, 1:2],
                    ], axis=1),
                np.concatenate([
                    np.ones(shape=(uvs.shape[0], 1)) * -height / 2.0,
                    uvs[:, 0:1],
                    uvs[:, 1:2],
                    ], axis=1),
            ]

            for h_value in np.linspace(-height/2.0, height/2.0, hSteps):
                hPcd = np.zeros(shape=(num, 3))
                hPcd[:, 0] = h_value
                hPcd[:, 1] = huv[:, 0]
                hPcd[:, 2] = huv[:, 1]
                pcds.append(hPcd)

        elif direction[1] == 1:
            pcds = [
                np.concatenate([
                    uvs[:, 0:1],
                    np.ones(shape=(uvs.shape[0], 1)) * -height / 2.0,
                    uvs[:, 1:2],
                    ], axis=1),
                np.concatenate([
                    uvs[:, 0:1],
                    np.ones(shape=(uvs.shape[0], 1)) * height / 2.0,
                    uvs[:, 1:2],
                    ], axis=1),
            ]

            for h_value in np.linspace(-height/2.0, height/2.0, hSteps):
                hPcd = np.zeros(shape=(num, 3))
                hPcd[:, 0] = huv[:, 0]
                hPcd[:, 1] = h_value
                hPcd[:, 2] = huv[:, 1]
                pcds.append(hPcd)

        elif direction[2] == 1:
            pcds = [
                np.concatenate([
                    uvs[:, 0:1],
                    uvs[:, 1:2],
                    np.ones(shape=(uvs.shape[0], 1)) * -height / 2.0,
                    ], axis=1),
                np.concatenate([
                    uvs[:, 0:1],
                    uvs[:, 1:2],
                    np.ones(shape=(uvs.shape[0], 1)) * height / 2.0,
                    ], axis=1),
            ]

            for h_value in np.linspace(-height / 2.0, height / 2.0, hSteps):
                hPcd = np.zeros(shape=(num, 3))
                hPcd[:, 0] = huv[:, 0]
                hPcd[:, 1] = huv[:, 1]
                hPcd[:, 2] = h_value
                pcds.append(hPcd)

        else:
            raise NotImplementedError

        pcd = np.concatenate(pcds, axis=0)
        pcd = pcd + xyz

        return pcd

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
    def create_complex_tube(xyzs:np.array, capping, radius, scalars=None):
        pointCloud_mesh = pyvista.PolyData(xyzs)
        the_cell = np.arange(0, xyzs.shape[0], 1)
        the_cell = np.insert(the_cell, 0, xyzs.shape[0])
        pointCloud_mesh.lines = the_cell
        if scalars is not None:
            pointCloud_mesh["scalars"] = scalars
            tube_mesh = pointCloud_mesh.tube(
                radius=radius, capping=capping, scalars='scalars', # radius_factor=2.0,
                absolute=True
            )
        else:
            tube_mesh = pointCloud_mesh.tube(radius=radius, capping=capping)
        return tube_mesh

    @staticmethod
    def create_sphere(xyz, radius):
        sphere = pyvista.Sphere(radius, center=xyz)
        return sphere

    @staticmethod
    def create_arrow(pose: np.array, vec: np.array, tip_length, tip_radius, shaft_radius):
        arrow = pyvista.Arrow(pose, vec, tip_length=tip_length, tip_radius=tip_radius, shaft_radius=shaft_radius)
        return arrow

    def addAxes(self, tip_length, tip_radius, shaft_radius):
        self.ploter.add_mesh(
            VisulizerVista.create_arrow(
                np.array([0., 0., 0.]), np.array([1., 0., 0.]), tip_length, tip_radius, shaft_radius
            ), color=(1.0, 0.0, 0.0)
        )
        self.ploter.add_mesh(
            VisulizerVista.create_arrow(
                np.array([0., 0., 0.]), np.array([0., 1., 0.]), tip_length, tip_radius, shaft_radius
            ), color=(0.0, 1.0, 0.0)
        )
        self.ploter.add_mesh(
            VisulizerVista.create_arrow(
                np.array([0., 0., 0.]), np.array([0., 0., 1.]), tip_length, tip_radius, shaft_radius
            ), color=(0.0, 0.0, 1.0)
        )

    def plot(self, mesh, color=(0.5, 0.1, 0.8), opacity=1.0, style=None, show_edges=False):
        self.ploter.add_mesh(mesh, color=color, opacity=opacity, style=style, show_edges=show_edges)

    def show(self):
        self.ploter.show()