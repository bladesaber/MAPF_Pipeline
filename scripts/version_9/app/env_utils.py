import numpy as np
import pandas as pd
import math

class Shape_Utils(object):
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

        num = max(math.ceil(2 * radius * np.pi / reso), 1)
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

    @staticmethod
    def create_PipeBoxShell(
            xyz, xmin_dist, xmax_dist, ymin_dist, ymax_dist, zmin_dist, zmax_dist, direction, reso, with_shell
    ):
        xmin, xmax = xyz[0] - xmin_dist, xyz[0] + xmax_dist
        ymin, ymax = xyz[1] - ymin_dist, xyz[1] + ymax_dist
        zmin, zmax = xyz[2] - zmin_dist, xyz[2] + zmax_dist

        pcd = None
        if with_shell:
            assert np.sum(np.abs(direction)) == np.max(np.abs(direction)) == 1.0
            pcd = Shape_Utils.create_BoxPcd(xmin, ymin, zmin, xmax, ymax, zmax, reso)
            if (direction[0] == 1) or (direction[0] == -1):
                pcd = pcd[pcd[:, 0] < xmax]
                pcd = pcd[pcd[:, 0] > xmin]

            elif (direction[1] == 1) or (direction[1] == -1):
                pcd = pcd[pcd[:, 1] < ymax]
                pcd = pcd[pcd[:, 1] > ymin]

            elif (direction[2] == 1) or (direction[2] == -1):
                pcd = pcd[pcd[:, 2] < zmax]
                pcd = pcd[pcd[:, 2] > zmin]

            else:
                raise ValueError

        return pcd, (xmin, xmax, ymin, ymax, zmin, zmax)

    @staticmethod
    def create_PipeCylinderShell(xyz, radius, hmin_dist, hmax_dist, direction, reso, with_shell):
        pcd = None
        if with_shell:
            height = hmin_dist + hmax_dist

            num = max(math.ceil(2 * radius * np.pi / reso), 1)
            hSteps = max(math.ceil(height / reso), 2)
            rads = np.deg2rad(np.linspace(0, 360.0, num))
            huv = np.zeros(shape=(num, 2))
            huv[:, 0] = np.cos(rads) * radius
            huv[:, 1] = np.sin(rads) * radius

            if direction[0] == 1:
                pcds = []
                for h_value in np.linspace(xyz[0] - hmin_dist, xyz[0] + hmax_dist, hSteps):
                    hPcd = np.zeros(shape=(num, 3))
                    hPcd[:, 0] = h_value
                    hPcd[:, 1] = huv[:, 0] + xyz[1]
                    hPcd[:, 2] = huv[:, 1] + xyz[2]
                    pcds.append(hPcd)

            elif direction[1] == 1:
                pcds = []
                for h_value in np.linspace(xyz[1] - hmin_dist, xyz[1] + hmax_dist, hSteps):
                    hPcd = np.zeros(shape=(num, 3))
                    hPcd[:, 0] = huv[:, 0] + xyz[0]
                    hPcd[:, 1] = h_value
                    hPcd[:, 2] = huv[:, 1] + xyz[2]
                    pcds.append(hPcd)

            elif direction[2] == 1:
                pcds = []
                for h_value in np.linspace(xyz[2] - hmin_dist, xyz[2] + hmax_dist, hSteps):
                    hPcd = np.zeros(shape=(num, 3))
                    hPcd[:, 0] = huv[:, 0] + xyz[0]
                    hPcd[:, 1] = huv[:, 1] + xyz[1]
                    hPcd[:, 2] = h_value
                    pcds.append(hPcd)

        return pcd, (xyz, radius, hmin_dist, hmax_dist)

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
    def removePointInSphereShell(
            xyzs, center, radius, with_bound=False, direction=None, reso=0.1, scale_dist=None
    ):
        distance = np.linalg.norm(xyzs - center, ord=2, axis=1)
        xyzs = xyzs[distance > radius + 0.1]

        if with_bound:
            scale_radius = radius + scale_dist
            num = max(math.ceil(2 * scale_radius * np.pi / reso), 1)
            rads = np.deg2rad(np.linspace(0, 360.0, num))
            rads = rads[:-1]  # 0 is the same as 360
            huv = np.zeros(shape=(rads.shape[0], 2))
            huv[:, 0] = np.cos(rads) * scale_radius
            huv[:, 1] = np.sin(rads) * scale_radius

            hPcd = np.zeros(shape=(rads.shape[0], 3))
            if direction[0] == 1:
                hPcd[:, 0] = center[0]
                hPcd[:, 1] = huv[:, 0] + center[1]
                hPcd[:, 2] = huv[:, 1] + center[2]
            elif direction[1] == 1:
                hPcd[:, 0] = huv[:, 0] + center[0]
                hPcd[:, 1] = center[1]
                hPcd[:, 2] = huv[:, 1] + center[2]
            else:
                hPcd[:, 0] = huv[:, 0] + center[0]
                hPcd[:, 1] = huv[:, 1] + center[1]
                hPcd[:, 2] = center[2]

            xyzs = np.concatenate([xyzs, hPcd], axis=0)

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


class GridEnv_Utils(object):
    @staticmethod
    def obstacle_convert_grid(xyzs:np.array, scale):
        scale_xyzs = xyzs * scale
        scale_xyzs = np.round(scale_xyzs, decimals=1)
        scale_xyzs = pd.DataFrame(scale_xyzs).drop_duplicates().values
        return scale_xyzs

    @staticmethod
    def pipe_convert_grid(xyz, scale, xmin, xmax, ymin, ymax, zmin, zmax):
        scale_xyz = xyz * scale
        xyz_grid = np.round(scale_xyz, decimals=0)
        xyz_grid[0] = np.maximum(np.minimum(xyz_grid[0], xmax), xmin)
        xyz_grid[1] = np.maximum(np.minimum(xyz_grid[1], ymax), ymin)
        xyz_grid[2] = np.maximum(np.minimum(xyz_grid[2], zmax), zmin)
        return scale_xyz, xyz_grid
