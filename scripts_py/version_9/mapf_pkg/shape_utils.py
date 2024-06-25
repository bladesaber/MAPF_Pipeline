import numpy as np
import pandas as pd
import math
from scipy.spatial import transform
from typing import List, Literal, Union
import pyvista


class AxisTransform(object):
    @staticmethod
    def euler_angles_to_rotation_mat(angles: Union[List[float], np.ndarray], is_degree=True, seq='xyz'):
        # Z      Y
        # |    /
        # |   /
        # |  /
        # | /
        # |/__________ X
        return transform.Rotation.from_euler(seq=seq, angles=angles, degrees=is_degree).as_matrix()

    @staticmethod
    def vector_to_rotation_mat(vector: np.ndarray):
        vector = vector / np.linalg.norm(vector, ord=2)
        gamma = math.atan2(vector[1], vector[0])
        Rz = transform.Rotation.from_euler(seq='xyz', angles=np.array([0.0, 0.0, gamma]), degrees=False).as_matrix()
        vector = np.linalg.inv(Rz) @ vector.reshape((-1, 1))
        vector = vector.reshape(-1)
        beta = math.atan2(vector[0], vector[2])
        Ry = transform.Rotation.from_euler(seq='xyz', angles=np.array([0.0, beta, 0.0]), degrees=False).as_matrix()
        rot_mat = Rz @ Ry
        return rot_mat

    @staticmethod
    def pcd_transform(pcd: np.ndarray, rotate_mat: np.ndarray):
        return rotate_mat.dot(pcd.T).T

    @staticmethod
    def norm_to_rotation_mat(vector: np.ndarray):
        axis_x = np.array([1., 0.])
        length_xy = np.linalg.norm(vector[:2], ord=2)
        if length_xy == 0:
            return AxisTransform.euler_angles_to_rotation_mat(
                [0, np.sign(vector[-1]) * 90, 0], is_degree=True, seq='xyz'
            )

        angel_0 = np.rad2deg(np.arccos(np.sum(axis_x * vector[:2]) / length_xy))
        angel_1 = np.rad2deg(np.arctan(vector[2] / length_xy))
        return AxisTransform.euler_angles_to_rotation_mat([0, angel_0, angel_1], is_degree=True, seq='xzy')


class ShapeUtils(object):
    @staticmethod
    def create_box_pcd(xmin, ymin, zmin, xmax, ymax, zmax, reso, is_solid=False):
        x_wall = np.linspace(xmin, xmax, math.ceil((xmax - xmin) / reso))
        y_wall = np.linspace(ymin, ymax, math.ceil((ymax - ymin) / reso))
        z_wall = np.linspace(zmin, zmax, math.ceil((zmax - zmin) / reso))

        if is_solid:
            xs, ys, zs = np.meshgrid(x_wall, y_wall, z_wall)
            pcd = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1)
            pcd = pcd.reshape((-1, 3))
            pcd = pd.DataFrame(pcd).drop_duplicates().values

        else:
            ys, zs = np.meshgrid(y_wall, z_wall)
            yzs = np.concatenate([ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1).reshape((-1, 2))
            xs, zs = np.meshgrid(x_wall, z_wall)
            xzs = np.concatenate([xs[..., np.newaxis], zs[..., np.newaxis]], axis=-1).reshape((-1, 2))
            xs, ys = np.meshgrid(x_wall, y_wall)
            xys = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1).reshape((-1, 2))

            xmin_wall = np.concatenate([np.full((yzs.shape[0], 1), xmin), yzs], axis=1)
            xmax_wall = np.concatenate([np.full((yzs.shape[0], 1), xmax), yzs], axis=1)
            ymin_wall = np.concatenate([xzs[:, :1], np.full((xzs.shape[0], 1), ymin), xzs[:, -1:]], axis=1)
            ymax_wall = np.concatenate([xzs[:, :1], np.full((xzs.shape[0], 1), ymax), xzs[:, -1:]], axis=1)
            zmin_wall = np.concatenate([xys, np.full((xys.shape[0], 1), zmin)], axis=1)
            zmax_wall = np.concatenate([xys, np.full((xys.shape[0], 1), zmax)], axis=1)

            pcd = np.concatenate([xmin_wall, xmax_wall, ymin_wall, ymax_wall, zmin_wall, zmax_wall], axis=0)
            pcd = pd.DataFrame(pcd).drop_duplicates().values

        return pcd

    @staticmethod
    def create_cylinder_pcd(center, radius, height, angles: List[float], reso, is_solid):
        u_steps, h_steps = math.ceil(radius / reso), max(math.ceil(height / reso), 2)

        # step 1: create cylinder along X Axis
        uvs = []
        for cell_radius in np.linspace(0, radius, u_steps):
            length = 2 * cell_radius * np.pi
            num = max(math.ceil(length / reso), 1)
            rads = np.deg2rad(np.linspace(0, 360.0, num))
            uv = np.zeros(shape=(num, 2))
            uv[:, 0] = np.cos(rads) * cell_radius
            uv[:, 1] = np.sin(rads) * cell_radius
            uvs.append(uv)
        uvs = np.concatenate(uvs, axis=0)

        if is_solid:
            pcds = []
            for h_value in np.linspace(-height / 2.0, height / 2.0, h_steps):
                pcds.append(np.concatenate([np.full((uvs.shape[0], 1), h_value), uvs[:, 0:1], uvs[:, 1:2]], axis=1))
            pcd = np.concatenate(pcds, axis=0)

        else:
            num = max(math.ceil(2 * radius * np.pi / reso), 1)
            rads = np.deg2rad(np.linspace(0, 360.0, num))
            huv = np.zeros(shape=(num, 2))
            huv[:, 0] = np.cos(rads) * radius
            huv[:, 1] = np.sin(rads) * radius

            pcds = [
                np.concatenate([np.full((uvs.shape[0], 1), height / 2.0), uvs[:, 0:1], uvs[:, 1:2]], axis=1),
                np.concatenate([np.full((uvs.shape[0], 1), -height / 2.0), uvs[:, 0:1], uvs[:, 1:2]], axis=1),
            ]
            for h_value in np.linspace(-height / 2.0, height / 2.0, h_steps):
                h_pcd = np.zeros(shape=(num, 3))
                h_pcd[:, 0] = h_value
                h_pcd[:, 1] = huv[:, 0]
                h_pcd[:, 2] = huv[:, 1]
                pcds.append(h_pcd)
            pcd = np.concatenate(pcds, axis=0)

        rot_mat = AxisTransform.euler_angles_to_rotation_mat(angles)
        pcd = AxisTransform.pcd_transform(pcd, rot_mat)
        pcd = pcd + center
        return pcd

    @staticmethod
    def create_sphere_pcd(center: np.ndarray, vector: np.ndarray, radius, reso):
        pcd = []
        x_step = max(math.ceil(radius * 2 / reso), 5)
        for h in np.linspace(-radius, +radius, x_step):
            y_radius = np.sqrt(np.power(radius, 2) - np.power(h, 2))
            if y_radius < 1e-4:
                r_num = 1
            else:
                r_num = max(math.ceil(2.0 * y_radius * np.pi / reso), 4)
            rads = np.deg2rad(np.linspace(0, 360.0, r_num))
            sub_pcd = np.zeros(shape=(r_num, 3))
            sub_pcd[:, 0] = h
            sub_pcd[:, 1] = np.cos(rads) * y_radius
            sub_pcd[:, 2] = np.sin(rads) * y_radius
            pcd.append(sub_pcd)
        pcd = np.concatenate(pcd, axis=0)
        # rot_mat = AxisTransform.euler_angles_to_rotation_mat(angles)
        rot_mat = AxisTransform.norm_to_rotation_mat(vector)
        pcd = AxisTransform.pcd_transform(pcd, rot_mat)
        pcd = pcd + center
        return pcd

    @staticmethod
    def remove_obstacle_point_in_pipe(
            obs_pcd: np.ndarray, center: np.ndarray, direction: np.ndarray, radius, tol_scale, is_input
    ):
        dist_0 = np.linalg.norm(obs_pcd - center, ord=2, axis=1)
        dist_1 = ShapeUtils.pcd2plane_dist(obs_pcd, center, direction)
        if is_input:
            non_valid = np.bitwise_or(
                dist_0 <= radius + 1e-3,
                np.bitwise_and(dist_1 > 0.0, dist_0 <= (radius + 1e-3) * tol_scale)
            )
        else:
            non_valid = np.bitwise_or(
                dist_0 <= radius + 1e-3,
                np.bitwise_and(dist_1 < 0.0, dist_0 <= (radius + 1e-3) * tol_scale)
            )
        return obs_pcd[~non_valid]

    @staticmethod
    def pcd2plane_dist(pcd: np.ndarray, center: np.ndarray, norm: np.ndarray):
        vec = pcd - center
        return np.sum(vec * norm, axis=1) / np.linalg.norm(norm, ord=2)


class StlUtils(object):
    @staticmethod
    def scale_stl(file: str, scale):
        mesh = pyvista.read_meshio(file)
        mesh.points = mesh.points * scale
        pyvista.save_meshio(file, mesh)


def main():
    # pcd = ShapeUtils.create_sphere_pcd(np.array([1., 1., 1.]), vector=np.array([1., 1., 0.]), radius=1.0, reso=0.1)
    # mesh = pyvista.PointSet(pcd)
    # mesh.plot()
    pass


if __name__ == '__main__':
    main()
