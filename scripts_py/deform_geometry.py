'''
Free-form deformation based on B-spline
Reference:
    [1] https://github.com/zhiqiangwang/pyspline
'''

import numpy as np
from typing import List, Dict, Union
import pyvista
import re
import torch

from pyspline import Curve, Surface, Volume


class BsplineBase(object):
    @staticmethod
    def compute_Bip(t, i: int, degree: int, knot: np.ndarray):
        if degree == 0:
            if knot[i] <= t < knot[i + 1]:
                return 1.0
            else:
                return 0.0

        if knot[i + degree] == knot[i]:
            c1 = 0.0
        else:
            c1 = (t - knot[i]) / (knot[i + degree] - knot[i])
            c1 = c1 * BsplineBase.compute_Bip(t, i, degree - 1, knot)

        if knot[i + degree + 1] == knot[i + 1]:
            c2 = 0.0
        else:
            c2 = (knot[i + degree + 1] - t) / (knot[i + degree + 1] - knot[i + 1])
            c2 = c2 * BsplineBase.compute_Bip(t, i + 1, degree - 1, knot)

        return c1 + c2

    @staticmethod
    def compute_bspline_mat(ts: np.ndarray, pcd_num: int, degree: int, knot: np.ndarray, fill_last_one=True):
        b_mat = np.zeros(shape=(ts.shape[0], pcd_num))
        for j, t in enumerate(ts):
            for i in range(pcd_num):
                b_mat[j, i] = BsplineBase.compute_Bip(t, i, degree, knot)
        if fill_last_one:
            b_mat[-1, -1] = 1.0
        return b_mat


class SurfaceBspline(BsplineBase):
    def __init__(
        self,
        pcd_surface: np.ndarray,
        uv_num: List[int],
        uv_degree: List[int],
        fit_nIter=100,
    ):
        '''
        Args:
            pcd_surface: [n, 3], surface points
            uv_num: [u_num, v_num], number of the control points
            uv_degree: [u_degree, v_degree], degree of the control points
            fit_nIter: int, number of the iterations
        '''
        self.uv_num = uv_num
        self.uv_degree = uv_degree
        surf = Surface(
            x=pcd_surface[:, :, 0],
            y=pcd_surface[:, :, 1],
            z=pcd_surface[:, :, 2],
            ku=uv_degree[0] + 1,
            kv=uv_degree[1] + 1,  # pyspline的配置需要加1，knot才能匹配
            nCtlu=uv_num[0],
            nCtlv=uv_num[1],
            nIter=fit_nIter,
        )
        self.coef = surf.coef  # control points
        self.knot = {"u": surf.tu, "v": surf.tv}

        pcd = pcd_surface.reshape((-1, 3))
        pcd_u, pcd_v, resid_distance = surf.projectPoint(pcd, nIter=fit_nIter)
        print(f"[Info]:Maximum Residual:{np.max(resid_distance)}")
        self.pcd_uv = np.concatenate([pcd_u.reshape((-1, 1)), pcd_v.reshape((-1, 1))], axis=1)

        # 由于Bspline的compute_Bip函数缺陷导致不能含有0.0或1.0这两个值
        self.pcd_uv = np.maximum(np.minimum(self.pcd_uv, 1 - 1e-5), 1e-5)

        # surf.getDerivative(self.pcd_uv[0, 0], self.pcd_uv[0, 1])  # 可用于计算表面曲率或切线信息
        self.mat: np.ndarray = self.compute_mat(self.pcd_uv, self.uv_num, self.uv_degree, self.knot["u"], self.knot["v"])

    def compute_mat(self, uvs: np.ndarray, uv_num, uv_degree, u_knot, v_knot):
        u_mat = self.compute_bspline_mat(uvs[:, 0], uv_num[0], uv_degree[0], u_knot, fill_last_one=False)
        v_mat = self.compute_bspline_mat(uvs[:, 1], uv_num[1], uv_degree[1], v_knot, fill_last_one=False)
        mat = u_mat[:, :, np.newaxis] * v_mat[:, np.newaxis, :]
        return mat

    def evaluate_uvw_from_raw(self, u, v):
        point = np.zeros(3)
        for i in range(self.uv_num[0]):
            for j in range(self.uv_num[1]):
                bu = self.compute_Bip(u, i, self.uv_degree[0], self.knot["u"])
                bv = self.compute_Bip(v, j, self.uv_degree[1], self.knot["v"])
                point += bu * bv * self.coef[i, j, :]
        return point

    def get_pcd(self):
        res = self.mat[..., np.newaxis] * self.coef
        return np.sum(np.sum(res, axis=1), axis=1)


class VolumeBspline(BsplineBase):
    def __init__(
        self,
        cube_range: List[List[float]],
        cube_num: List[int],
        cube_degree: List[int],
        volume_pcd: np.ndarray,
        fit_nIter=100,
    ):
        '''
        Args:
            cube_range: [[x_min, x_max], [y_min, y_max], [z_min, z_max]], range of the cube
            cube_num: [x_num, y_num, z_num]
            cube_degree: [x_degree, y_degree, z_degree]
            volume_pcd: [n, 3]
            fit_nIter: int
            with_torch: bool
        '''
        self.uvw_num = cube_num
        self.uvw_degree = cube_degree

        xs = np.linspace(cube_range[0][0], cube_range[0][1], num=cube_num[0])
        ys = np.linspace(cube_range[1][0], cube_range[1][1], num=cube_num[1])
        zs = np.linspace(cube_range[2][0], cube_range[2][1], num=cube_num[2])
        xs, ys, zs = np.meshgrid(xs, ys, zs)
        cube_pcd = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1)  # [x_num, y_num, z_num, 3]

        vol = Volume(
            X=cube_pcd,
            ku=cube_degree[0] + 1,
            kv=cube_degree[1] + 1,
            kw=cube_degree[2] + 1,
            nCtlu=cube_num[0],
            nCtlv=cube_num[1],
            nCtlw=cube_num[2],
            nIter=fit_nIter,
        )
        vol.coef = cube_pcd
        self.coef = vol.coef
        self.knot = {"u": vol.tu, "v": vol.tv, "w": vol.tw}

        pcd_u, pcd_v, pcd_w, resid_distance = vol.projectPoint(volume_pcd, nIter=fit_nIter)
        print(f"[Info]:Maximum Residual:{np.max(resid_distance)}")
        self.pcd_uvw = np.concatenate([pcd_u.reshape((-1, 1)), pcd_v.reshape((-1, 1)), pcd_w.reshape((-1, 1))], axis=1)  # [n, 3]

        self.mat: np.ndarray = self.compute_mat(
            self.pcd_uvw,
            self.uvw_num,
            self.uvw_degree,
            self.knot["u"],
            self.knot["v"],
            self.knot["w"],
        )

    def compute_mat(self, uvws: np.ndarray, uvw_num, uvw_degree, u_knot, v_knot, w_knot):
        u_mat = self.compute_bspline_mat(uvws[:, 0], uvw_num[0], uvw_degree[0], u_knot, fill_last_one=False)
        v_mat = self.compute_bspline_mat(uvws[:, 1], uvw_num[1], uvw_degree[1], v_knot, fill_last_one=False)
        w_mat = self.compute_bspline_mat(uvws[:, 2], uvw_num[2], uvw_degree[2], w_knot, fill_last_one=False)
        mat = u_mat[:, :, np.newaxis] * v_mat[:, np.newaxis, :]
        mat = mat[:, :, :, np.newaxis] * w_mat[:, np.newaxis, np.newaxis, :]
        return mat

    def evaluate_uvw_from_raw(self, u, v, w):
        point = np.zeros(3)
        for i in range(self.uvw_num[0]):
            for j in range(self.uvw_num[1]):
                for k in range(self.uvw_num[2]):
                    bu = self.compute_Bip(u, i, self.uvw_degree[0], self.knot["u"])
                    bv = self.compute_Bip(v, j, self.uvw_degree[1], self.knot["v"])
                    bw = self.compute_Bip(w, k, self.uvw_degree[2], self.knot["w"])
                    point += bu * bv * bw * self.coef[i, j, k, :]
        return point

    def get_pcd(self):
        res = self.mat[..., np.newaxis] * self.coef
        return np.sum(np.sum(np.sum(res, axis=1), axis=1), axis=1)


def debug_surface():
    x = np.linspace(-4, 4, 30)
    y = np.linspace(-4, 4, 30)
    x, y = np.meshgrid(x, y)
    z = np.cos(x) * np.sin(y)
    pcd_surface = np.concatenate(
        [
            x[:, :, np.newaxis],
            y[:, :, np.newaxis],
            z[:, :, np.newaxis],
        ],
        axis=-1,
    )

    mesh = SurfaceBspline(pcd_surface, uv_num=[15, 15], uv_degree=[3, 3])
    reproject_pcd = mesh.get_pcd()
    pcd = pcd_surface.reshape((-1, 3))

    error = np.linalg.norm(reproject_pcd - pcd, ord=2, axis=1)
    print(f"Max Error:{np.max(error)}")

    plotter = pyvista.Plotter()
    plotter.add_mesh(pyvista.PointSet(pcd), color=(1.0, 0.0, 0.0))
    plotter.add_mesh(pyvista.PointSet(reproject_pcd), color=(0.0, 1.0, 0.0))
    plotter.show()


def debug_volume():
    mesh = pyvista.read("/home/admin123456/Desktop/work/para_geo/model.vtk")
    mesh_pcd = np.asarray(mesh.points)

    mesh_min = np.min(mesh_pcd, axis=0)
    mesh_max = np.max(mesh_pcd, axis=0)
    cube_range = [
        [mesh_min[0] - 1.0, mesh_max[0] + 1.0],
        [mesh_min[1] - 1.0, mesh_max[1] + 1.0],
        [mesh_min[2] - 1.0, mesh_max[2] + 1.0],
    ]
    cube_num = [5, 5, 5]
    cube_degree = [3, 3, 3]
    mesh = VolumeBspline(cube_range, cube_num, cube_degree, mesh_pcd)
    reproject_pcd = mesh.get_pcd()

    error = np.linalg.norm(reproject_pcd - mesh_pcd, ord=2, axis=1)
    print(f"Max Error:{np.max(error)}")

    plotter = pyvista.Plotter()
    plotter.add_mesh(pyvista.PointSet(mesh_pcd), color=(1.0, 0.0, 0.0))
    plotter.add_mesh(pyvista.PointSet(reproject_pcd), color=(0.0, 1.0, 0.0))
    plotter.show()


if __name__ == "__main__":
    debug_surface()