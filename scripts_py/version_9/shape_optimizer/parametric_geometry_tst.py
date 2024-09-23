import numpy as np
from typing import List, Dict
import pyvista

from scripts_py.version_9.mapf_pkg.bspline_utils import BsplineUtils


class SurfaceBspline(object):
    def __init__(self, xy_min: np.ndarray, xy_max: np.ndarray, xy_num: np.ndarray, degrees: List[int]):
        xs = np.linspace(xy_min[0], xy_max[0], num=xy_num[0])
        ys = np.linspace(xy_min[1], xy_max[1], num=xy_num[1])
        xs, ys = np.meshgrid(xs, ys)
        self.control_pcd = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1)
        self.ndims = self.control_pcd.shape
        self.bspline_cfg: Dict[int, Dict] = {
            0: {
                'degree': degrees[0],
                'knots': BsplineUtils.create_knots(self.ndims[0], degrees[0], clamp=True)
            },
            1: {
                'degree': degrees[1],
                'knots': BsplineUtils.create_knots(self.ndims[1], degrees[1], clamp=True)
            }
        }
        self.aug_cfg = {}

    def compute_mat(self, uvs: np.ndarray):
        u_cfg, v_cfg = self.bspline_cfg[0], self.bspline_cfg[1]
        # (num_u, num_u_control_points)
        u_mat = BsplineUtils.compute_mat(
            uvs[:, 0], self.ndims[0], u_cfg['degree'], u_cfg['knots'], fill_last_one=False
        )
        # (num_v, num_v_control_points)
        v_mat = BsplineUtils.compute_mat(
            uvs[:, 1], self.ndims[1], v_cfg['degree'], v_cfg['knots'], fill_last_one=False
        )
        mat = u_mat[:, :, np.newaxis] * v_mat[:, np.newaxis, :]
        self.aug_cfg['mat'] = mat

        return mat

    def evaluate_uv(self, u, v):
        u_cfg, v_cfg = self.bspline_cfg[0], self.bspline_cfg[1]
        point = np.zeros(2)
        for i in range(self.ndims[0]):
            for j in range(self.ndims[1]):
                bu = BsplineUtils.compute_Bip(u, i, u_cfg['degree'], u_cfg['knots'])
                bv = BsplineUtils.compute_Bip(v, j, v_cfg['degree'], v_cfg['knots'])
                point += bu * bv * self.control_pcd[i, j, :]
        return point

    def sample_pcd(self, uvs: np.ndarray, force_update=False):
        if force_update:
            mat = self.compute_mat(uvs)
        else:
            mat = self.aug_cfg['mat']

        res = mat[..., np.newaxis] * self.control_pcd
        return np.sum(np.sum(res, axis=1), axis=1)


class VolumetricBspline(object):
    def __init__(self, xyz_min: np.ndarray, xyz_max: np.ndarray, xyz_num: np.ndarray, degrees: List[int]):
        xs = np.linspace(xyz_min[0], xyz_max[0], num=xyz_num[0])
        ys = np.linspace(xyz_min[1], xyz_max[1], num=xyz_num[1])
        zs = np.linspace(xyz_min[2], xyz_max[2], num=xyz_num[2])
        xs, ys, zs = np.meshgrid(xs, ys, zs)
        self.control_pcd = np.concatenate([
            xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]
        ], axis=-1)
        self.ndims = self.control_pcd.shape
        self.bspline_cfg: Dict[int, Dict] = {
            0: {
                'degree': degrees[0],
                'knots': BsplineUtils.create_knots(self.ndims[0], degrees[0], clamp=True)
            },
            1: {
                'degree': degrees[1],
                'knots': BsplineUtils.create_knots(self.ndims[1], degrees[1], clamp=True)
            },
            2: {
                'degree': degrees[2],
                'knots': BsplineUtils.create_knots(self.ndims[2], degrees[2], clamp=True)
            }
        }
        self.aug_cfg = {}

    def compute_mat(self, uvws: np.ndarray):
        u_cfg, v_cfg, w_cfg = self.bspline_cfg[0], self.bspline_cfg[1], self.bspline_cfg[2]

        u_mat = BsplineUtils.compute_mat(
            uvws[:, 0], self.ndims[0], u_cfg['degree'], u_cfg['knots'], fill_last_one=False
        )
        v_mat = BsplineUtils.compute_mat(
            uvws[:, 1], self.ndims[1], v_cfg['degree'], v_cfg['knots'], fill_last_one=False
        )
        w_mat = BsplineUtils.compute_mat(
            uvws[:, 2], self.ndims[2], w_cfg['degree'], w_cfg['knots'], fill_last_one=False
        )
        mat = u_mat[:, :, np.newaxis] * v_mat[:, np.newaxis, :]
        mat = mat[:, :, :, np.newaxis] * w_mat[:, np.newaxis, np.newaxis, :]
        self.aug_cfg['mat'] = mat
        return mat

    def evaluate_uvw(self, u, v, w):
        u_cfg, v_cfg, w_cfg = self.bspline_cfg[0], self.bspline_cfg[1], self.bspline_cfg[2]
        point = np.zeros(3)
        for i in range(self.ndims[0]):
            for j in range(self.ndims[1]):
                for k in range(self.ndims[2]):
                    bu = BsplineUtils.compute_Bip(u, i, u_cfg['degree'], u_cfg['knots'])
                    bv = BsplineUtils.compute_Bip(v, j, v_cfg['degree'], v_cfg['knots'])
                    bw = BsplineUtils.compute_Bip(w, k, w_cfg['degree'], w_cfg['knots'])
                    point += bu * bv * bw * self.control_pcd[i, j, k, :]
        return point

    def sample_pcd(self, uvws: np.ndarray, force_update=False):
        if force_update:
            mat = self.compute_mat(uvws)
        else:
            mat = self.aug_cfg['mat']

        res = mat[..., np.newaxis] * self.control_pcd
        return np.sum(np.sum(np.sum(res, axis=1), axis=1), axis=1)


def debug_surface_spline():
    import matplotlib.pyplot as plt

    mesh_pcd = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [1.5, 1.0],
        [2.0, 1.0],
        [2.5, 0.5],
        [3.0, 0.0],
        [2.5, 0.0],
        [2.0, 0.0],
        [1.5, 0.0],
        [1.0, 0.0],
        [0.5, 0.0]
    ]) + 0.1

    xy_min = np.min(mesh_pcd, axis=0) * 0.95
    xy_max = np.max(mesh_pcd, axis=0) * 1.05

    model = SurfaceBspline(xy_min, xy_max, xy_num=np.array([5, 5]), degrees=[2, 2])

    us, vs = np.meshgrid(np.linspace(0.0, 1.0, 50), np.linspace(0.0, 1.0, 30))
    uvs = np.concatenate([us[..., np.newaxis], vs[..., np.newaxis]], axis=-1).reshape((-1, 2))
    pcd = model.sample_pcd(uvs, force_update=True)

    control_pcd = model.control_pcd.reshape((-1, 2))
    plt.scatter(control_pcd[:, 0], control_pcd[:, 1], c='g', s=10.0)
    plt.scatter(mesh_pcd[:, 0], mesh_pcd[:, 1], c='r', s=10.0)
    plt.scatter(pcd[:, 0], pcd[:, 1], c='b', s=2.0)
    plt.show()


def debug_volume_spline():
    model = VolumetricBspline(
        xyz_min=np.array([0., 0., 0.]),
        xyz_max=np.array([1., 1., 1.]),
        xyz_num=np.array([10, 10, 10]),
        degrees=[3, 3, 3]
    )

    us, vs, ws = np.meshgrid(np.linspace(0.0, 1.0, 30), np.linspace(0.0, 1.0, 30), np.linspace(0.0, 1.0, 30))
    uvws = np.concatenate([
        us[..., np.newaxis], vs[..., np.newaxis], ws[..., np.newaxis]
    ], axis=-1).reshape((-1, 3))
    pcd = model.sample_pcd(uvws, force_update=True)

    mesh = pyvista.PointSet(pcd)
    mesh.plot()


if __name__ == '__main__':
    # debug_surface_spline()
    debug_volume_spline()
