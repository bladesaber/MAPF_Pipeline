import numpy as np
import math
from typing import Dict, List
from scipy.optimize import minimize


class BsplineUtils(object):
    @staticmethod
    def create_knots(
            pcd_num: int, degree: int, range_min=0.0, range_max=1.0, clamp=True,
            uniform_weight: float = -1.0, uniform_step: int = np.inf
    ):
        """
        pcd: control point cloud
        """
        knot = np.linspace(range_min, range_max, pcd_num - degree + 1)

        # todo wrong here
        # if uniform_weight > 0:
        #     assert 0.0 < uniform_weight < 1.0
        #
        #     num = knot.shape[0] - 2
        #     step = min(math.floor(num * 0.5), uniform_step)
        #
        #     for i in range(step):
        #         begin_loc = i + 1
        #         knot[begin_loc] += (knot[begin_loc + 1] - knot[begin_loc]) * uniform_weight
        #
        #         end_loc = num - i
        #         knot[end_loc] -= (knot[end_loc] - knot[end_loc - 1]) * uniform_weight

        if clamp:
            knot = np.concatenate([
                np.full(shape=(degree,), fill_value=range_min), knot, np.full(shape=(degree,), fill_value=range_max)
            ])
        return knot

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
            c1 = c1 * BsplineUtils.compute_Bip(t, i, degree - 1, knot)

        if knot[i + degree + 1] == knot[i + 1]:
            c2 = 0.0
        else:
            c2 = (knot[i + degree + 1] - t) / (knot[i + degree + 1] - knot[i + 1])
            c2 = c2 * BsplineUtils.compute_Bip(t, i + 1, degree - 1, knot)

        return c1 + c2

    @staticmethod
    def compute_mat(t_list: np.ndarray, pcd_num: int, degree: int, knot: np.ndarray, fill_last_one=True):
        b_mat = np.zeros(shape=(t_list.shape[0], pcd_num))
        for j, t in enumerate(t_list):
            for i in range(pcd_num):
                b_mat[j, i] = BsplineUtils.compute_Bip(t, i, degree, knot)
            if fill_last_one:
                if j == t_list.shape[0] - 1:
                    b_mat[-1, -1] = 1.0
        return b_mat

    @staticmethod
    def compute_Bip_first_order(t, i: int, degree: int, knot: np.ndarray):
        if degree == 0:
            return 0.0

        if knot[i + degree] == knot[i]:
            c1 = 0.0
        else:
            c1 = degree / (knot[i + degree] - knot[i])
            c1 = c1 * BsplineUtils.compute_Bip(t, i, degree - 1, knot)

        if knot[i + degree + 1] == knot[i + 1]:
            c2 = 0.0
        else:
            c2 = -degree / (knot[i + degree + 1] - knot[i + 1])
            c2 = c2 * BsplineUtils.compute_Bip(t, i + 1, degree - 1, knot)
        return c1 + c2

    @staticmethod
    def compute_Bip_second_order(t, i: int, degree: int, knot: np.ndarray):
        if degree <= 1:
            return 0.0

        if knot[i + degree] == knot[i]:
            c1 = 0.0
        else:
            c1 = degree / (knot[i + degree] - knot[i])
            c1 = c1 * BsplineUtils.compute_Bip_first_order(t, i, degree - 1, knot)

        if knot[i + degree + 1] == knot[i + 1]:
            c2 = 0.0
        else:
            c2 = -degree / (knot[i + degree + 1] - knot[i + 1])
            c2 = c2 * BsplineUtils.compute_Bip_first_order(t, i + 1, degree - 1, knot)
        return c1 + c2

    @staticmethod
    def compute_first_order_mat(t_list: np.ndarray, pcd: np.ndarray, degree: int, knot: np.ndarray):
        """
        Since I can't estimate the real distance of t_list, the curvature will be wrong estimate,
         value of curvature related to t_list
        """
        num = pcd.shape[0]
        b_mat = np.zeros(shape=(t_list.shape[0], num))
        for j, t in enumerate(t_list):
            for i in range(num):
                b_mat[j, i] = BsplineUtils.compute_Bip_first_order(t, i, degree, knot)
        return b_mat

    @staticmethod
    def compute_second_order_mat(t_list: np.ndarray, pcd: np.ndarray, degree: int, knot: np.ndarray):
        """
        Since I can't estimate the real distance of t_list, the curvature will be wrong estimate,
         value of curvature related to t_list
        """
        num = pcd.shape[0]
        b_mat = np.zeros(shape=(t_list.shape[0], num))
        for j, t in enumerate(t_list):
            for i in range(num):
                b_mat[j, i] = BsplineUtils.compute_Bip_second_order(t, i, degree, knot)
        return b_mat

    @staticmethod
    def compute_curvature(t_list: np.ndarray, pcd: np.ndarray, degree: int, knot: np.ndarray):
        mat1 = BsplineUtils.compute_first_order_mat(t_list, pcd, degree, knot)
        pcd_order1 = mat1.dot(pcd)
        mat2 = BsplineUtils.compute_second_order_mat(t_list, pcd, degree, knot)
        pcd_order2 = mat2.dot(pcd)
        a = np.linalg.norm(
            np.cross(pcd_order1, pcd_order2, axis=1).reshape((pcd_order1.shape[0], -1)),
            axis=1, ord=2
        )
        b = np.power(np.linalg.norm(pcd_order1, axis=1, ord=2), 3.0)
        return a / (b + 1e-8)

    @staticmethod
    def compute_menger_curvature(
            pcd: np.ndarray, start_vec: np.ndarray = None, end_vec: np.ndarray = None
    ):
        pcd_list = []
        if start_vec is not None:
            pcd_list.append((pcd[0] - start_vec).reshape((1, -1)))
        pcd_list.append(pcd)
        if end_vec is not None:
            pcd_list.append((pcd[-1] + end_vec).reshape((1, -1)))
        pcd = np.concatenate(pcd_list, axis=0)

        pcd_0 = pcd[:-2, :]  # x
        pcd_1 = pcd[1:-1, :]  # y
        pcd_2 = pcd[2:, :]  # z

        a_lengths = np.linalg.norm(pcd_1 - pcd_0, ord=2, axis=1)
        b_lengths = np.linalg.norm(pcd_2 - pcd_1, ord=2, axis=1)
        c_lengths = np.linalg.norm(pcd_0 - pcd_2, ord=2, axis=1)
        s = (a_lengths + b_lengths + c_lengths) * 0.5 + 1e-12
        A = np.sqrt(s * (s - a_lengths) * (s - b_lengths) * (s - c_lengths))
        radius = a_lengths * b_lengths * c_lengths * 0.25 / (A + 1e-16)
        return 1.0 / radius

    @staticmethod
    def compute_approximate_curvature(pcd: np.ndarray, start_vec: np.ndarray = None, end_vec: np.ndarray = None):
        pcd_list = []
        if start_vec is not None:
            pcd_list.append((pcd[0] - start_vec).reshape((1, -1)))
        pcd_list.append(pcd)
        if end_vec is not None:
            pcd_list.append((pcd[-1] + end_vec).reshape((1, -1)))
        pcd = np.concatenate(pcd_list, axis=0)

        vectors = pcd[1:, :] - pcd[:-1, :]
        lengths = np.linalg.norm(vectors, axis=1, ord=2)
        vector0, vector1 = vectors[:-1, :], vectors[1:, :]
        length0, length1 = lengths[:-1], lengths[1:]
        cos_theta = np.sum(vector0 * vector1, axis=1) / (length0 * length1)

        curvature = np.arccos(cos_theta) / np.minimum(length0, length1)  # real curvature
        # arc_theta = np.log((1.0 - cos_theta) * 11.0 + 1.0)
        # curvature = arc_theta / np.minimum(length0, length1)

        return curvature

    @staticmethod
    def compute_ts(control_pcd: np.ndarray, pcd: np.ndarray, degree):
        knots = BsplineUtils.create_knots(control_pcd.shape[0], degree)

        def cost_fun(t_list):
            mat = BsplineUtils.compute_mat(t_list, control_pcd.shape[0], degree, knots, fill_last_one=False)
            pcd_tmp = mat.dot(control_pcd)
            cost = np.sum(np.power(pcd - pcd_tmp, 2.0))
            return cost

        res = minimize(cost_fun, x0=np.full(pcd.shape[0], fill_value=0.5))
        ts = res.x

        return ts

    @staticmethod
    def compute_spline_length(
            c_pcd: np.ndarray, degree: int,
            sample_num_init: int = 50, sample_change_ratio: float = 1.5,
            uniform_weight=0.5, uniform_step=3, atol=5e-1
    ):
        knot = BsplineUtils.create_knots(c_pcd.shape[0], degree)
        sample_num, last_length = sample_num_init, None
        for _ in range(20):
            ts = np.linspace(0.0, 1.0, sample_num)
            spline_mat = BsplineUtils.compute_mat(ts, c_pcd.shape[0], degree, knot)
            r_pcd = spline_mat.dot(c_pcd)
            length = np.sum(np.linalg.norm(r_pcd[1:, :] - r_pcd[:-1, :], axis=1, ord=2))
            # print(f"[DEBUG]: Sample_num:{sample_num} length:{length}.")
            if last_length is not None:
                abs_tol = np.abs(length - last_length)
                if abs_tol < atol:
                    return length

            last_length = length
            sample_num = int(sample_num * sample_change_ratio)
        return None


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
        self.control_pcd = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1)
        self.ndims = self.control_pcd.shape
        self.bspline_cfg: Dict[int, Dict] = {
            0: {
                'degree': degrees[0], 'knots': BsplineUtils.create_knots(self.ndims[0], degrees[0], clamp=True)
            },
            1: {
                'degree': degrees[1], 'knots': BsplineUtils.create_knots(self.ndims[1], degrees[1], clamp=True)
            },
            2: {
                'degree': degrees[2], 'knots': BsplineUtils.create_knots(self.ndims[2], degrees[2], clamp=True)
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


class FreeFormVolume(object):
    # todo 基于BsplineUtils.compute_ts的不连续 Volumetric Bspline
    pass

# a = SurfaceBspline(np.array([1., 1.]), np.array([3., 3.]), np.array([2, 2]), [1, 1])
# us, vs = np.linspace(0.01, 0.99, 50), np.linspace(0.01, 0.99, 50)
# us, vs = np.meshgrid(us, vs)
# uvs = np.concatenate([us[..., np.newaxis], vs[..., np.newaxis]], axis=-1).reshape((-1, 2))
# r = a.sample_pcd(uvs, force_update=True)
#
# import matplotlib.pyplot as plt
# b = a.control_pcd.reshape((-1, 2))
# plt.scatter(b[:, 0], b[:, 1], s=20.0, c='r')
# plt.scatter(r[:, 0], r[:, 1], s=10.0, c='b')
# plt.show()
