import numpy as np
import math
from typing import Dict, List
from scipy.optimize import minimize


class BsplineUtils(object):
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
    def compute_mat(
        ts: np.ndarray, pcd_num: int, degree: int, knot: np.ndarray, fill_last_one=True
    ):
        b_mat = np.zeros(shape=(ts.shape[0], pcd_num))
        for j, t in enumerate(ts):
            for i in range(pcd_num):
                b_mat[j, i] = BsplineUtils.compute_Bip(t, i, degree, knot)
        if fill_last_one:
            b_mat[-1, -1] = 1.0
        return b_mat

    @staticmethod
    def create_knot_cfg(pcd_num: int, degree: int, left_clamp=True, right_clamp=True):
        if left_clamp and right_clamp:
            knot = np.linspace(0.0, 1.0, pcd_num - degree + 1)
            knot = np.concatenate(
                [
                    np.full(shape=(degree,), fill_value=0.0),
                    knot,
                    np.full(shape=(degree,), fill_value=1.0),
                ]
            )
            fill_last_one = True

        elif left_clamp and (not right_clamp):
            knot = np.linspace(0.0, 1.0, pcd_num + 1)
            knot = np.concatenate([np.full(shape=(degree,), fill_value=0.0), knot])
            fill_last_one = False

        elif (not left_clamp) and right_clamp:
            knot = np.linspace(0.0, 1.0, pcd_num + 1)
            knot = np.concatenate([knot, np.full(shape=(degree,), fill_value=1.0)])
            fill_last_one = True

        else:
            knot = np.linspace(0.0, 1.0, pcd_num + degree + 1)
            fill_last_one = False

        return {"knot": knot, "fill_last_one": fill_last_one}

    @staticmethod
    def create_ts(
        sample_num: int,
        degree: int,
        knot: np.ndarray,
        left_clamp=True,
        right_clamp=True,
        left_split_num=None,
        right_split_num=None,
    ):
        if left_split_num is None:
            left_split_num = degree
        assert left_split_num >= degree

        if right_split_num is None:
            right_split_num = degree
        assert right_split_num >= degree

        if left_clamp and right_clamp:
            ts = np.linspace(0.0, 1.0, sample_num)
        elif left_clamp and (not right_clamp):
            ts = np.linspace(0.0, knot[-right_split_num - 1], sample_num)
        elif (not left_clamp) and right_clamp:
            ts = np.linspace(knot[left_split_num], 1.0, sample_num)
        else:
            # Restrict to range where the curve is defined
            ts = np.linspace(
                knot[left_split_num], knot[-right_split_num - 1], sample_num
            )
        return ts

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
    def compute_first_order_mat(
        t_list: np.ndarray, pcd_num: int, degree: int, knot: np.ndarray
    ):
        """
        Since I can't estimate the real distance of t_list, the curvature will be wrong estimate,
         value of curvature related to t_list
        """
        b_mat = np.zeros(shape=(t_list.shape[0], pcd_num))
        for j, t in enumerate(t_list):
            for i in range(pcd_num):
                b_mat[j, i] = BsplineUtils.compute_Bip_first_order(t, i, degree, knot)
        return b_mat

    @staticmethod
    def compute_second_order_mat(
        t_list: np.ndarray, pcd_num: int, degree: int, knot: np.ndarray
    ):
        """
        Since I can't estimate the real distance of t_list, the curvature will be wrong estimate,
         value of curvature related to t_list
        """
        b_mat = np.zeros(shape=(t_list.shape[0], pcd_num))
        for j, t in enumerate(t_list):
            for i in range(pcd_num):
                b_mat[j, i] = BsplineUtils.compute_Bip_second_order(t, i, degree, knot)
        return b_mat

    @staticmethod
    def compute_curvature(
        t_list: np.ndarray, pcd: np.ndarray, degree: int, knot: np.ndarray
    ):
        mat1 = BsplineUtils.compute_first_order_mat(t_list, pcd.shape[0], degree, knot)
        pcd_order1 = mat1.dot(pcd)
        mat2 = BsplineUtils.compute_second_order_mat(t_list, pcd.shape[0], degree, knot)
        pcd_order2 = mat2.dot(pcd)
        a = np.linalg.norm(
            np.cross(pcd_order1, pcd_order2, axis=1).reshape((pcd_order1.shape[0], -1)),
            axis=1,
            ord=2,
        )
        b = np.power(np.linalg.norm(pcd_order1, axis=1, ord=2), 3.0)
        return a / (b + 1e-8)

    @staticmethod
    def compute_menger_curvature(
        pcd: np.ndarray,
        start_vec: np.ndarray | None = None,
        end_vec: np.ndarray | None = None,
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
    def compute_approximate_curvature(
        pcd: np.ndarray,
        start_vec: np.ndarray | None = None,
        end_vec: np.ndarray | None = None,
    ):
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

        curvature = np.arccos(cos_theta) / np.minimum(
            length0, length1
        )  # real curvature
        # arc_theta = np.log((1.0 - cos_theta) * 11.0 + 1.0)
        # curvature = arc_theta / np.minimum(length0, length1)

        return curvature

    @staticmethod
    def compute_ts(
        control_pcd: np.ndarray,
        pcd: np.ndarray,
        degree: int,
        left_clamp: bool,
        right_clamp: bool,
    ):
        knot_cfg = BsplineUtils.create_knot_cfg(
            control_pcd.shape[0], degree, left_clamp, right_clamp
        )

        def cost_fun(ts):
            mat = BsplineUtils.compute_mat(
                ts,
                control_pcd.shape[0],
                degree,
                knot_cfg["knot"],
                knot_cfg["fill_last_one"],
            )
            pcd_tmp = mat.dot(control_pcd)
            cost = np.sum(np.power(pcd - pcd_tmp, 2.0))
            return cost

        res = minimize(cost_fun, x0=np.full(pcd.shape[0], fill_value=0.5))
        ts = res.x

        return ts

    @staticmethod
    def compute_spline_length(
        c_pcd: np.ndarray,
        degree: int,
        left_clamp: bool,
        right_clamp: bool,
        sample_num_init: int = 50,
        sample_change_ratio: float = 1.5,
        atol=5e-1,
    ):
        knot_cfg = BsplineUtils.create_knot_cfg(
            c_pcd.shape[0], degree, left_clamp, right_clamp
        )
        sample_num, last_length = sample_num_init, None
        for _ in range(20):
            ts = BsplineUtils.create_ts(
                sample_num, degree, knot_cfg["knot"], left_clamp, right_clamp
            )
            spline_mat = BsplineUtils.compute_mat(
                ts, c_pcd.shape[0], degree, knot_cfg["knot"], knot_cfg["fill_last_one"]
            )
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
