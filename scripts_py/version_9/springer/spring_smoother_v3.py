import math
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from copy import deepcopy

from scipy import optimize
from scipy.optimize import Bounds
from sklearn.neighbors import KDTree

from scripts_py.version_9.app.env_utils import Shape_Utils
from scripts_py.version_9.springer.smoother_utils import ConnectVisulizer, RecordAnysisVisulizer

import warnings
# from pandas.core.common import SettingWithCopyWarning
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class BSpline_utils(object):
    @staticmethod
    def compute_B_ik_x(x: float, k: int, i: int, t: np.array):
        """
        x: value for [0, 1]
        k: degree
        i: the idx of control point
        t: k-nots
        Evaluates B_{i,k}(x), the influence of the i control points to x
        """
        if k == 0:
            return 1.0 if t[i] <= x < t[i + 1] else 0.0
        if t[i + k] == t[i]:
            c1 = 0.0
        else:
            c1 = (x - t[i]) / (t[i + k] - t[i]) * BSpline_utils.compute_B_ik_x(x, k - 1, i, t)
        if t[i + k + 1] == t[i + 1]:
            c2 = 0.0
        else:
            c2 = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * BSpline_utils.compute_B_ik_x(x, k - 1, i + 1, t)
        return c1 + c2

    @staticmethod
    def compute_Bmat(xs: np.array, t: np.array, k: int, c_mat: np.array):
        """
        xs: value for [0, 1], the percentage segments of spline
        t: k-nots
        k: degree
        assume the length of t(k-nots) is equal to the length of control points
        """
        n_size = c_mat.shape[0]
        b_mat = np.zeros(shape=(xs.shape[0], n_size))
        for j, x in enumerate(xs):
            for i in range(n_size):
                b_mat[j, i] = BSpline_utils.compute_B_ik_x(x, k, i, t)
        b_mat[-1, -1] = 1.0
        return b_mat

    @staticmethod
    def compute_Bmat_uniform(c_mat: np.array, k: int, sample_num):
        """
        :param c_points: control points
        :param k: degree
        :param sample_num: num of sample points
        """
        assert c_mat.shape[0] > k
        ts = np.linspace(0.0, 1.0, num=c_mat.shape[0] - k + 1)
        ts = np.concatenate([ts[0] * np.ones(shape=(k,)), ts, ts[-1] * np.ones(shape=(k,))])
        xs = np.linspace(0.0, 1.0, sample_num)
        b_mat = BSpline_utils.compute_Bmat(xs, ts, k, c_mat)
        return b_mat

    @staticmethod
    def compute_spline_length(c_mat: np.array, k: int, tol=0.1, init_sample_num=50):
        sample_num = init_sample_num
        b_mat = BSpline_utils.compute_Bmat_uniform(c_mat, k, sample_num)
        path_xyzs = b_mat.dot(c_mat)
        last_path_length = np.sum(np.linalg.norm(path_xyzs[1:, :] - path_xyzs[:-1, :], ord=2, axis=1))

        run_times = 0
        while True:
            sample_num = math.ceil(sample_num * 1.2)
            b_mat = BSpline_utils.compute_Bmat_uniform(c_mat, k, sample_num)
            path_xyzs = b_mat.dot(c_mat)
            cur_path_length = np.sum(np.linalg.norm(path_xyzs[1:, :] - path_xyzs[:-1, :], ord=2, axis=1))
            if np.abs(cur_path_length - last_path_length) > tol:
                last_path_length = cur_path_length
            else:
                return cur_path_length

            run_times += 1
            if run_times > 50:
                return -1

    @staticmethod
    def compare_curvature(path_xyz, start_vec, end_vec):
        vecs = path_xyz[1:, :] - path_xyz[:-1, :]
        vec01 = np.concatenate([start_vec, vecs], axis=0)
        vec12 = np.concatenate([vecs, end_vec], axis=0)
        length0 = np.sqrt(np.sum(np.power(vec01, 2.0), axis=1))
        length1 = np.sqrt(np.sum(np.power(vec12, 2.0), axis=1))
        cos_theta = np.sum(vec01 * vec12, axis=1) / (length0 * length1)

        # ------ 由于 arccos 的不稳定，采用近似函数
        arc_theta = np.arccos(cos_theta)
        real_curvature = np.arccos(cos_theta) / np.minimum(length0, length1)

        # arc_theta = (1.0 - cos_theta) * np.pi / 2.0
        arc_theta = np.log((1.0 - cos_theta) * 11.0 + 1.0)
        fake_curvature = arc_theta / np.minimum(length0, length1)

        return real_curvature, fake_curvature

    @staticmethod
    def sample_uniform(c_mat: np.array, k: int, sample_num, point_num=100):
        # todo wrong function
        b_mat = BSpline_utils.compute_Bmat_uniform(c_mat, k, point_num)
        path_xyzs = b_mat.dot(c_mat)
        path_lengths = np.linalg.norm(path_xyzs[1:, :] - path_xyzs[:-1, :], ord=2, axis=1)

        sample_pcd = [path_xyzs[0, :]]
        step_length = np.sum(path_lengths) / sample_num
        last_length, cur_length = 0.0, 0.0
        for i, sub_length in enumerate(path_lengths[:-1]):
            if cur_length + sub_length > last_length + step_length:
                sample_pcd.append(path_xyzs[i + 1])
                last_length = cur_length
            cur_length += sub_length
        sample_pcd.append(path_xyzs[-1, :])
        return np.array(sample_pcd)

    @staticmethod
    def get_mini_sample_num(c_mat: np.array, k: int, radius, length_tol=0.05, sample_tol=0.05):
        real_length = BSpline_utils.compute_spline_length(c_mat, k, length_tol)
        sample_num = max(math.ceil(real_length / radius), k + 1)

        run_times = 0
        while True:
            b_mat = BSpline_utils.compute_Bmat_uniform(c_mat, k, sample_num)
            path_xyzs = b_mat.dot(c_mat)
            sample_length = np.sum(np.linalg.norm(path_xyzs[1:, :] - path_xyzs[:-1, :], ord=2, axis=1))
            if np.abs(sample_length - real_length) < sample_tol:
                return sample_num

            sample_num += 3
            run_times += 1

            if run_times > 50:
                return -1


class VarTree(object):

    def __init__(self):
        self.xs_init = []
        self.nodes_info = {}
        self.name_to_nodeIdx = {}

    def add_node(self, node_info: dict):
        """
        :param node_info: { name: str position: xyz, pose_edges: list[dict] }
        """

        node_idx = len(self.nodes_info)
        var_info = {'idx': node_idx, 'node_type': node_info['node_type'], 'name': node_info['name']}

        radian_info = None
        for init_value, xyz_tag, edge_tag in zip(
                node_info['position'], ['x', 'y', 'z'], ['pose_edge_x', 'pose_edge_y', 'pose_edge_z']
        ):
            edge = node_info[edge_tag]
            if edge['type'] == 'fix_value':
                var_info[xyz_tag] = {'type': 'fix_value', 'param': {'value': init_value}}

            elif edge['type'] == 'var_value':
                self.xs_init.append(init_value)
                var_info[xyz_tag] = {'type': 'var_value', 'param': {'col': len(self.xs_init) - 1}}

            elif edge['type'] == 'value_shift':
                var_info[xyz_tag] = {
                    'type': 'value_shift', 'param': {'ref_obj': edge['ref_obj'], 'value': edge['value']}
                }

            elif edge['type'] == 'radius_fix':
                if radian_info is None:
                    self.xs_init.append(0.0)
                    var_info['radian'] = {'type': 'var_value', 'param': {'col': len(self.xs_init) - 1}}

                var_info[xyz_tag] = {
                    'type': 'radius_fix',
                    'param': {'ref_name': edge['ref_obj'], 'is_cos': edge['is_cos'], 'radius': edge['radius']}
                }

            else:
                raise NotImplementedError

        if node_info['node_type'] == 'structor':
            var_info.update({
                'shape_pcd': node_info['shape_pcd'],
                'exclude_edges': node_info['exclude_edges'],
                'reso': node_info['shape_reso']
            })

        self.nodes_info[node_idx] = var_info
        self.name_to_nodeIdx[node_info['name']] = node_idx

        return self.nodes_info[node_idx]

    @staticmethod
    def get_cell(node_info, tag, xs, name_to_nodeIdx, nodes_info, return_method):
        """
        :param return_method: log | value | col
        """
        tag_cfg = node_info[tag]

        if tag_cfg['type'] == 'fix_value':
            if return_method == 0:
                return f"{tag_cfg['param']['value']}"
            elif return_method == 1:
                return tag_cfg['param']['value']

        elif tag_cfg['type'] == 'var_value':
            if return_method == 0:
                return f"{tag_cfg['name']}.{tag_cfg['xyz_tag']}"
            elif return_method == 1:
                return xs[tag_cfg['param']['col']]
            elif return_method == 2:
                return tag_cfg['param']['col']

        elif tag_cfg['type'] == 'value_shift':
            params = tag_cfg['param']
            ref_node = nodes_info[name_to_nodeIdx[params['ref_obj']]]
            ref_res = VarTree.get_cell(ref_node, tag, xs, name_to_nodeIdx, nodes_info, return_method)
            if return_method == 0:
                return f"{ref_res} + {params['value']}"
            elif return_method == 1:
                return ref_res + params['value']

        elif tag_cfg['type'] == 'radius_fix':
            params = tag_cfg['param']
            radian_res = VarTree.get_cell(node_info, 'radian', xs, name_to_nodeIdx, nodes_info, return_method)

            ref_node = nodes_info[name_to_nodeIdx[params['ref_obj']]]
            ref_res = VarTree.get_cell(ref_node, tag, xs, name_to_nodeIdx, nodes_info, return_method)
            radius = params['radius']

            if params['is_cos']:
                if return_method == 0:
                    return f"{ref_res} + {radius} * cos({radian_res})"
                elif return_method == 1:
                    return ref_res + radius * np.cos(radian_res)

            else:
                if return_method == 0:
                    return f"{ref_res} + {radius} * sin({radian_res})"
                elif return_method == 1:
                    return ref_res + radius * np.sin(radian_res)

        return -999

    @staticmethod
    def get_node_from_name(name, name_to_nodeIdx, nodes_info):
        return nodes_info[name_to_nodeIdx[name]]

    @staticmethod
    def get_xyz_from_node(node, xs, name_to_nodeIdx, nodes_info, return_method):
        res_x = VarTree.get_cell(node, 'x', xs, name_to_nodeIdx, nodes_info, return_method=return_method)
        res_y = VarTree.get_cell(node, 'y', xs, name_to_nodeIdx, nodes_info, return_method=return_method)
        res_z = VarTree.get_cell(node, 'z', xs, name_to_nodeIdx, nodes_info, return_method=return_method)
        return [res_x, res_y, res_z]

    @staticmethod
    def get_radian_from_node(node, xs, name_to_nodeIdx, nodes_info, return_method):
        res_radian = VarTree.get_cell(node, 'radian', xs, name_to_nodeIdx, nodes_info, return_method=return_method)
        return res_radian

    def get_xs_init(self):
        return np.array(self.xs_init).copy()


class OptimizerScipy(object):
    @staticmethod
    def penalty_bound_func(x, bound, weight, method):
        """
        :param method: larger | less
        """
        if method == 'larger':
            return 1.0 / (1.0 + np.exp((x - bound) * weight))
        else:
            return 1.0 / (1.0 + np.exp(-1 * (x - bound) * weight))

    @staticmethod
    def pentalty_relu_func(x, bound, weight, method):
        """
        :param method: larger | less
        """
        if method == 'larger':
            if x >= bound:
                return 0.0
            else:
                return (bound - x) * weight
        else:
            if x <= bound:
                return 0.0
            else:
                return (x - bound) * weight

    @staticmethod
    def cost_path_length(path_xyz):
        cost = np.sum(np.power(path_xyz[1:, :] - path_xyz[:-1, :], 2))
        return cost

    @staticmethod
    def cost_path_curvature(path_xyz, start_vec, end_vec, k=1.0 / 3.0):
        vecs = path_xyz[1:, :] - path_xyz[:-1, :]
        vec01 = np.concatenate([start_vec, vecs], axis=0)
        vec12 = np.concatenate([vecs, end_vec], axis=0)
        length0 = np.sqrt(np.sum(np.power(vec01, 2.0), axis=1))
        length1 = np.sqrt(np.sum(np.power(vec12, 2.0), axis=1))
        cos_theta = np.sum(vec01 * vec12, axis=1) / (length0 * length1)

        # ------ 由于 arccos 的不稳定，采用近似函数
        # arc_theta = (1.0 - cos_theta) * np.pi / 2.0
        arc_theta = np.log((1.0 - cos_theta) * 11.0 + 1.0)
        curvature = arc_theta / np.minimum(length0, length1)

        return np.sum(np.maximum(curvature - k, 0.0))

    @staticmethod
    def penalty_shape_conflict(xyz0, xyz1, threshold, method, weight=1.0):
        dist = np.sqrt(np.sum(np.power(xyz0 - xyz1, 2.0)))
        penalty = OptimizerScipy.pentalty_relu_func(dist, threshold, weight=weight, method=method)
        return penalty

    @staticmethod
    def penalty_plane_conflict(v0, v1, threshold, method, weight=1.0):
        penalty = OptimizerScipy.pentalty_relu_func(v0 - v1, threshold, weight=weight, method=method)
        return penalty

    def target_func(
            self, xs, xs_init,
            paths_cfg, constraints, structor_nodes,
            name_to_nodeIdx, nodes_info, opt_step,
            curvature_weight, penalty_weight, volume_cost_weight,
            debug_vis=False
    ):
        xs_new = xs + xs_init

        # ------ step 1 reconstruct env
        # ------ step 1.1 reconstruct path
        obj_dict, path_names, path_idx_2_name = {}, [], {}
        for path_idx in paths_cfg.keys():
            path_cfg = paths_cfg[path_idx]

            src_node = nodes_info[path_cfg['src_node_idx']]
            src_x, src_y, src_z = VarTree.get_xyz_from_node(src_node, xs_new, name_to_nodeIdx, nodes_info, 1)
            end_node = nodes_info[path_cfg['end_node_idx']]
            end_x, end_y, end_z = VarTree.get_xyz_from_node(end_node, xs_new, name_to_nodeIdx, nodes_info, 1)
            cell_xyzs = xs_new[path_cfg['path_cols']].reshape((-1, 3))
            c_mat = np.concatenate((
                [[src_x, src_y, src_z]],
                cell_xyzs,
                [[end_x, end_y, end_z]]
            ), axis=0)

            obj_dict[path_cfg['name']] = path_cfg['b_mat'].dot(c_mat)
            path_names.append(path_cfg['name'])
            path_idx_2_name[path_idx] = path_cfg['name']

        # ------ step 1.2 reconstruct structor
        for node in structor_nodes:
            pose_xyz = VarTree.get_xyz_from_node(node, xs_new, name_to_nodeIdx, nodes_info, 1)
            pose_xyz = np.array(pose_xyz)
            obj_dict[node['name']] = pose_xyz + node['shape_pcd']

        # self.plot_env(path_dict, structor_dict)

        # ------ step 2 compute cost
        # ------ step 2.1 compute path length cost
        length_cost = 0.0
        for path_name in path_names:
            path_xyzs = obj_dict[path_name]
            length_cost += self.cost_path_length(path_xyzs)

        """
        curvature_cost = 0.0
        for path_idx in paths_cfg.keys():
            path_cfg = paths_cfg[path_idx]
            path_xyzs = obj_dict[path_idx_2_name[path_idx]]
            curvature_cost += self.cost_path_curvature(
                path_xyzs, path_cfg['start_vec'], path_cfg['end_vec'], k=0.5
            )
        """

        # ------ step 2.2 compute shape conflict penalty
        planeMax_node = VarTree.get_node_from_name('planeMax', name_to_nodeIdx, nodes_info)
        planeMax_xyz = VarTree.get_xyz_from_node(planeMax_node, xs_new, name_to_nodeIdx, nodes_info, 1)
        planeMin_node = VarTree.get_node_from_name('planeMin', name_to_nodeIdx, nodes_info)
        planeMin_xyz = VarTree.get_xyz_from_node(planeMin_node, xs_new, name_to_nodeIdx, nodes_info, 1)

        penalty_cost = 0.0
        for cfg in constraints:
            if cfg['type'] == 'shape_conflict':
                info_0, info_1, threshold = cfg['info_0'], cfg['info_1'], cfg['threshold']
                xyz0 = obj_dict[info_0['name']][info_0['group_t_idx'], :]
                xyz1 = obj_dict[info_1['name']][info_1['group_t_idx'], :]
                penalty_cost += self.penalty_shape_conflict(xyz0, xyz1, threshold, method='larger')

            elif cfg['type'] == 'plane_max_conflict':
                info, xyz_tag, threshold = cfg['info'], cfg['xyz_tag'], cfg['threshold']
                xyz = obj_dict[info['name']][info['group_t_idx'], :]

                if xyz_tag == 'x':
                    penalty_cost += self.penalty_plane_conflict(planeMax_xyz[0], xyz[0], threshold, 'larger')
                elif xyz_tag == 'y':
                    penalty_cost += self.penalty_plane_conflict(planeMax_xyz[1], xyz[1], threshold, 'larger')
                else:
                    penalty_cost += self.penalty_plane_conflict(planeMax_xyz[2], xyz[2], threshold, 'larger')

            elif cfg['type'] == 'plane_min_conflict':
                info, xyz_tag, threshold = cfg['info'], cfg['xyz_tag'], cfg['threshold']
                xyz = obj_dict[info['name']][info['group_t_idx'], :]

                if xyz_tag == 'x':
                    penalty_cost += self.penalty_plane_conflict(xyz[0], planeMin_xyz[0], threshold, 'larger')
                elif xyz_tag == 'y':
                    penalty_cost += self.penalty_plane_conflict(xyz[1], planeMin_xyz[1], threshold, 'larger')
                else:
                    penalty_cost += self.penalty_plane_conflict(xyz[2], planeMin_xyz[2], threshold, 'larger')

            else:
                raise NotImplementedError

        # ------ step 2.3 compute volume cost
        x_dif, y_dif, z_dif = VarTree.get_xyz_from_node(planeMax_node, xs, name_to_nodeIdx, nodes_info, 1)
        volume_cost = x_dif / opt_step + y_dif / opt_step + z_dif / opt_step

        cost = length_cost + penalty_cost * penalty_weight + \
               volume_cost * volume_cost_weight  # + curvature_cost * curvature_weight

        if debug_vis:
            print(
                f"[Info] length_cost:{length_cost:.4f} "
                # f"curvature_cost:{curvature_cost:.4f} "
                f"penalty_cost:{penalty_cost * penalty_weight:.4f} "
                f"volume_cost:{volume_cost * volume_cost_weight:.4f}"
                )
            return length_cost, penalty_cost * penalty_weight, volume_cost * volume_cost_weight

        return cost

    @staticmethod
    def log_info(xs, info, cost_func):
        print(f"Iter:{info['iter']} Cost:{cost_func(xs)}")
        info['iter'] += 1

    def solve_problem(
            self, xs_init, var_tree: VarTree, paths_cfg, constraints, opt_step,
            penalty_weight=1000.0, volume_cost_weight=0.1
    ):
        assert opt_step > 0.0

        structor_nodes = []
        for node_idx in var_tree.nodes_info.keys():
            node = var_tree.nodes_info[node_idx]
            if node['node_type'] == 'structor':
                structor_nodes.append(node)

        pack_dict = {
            'xs_init': xs_init,
            'paths_cfg': paths_cfg,
            'constraints': constraints,
            'structor_nodes': structor_nodes,
            'opt_step': opt_step,
            'curvature_weight': 1.0,
            'volume_cost_weight': volume_cost_weight,
            'penalty_weight': penalty_weight,
            'name_to_nodeIdx': var_tree.name_to_nodeIdx,
            'nodes_info': var_tree.nodes_info
        }
        problem = partial(self.target_func, **pack_dict)
        log_problem = partial(problem, debug_vis=True)

        xs = np.zeros(xs_init.shape)
        lbs = np.ones(xs.shape) * -opt_step
        ubs = np.ones(xs.shape) * opt_step
        res = optimize.minimize(
            problem, xs,
            bounds=Bounds(lbs, ubs), tol=0.01,
            # options={'maxiter': 3, 'disp': True},
            # callback=partial(self.log_info, info={'iter': 0}, cost_func=problem)
        )

        # log_problem(np.zeros(xs_init.shape))
        length_cost, penalty_cost, volume_cost = log_problem(res.x)

        opt_xs = res.x
        if not res.success:
            print(f"[Debug]: State:{res.success} StateCode:{res.status} OptCost:{res.fun} "
                  f"optXs:{opt_xs.min()}->{opt_xs.max()} msg:{res.message}")
        # else:
        #     print(f"[Debug]: OptCost:{res.fun} optXs:{opt_xs.min()}->{opt_xs.max()}")

        return opt_xs, (length_cost, penalty_cost, volume_cost)


class EnvParser(object):

    def create_graph(self, pipes_cfg, structos_cfg, bgs_cfg, paths_cfg, k, length_tol, sample_tol, with_check=True):
        var_tree = VarTree()

        for cfg in pipes_cfg:
            var_tree.add_node(cfg)

        for cfg in structos_cfg:
            var_tree.add_node(cfg)

        for cfg in bgs_cfg:
            var_tree.add_node(cfg)

        name_to_nodeIdx = var_tree.name_to_nodeIdx
        nodes_info = var_tree.nodes_info
        xs = var_tree.get_xs_init()

        for path_idx in paths_cfg.keys():
            path_cfg = paths_cfg[path_idx]
            xyzs = path_cfg['xyzs']
            src_name, end_name = path_cfg['src_name'], path_cfg['end_name']
            src_node = VarTree.get_node_from_name(src_name, name_to_nodeIdx, nodes_info)
            end_node = VarTree.get_node_from_name(end_name, name_to_nodeIdx, nodes_info)

            # ------ check path
            if with_check:
                src_xyz = VarTree.get_xyz_from_node(src_node, xs, name_to_nodeIdx, nodes_info, 1)
                end_xyz = VarTree.get_xyz_from_node(end_node, xs, name_to_nodeIdx, nodes_info, 1)
                # print(path_cfg['name'], src_xyz, xyzs[0], end_xyz, xyzs[-1])
                assert np.all(np.isclose(src_xyz, xyzs[0])) and np.all(np.isclose(end_xyz, xyzs[-1]))
            # ------

            path_cols = []
            for i, cell_xyz in enumerate(xyzs[1:-1]):
                cell_node = var_tree.add_node({
                    'node_type': 'cell', 'position': cell_xyz, 'name': f"{path_cfg['name']}_cell{i + 1}",
                    'pose_edge_x': {'type': 'var_value'},
                    'pose_edge_y': {'type': 'var_value'},
                    'pose_edge_z': {'type': 'var_value'}
                })
                xyz_col = VarTree.get_xyz_from_node(cell_node, xs, name_to_nodeIdx, nodes_info, return_method=2)
                path_cols.append(xyz_col)
            path_cols = np.array(path_cols).reshape(-1)

            assert k < xyzs.shape[0]
            sample_num = BSpline_utils.get_mini_sample_num(
                xyzs, k, path_cfg['radius'], length_tol=length_tol, sample_tol=sample_tol
            )
            assert sample_num > 0

            b_mat = BSpline_utils.compute_Bmat_uniform(c_mat=xyzs, k=k, sample_num=sample_num)
            path_cfg.update({
                'src_node_idx': src_node['idx'], 'end_node_idx': end_node['idx'],
                'path_cols': path_cols, 'b_mat': b_mat,
                'start_vec': path_cfg['start_vec'].reshape((1, -1)),
                'end_vec': path_cfg['end_vec'].reshape((1, -1)),
            })

        return var_tree, paths_cfg

    def get_structor_conflict_plane(self, node, node_df: pd.DataFrame, planeMax_xyz, planeMin_xyz, opt_step):
        xyzs, constraints = node_df[['x', 'y', 'z']].values, []

        exclude_tags = []
        if 'plane_max_conflict' in node['exclude_edges'].keys():
            exclude_tags = node['exclude_edges']['plane_max_conflict']
        for xyz_tag, xyz_col in zip(['x', 'y', 'z'], [0, 1, 2]):
            if xyz_tag in exclude_tags:
                continue

            conflict_difs = xyzs[:, xyz_col] + node_df['radius'].values + opt_step - planeMax_xyz[xyz_col]
            max_dif = np.max(conflict_difs)
            idxs = np.nonzero(conflict_difs == max_dif)[0]
            if idxs.shape[0] > 15:
                idxs = np.random.choice(idxs, size=math.ceil(idxs.shape[0] * 0.35), replace=False)
            for idx in idxs:
                if conflict_difs[idx] >= 0.0:
                    cur_series = node_df.iloc[idx]
                    constraints.append({
                        'type': 'plane_max_conflict', 'xyz_tag': xyz_tag, 'threshold': cur_series['radius'],
                        'info': {
                            'group_t_idx': cur_series['group_t_idx'], 'name': cur_series['name'],
                            'debug_xyz': cur_series[['x', 'y', 'z']].values
                        }
                    })

        exclude_tags = []
        if 'plane_min_conflict' in node['exclude_edges'].keys():
            exclude_tags = node['exclude_edges']['plane_min_conflict']
        for xyz_tag, xyz_col in zip(['x', 'y', 'z'], [0, 1, 2]):
            if xyz_tag in exclude_tags:
                continue

            conflict_difs = xyzs[:, xyz_col] - node_df['radius'].values - opt_step - planeMin_xyz[xyz_col]
            min_dif = np.min(conflict_difs)
            idxs = np.nonzero(conflict_difs == min_dif)[0]
            if idxs.shape[0] > 15:
                idxs = np.random.choice(idxs, size=math.ceil(idxs.shape[0] * 0.35), replace=False)
            for idx in idxs:
                if conflict_difs[idx] <= 0.0:
                    cur_series = node_df.iloc[idx]
                    constraints.append({
                        'type': 'plane_min_conflict', 'xyz_tag': xyz_tag, 'threshold': cur_series['radius'],
                        'info': {
                            'group_t_idx': cur_series['group_t_idx'], 'name': cur_series['name'],
                            'debug_xyz': cur_series[['x', 'y', 'z']].values
                        }
                    })

        return constraints

    def get_path_conflict_plane(self, node_df: pd.DataFrame, planeMax_xyz, planeMin_xyz, opt_step):
        xyzs, constraints = node_df[['x', 'y', 'z']].values, []
        for xyz_tag, xyz_col in zip(['x', 'y', 'z'], [0, 1, 2]):
            conflicts = xyzs[:, xyz_col] + node_df['radius'].values + opt_step >= planeMax_xyz[xyz_col]
            for i in np.nonzero(conflicts)[0]:
                cur_series = node_df.iloc[i]
                constraints.append({
                    'type': 'plane_max_conflict', 'xyz_tag': xyz_tag, 'threshold': cur_series['radius'],
                    'info': {
                        'group_t_idx': cur_series['group_t_idx'], 'name': cur_series['name'],
                        'debug_xyz': cur_series[['x', 'y', 'z']].values
                    }
                })

            conflicts = xyzs[:, xyz_col] - node_df['radius'].values - opt_step <= planeMin_xyz[xyz_col]
            for i in np.nonzero(conflicts)[0]:
                cur_series = node_df.iloc[i]
                constraints.append({
                    'type': 'plane_min_conflict', 'xyz_tag': xyz_tag, 'threshold': cur_series['radius'],
                    'info': {
                        'group_t_idx': cur_series['group_t_idx'], 'name': cur_series['name'],
                        'debug_xyz': cur_series[['x', 'y', 'z']].values
                    }
                })
        return constraints

    def get_shape_conflict(self, nodes_df: pd.DataFrame, opt_step, path_names: list, structor_names: list):
        constraints = []
        max_step_length = np.linalg.norm(np.array([opt_step, opt_step, opt_step]), ord=2) * 1.05

        # --- 由于path的约束比structor更紧致，因此必须先由path开始
        sort_names = path_names + structor_names

        for name in sort_names[:-1]:
            cur_df: pd.DataFrame = nodes_df[nodes_df['name'] == name]
            other_df: pd.DataFrame = nodes_df[nodes_df['name'] != name]

            cur_type = cur_df.iloc[0]['df_type']
            max_search_radius = cur_df['radius'].max() + other_df['radius'].max() + max_step_length * 2.0

            cur_xyzs = cur_df[['x', 'y', 'z']].values
            other_xyzs = other_df[['x', 'y', 'z']].values
            other_tree = KDTree(other_xyzs)
            idxs_list, dists_list = other_tree.query_radius(cur_xyzs, max_search_radius, return_distance=True)

            if cur_type == 'path':
                for i, (idxs, dists) in enumerate(zip(idxs_list, dists_list)):
                    if idxs.shape[0] == 0:
                        continue

                    cur_radius = cur_df.iloc[i]['radius']
                    other_radius_list = other_df.iloc[idxs]['radius'].values
                    real_thresholds = cur_radius + other_radius_list

                    inside_bool = dists <= real_thresholds + max_step_length * 2.0
                    idxs = idxs[inside_bool]
                    if idxs.shape[0] == 0:
                        continue
                    real_thresholds = real_thresholds[inside_bool]
                    dists = dists[inside_bool]

                    cur_series = cur_df.iloc[i]
                    conflicts_df: pd.DataFrame = other_df.iloc[idxs]
                    conflicts_df['dist'] = dists
                    conflicts_df['real_thresholds'] = real_thresholds

                    for group_idx_tuple, group_data in conflicts_df.groupby(by=['name']):
                        group_dists = group_data['dist'].values
                        select_group_idxs = np.nonzero(np.isclose(
                            group_dists, np.min(group_dists), atol=cur_series['radius'] * 0.1
                        ))[0]
                        # select_group_idxs = list(range(group_data.shape[0]))

                        for select_group_idx in select_group_idxs:
                            other_series = group_data.iloc[select_group_idx]
                            constraints.append({
                                'type': 'shape_conflict',
                                'info_0': {
                                    'group_t_idx': cur_series['group_t_idx'], 'name': cur_series['name'],
                                    'debug_xyz': cur_series[['x', 'y', 'z']].values
                                },
                                'info_1': {
                                    'group_t_idx': other_series['group_t_idx'], 'name': other_series['name'],
                                    'debug_xyz': other_series[['x', 'y', 'z']].values
                                },
                                'threshold': other_series['real_thresholds']
                            })

            else:
                record = {}
                for i, (idxs, dists) in enumerate(zip(idxs_list, dists_list)):
                    if idxs.shape[0] == 0:
                        continue

                    cur_radius = cur_df.iloc[i]['radius']
                    other_radius_list = other_df.iloc[idxs]['radius'].values
                    real_thresholds = cur_radius + other_radius_list
                    inside_bool = dists <= real_thresholds + max_step_length * 2.0

                    idxs = idxs[inside_bool]
                    if idxs.shape[0] == 0:
                        continue
                    real_thresholds = real_thresholds[inside_bool]
                    dists = dists[inside_bool]

                    cur_series = cur_df.iloc[i]
                    conflicts_df: pd.DataFrame = other_df.iloc[idxs]
                    conflicts_df['dist'] = dists
                    conflicts_df['real_thresholds'] = real_thresholds

                    for group_idx_tuple, group_data in conflicts_df.groupby(by=['name']):
                        select_group_idx = np.argmin(group_data['dist'])
                        other_series = group_data.iloc[select_group_idx]
                        other_name, other_group_t_idx = other_series['name'], other_series['group_t_idx']

                        if other_name not in record.keys():
                            record[other_name] = {}

                        if other_group_t_idx not in record[other_name].keys():
                            record[other_name][other_group_t_idx] = {
                                'series0': cur_series, 'series1': other_series, 'score': other_series['dist']
                            }
                        else:
                            if other_series['dist'] < record[other_name][other_group_t_idx]['score']:
                                record[other_name][other_group_t_idx] = {
                                    'series0': cur_series, 'series1': other_series, 'score': other_series['dist']
                                }

                for name_key in record.keys():
                    for t_idx_key in record[name_key].keys():
                        cur_series = record[name_key][t_idx_key]['series0']
                        other_series = record[name_key][t_idx_key]['series1']
                        constraints.append({
                            'type': 'shape_conflict',
                            'info_0': {
                                'group_t_idx': cur_series['group_t_idx'], 'name': cur_series['name'],
                                'debug_xyz': cur_series[['x', 'y', 'z']].values
                            },
                            'info_1': {
                                'group_t_idx': other_series['group_t_idx'], 'name': other_series['name'],
                                'debug_xyz': other_series[['x', 'y', 'z']].values
                            },
                            'threshold': other_series['real_thresholds']
                        })

            # ------ 为防止重复生成，这是必须的
            nodes_df = other_df

        return constraints

    def get_constraints(self, xs, var_tree, paths_cfg, opt_step=0.1):
        name_to_nodeIdx = var_tree.name_to_nodeIdx
        nodes_info = var_tree.nodes_info

        # ------ step 1 create data point dataframe
        dfs = []

        # ------ step 1.1 record structor data point
        structor_names = []
        for node_idx in nodes_info.keys():
            node = nodes_info[node_idx]
            if node['node_type'] in ['cell', 'connector']:
                continue

            x, y, z = VarTree.get_xyz_from_node(node, xs, name_to_nodeIdx, nodes_info, 1)
            pcd_world = np.array([x, y, z]) + node['shape_pcd']
            sub_df = pd.DataFrame(pcd_world, columns=['x', 'y', 'z'])
            sub_df[['radius', 'name', 'node_idx', 'df_type']] = node['reso'] * 0.5, node['name'], node_idx, 'structor'
            sub_df['group_t_idx'] = np.arange(0, pcd_world.shape[0], 1)
            dfs.append(sub_df)
            structor_names.append(node['name'])

        # ------ step 1.2 record path data point
        path_names = []
        for path_idx in paths_cfg.keys():
            path_cfg = paths_cfg[path_idx]

            src_node = nodes_info[path_cfg['src_node_idx']]
            src_x, src_y, src_z = VarTree.get_xyz_from_node(src_node, xs, name_to_nodeIdx, nodes_info, 1)
            end_node = nodes_info[path_cfg['end_node_idx']]
            end_x, end_y, end_z = VarTree.get_xyz_from_node(end_node, xs, name_to_nodeIdx, nodes_info, 1)
            cell_xyzs = xs[path_cfg['path_cols']].reshape((-1, 3))
            c_mat = np.concatenate((
                [[src_x, src_y, src_z]],
                cell_xyzs,
                [[end_x, end_y, end_z]]
            ), axis=0)
            path_xyzs = path_cfg['b_mat'].dot(c_mat)

            sub_df = pd.DataFrame(path_xyzs, columns=['x', 'y', 'z'])
            sub_df[['radius', 'name', 'node_idx', 'df_type']] = path_cfg['radius'], path_cfg['name'], -1, 'path'
            sub_df['group_t_idx'] = np.arange(0, path_xyzs.shape[0], 1)
            dfs.append(sub_df)
            path_names.append(path_cfg['name'])

        dfs = pd.concat(dfs, axis=0, ignore_index=True)

        # ------ step 2 find conflict
        constraints = []

        # ------ step 2.1 find possible conflict between plane and path/structor
        planeMax_node = VarTree.get_node_from_name('planeMax', name_to_nodeIdx, nodes_info)
        planeMax_xyz = VarTree.get_xyz_from_node(planeMax_node, xs, name_to_nodeIdx, nodes_info, 1)
        planeMin_node = VarTree.get_node_from_name('planeMin', name_to_nodeIdx, nodes_info)
        planeMin_xyz = VarTree.get_xyz_from_node(planeMin_node, xs, name_to_nodeIdx, nodes_info, 1)

        for name in dfs['name'].unique():
            cur_df: pd.DataFrame = dfs[dfs['name'] == name]

            # ------ is structor
            if name in structor_names:
                tag_node = nodes_info[name_to_nodeIdx[name]]
                plane_constraints = self.get_structor_conflict_plane(
                    tag_node, cur_df, planeMax_xyz, planeMin_xyz, opt_step
                )
                constraints.extend(plane_constraints)
            else:
                plane_constraints = self.get_path_conflict_plane(cur_df, planeMax_xyz, planeMin_xyz, opt_step)
                constraints.extend(plane_constraints)

        # ------ step 2.2 find possible conflict between structor and path
        shape_constraints = self.get_shape_conflict(dfs, opt_step, path_names, structor_names)
        constraints.extend(shape_constraints)

        return constraints

    def plot_env(
            self, xs, var_tree, constraints, paths_cfg, with_path=False, with_structor=False,
            with_bound=False, with_constraint=False, with_control_points=False, with_tube=False
    ):
        name_to_nodeIdx = var_tree.name_to_nodeIdx
        nodes_info = var_tree.nodes_info

        vis = ConnectVisulizer()
        obj_dict = {}
        for path_idx in paths_cfg.keys():
            path_cfg = paths_cfg[path_idx]

            src_node = nodes_info[path_cfg['src_node_idx']]
            src_x, src_y, src_z = VarTree.get_xyz_from_node(src_node, xs, name_to_nodeIdx, nodes_info, 1)
            end_node = nodes_info[path_cfg['end_node_idx']]
            end_x, end_y, end_z = VarTree.get_xyz_from_node(end_node, xs, name_to_nodeIdx, nodes_info, 1)
            cell_xyzs = xs[path_cfg['path_cols']].reshape((-1, 3))
            c_mat = np.concatenate((
                [[src_x, src_y, src_z]],
                cell_xyzs,
                [[end_x, end_y, end_z]]
            ), axis=0)
            path_xyzs = path_cfg['b_mat'].dot(c_mat)
            obj_dict[path_cfg['name']] = path_xyzs

            if with_path:
                vis.plot_connect(path_xyzs, color=np.array([0., 0., 1.]), opacity=1.0)

            if with_control_points:
                vis.plot_connect(c_mat, color=np.array([1., 0., 1.]), opacity=1.0)

            if with_tube:
                vis.plot_tube(path_xyzs, radius=path_cfg['radius'], color=np.array([0.8, 0.5, 0.3]), opacity=0.2)

        structor_names = {}
        for node_idx in nodes_info.keys():
            node = nodes_info[node_idx]
            if node['node_type'] != 'structor':
                continue

            pose_xyz = VarTree.get_xyz_from_node(node, xs, name_to_nodeIdx, nodes_info, 1)
            pose_xyz = np.array(pose_xyz)
            pcd_world = pose_xyz + node['shape_pcd']
            obj_dict[node['name']] = pcd_world
            structor_names[node['name']] = pose_xyz

        planeMax_node = VarTree.get_node_from_name('planeMax', name_to_nodeIdx, nodes_info)
        planeMax_xyz = VarTree.get_xyz_from_node(planeMax_node, xs, name_to_nodeIdx, nodes_info, 1)
        planeMin_node = VarTree.get_node_from_name('planeMin', name_to_nodeIdx, nodes_info)
        planeMin_xyz = VarTree.get_xyz_from_node(planeMin_node, xs, name_to_nodeIdx, nodes_info, 1)

        if with_bound:
            vis.plot_bound(
                planeMin_xyz[0], planeMin_xyz[1], planeMin_xyz[2],
                planeMax_xyz[0], planeMax_xyz[1], planeMax_xyz[2],
                color=np.array([0.5, 0.75, 1.0])
            )

        if with_constraint:
            for cfg in constraints:
                if cfg['type'] == 'shape_conflict':
                    info_0, info_1, threshold = cfg['info_0'], cfg['info_1'], cfg['threshold']
                    xyz0 = obj_dict[info_0['name']][info_0['group_t_idx'], :]
                    xyz1 = obj_dict[info_1['name']][info_1['group_t_idx'], :]

                    # try:
                    #     assert np.all(np.isclose(xyz0, np.array(info_0['debug_xyz'])))
                    # except Exception as e:
                    #     print(info_0)
                    #     print(xyz0)
                    #     raise ValueError(e)
                    #
                    # try:
                    #     assert np.all(np.isclose(xyz1, np.array(info_1['world_xyz'])))
                    # except Exception as e:
                    #     print(info_1)
                    #     print(xyz1)
                    #     raise ValueError(e)

                    vis.plot_connect(np.array([xyz0, xyz1]), color=np.array([0., 1., 0.]))

                elif cfg['type'] == 'plane_max_conflict':
                    info, xyz_tag = cfg['info'], cfg['xyz_tag']
                    xyz = obj_dict[info['name']][info['group_t_idx'], :]

                    if xyz_tag == 'x':
                        xyzs = np.array([xyz, [planeMax_xyz[0], xyz[1], xyz[2]]])
                    elif xyz_tag == 'y':
                        xyzs = np.array([xyz, [xyz[0], planeMax_xyz[1], xyz[2]]])
                    else:
                        xyzs = np.array([xyz, [xyz[0], xyz[1], planeMax_xyz[2]]])

                    vis.plot_connect(xyzs, color=np.array([1., 0.5, 0.]))

                elif cfg['type'] == 'plane_min_conflict':
                    info, xyz_tag = cfg['info'], cfg['xyz_tag']
                    xyz = obj_dict[info['name']][info['group_t_idx'], :]

                    if xyz_tag == 'x':
                        xyzs = np.array([xyz, [planeMin_xyz[0], xyz[1], xyz[2]]])
                    elif xyz_tag == 'y':
                        xyzs = np.array([xyz, [xyz[0], planeMin_xyz[1], xyz[2]]])
                    else:
                        xyzs = np.array([xyz, [xyz[0], xyz[1], planeMin_xyz[2]]])

                    vis.plot_connect(xyzs, color=np.array([0., 0.5, 1.]))

        if with_structor:
            for name in structor_names.keys():
                pcd_world = obj_dict[name]
                radius = np.min(np.max(pcd_world, axis=0) - np.min(pcd_world, axis=0)) * 0.25
                vis.plot_structor(
                    xyz=structor_names[name], radius=radius, shape_xyzs=pcd_world,
                    color=np.array([0.5, 0.5, 0.5]), with_center=True
                )

        vis.show()

    def record_video(
            self,
            vis: RecordAnysisVisulizer,
            xs, var_tree, paths_cfg, with_path=False, with_structor=False,
            with_bound=False, with_control_points=False, with_tube=False,
    ):
        vis.clear_objs()

        name_to_nodeIdx = var_tree.name_to_nodeIdx
        nodes_info = var_tree.nodes_info

        obj_dict = {}
        for path_idx in paths_cfg.keys():
            path_cfg = paths_cfg[path_idx]

            src_node = nodes_info[path_cfg['src_node_idx']]
            src_x, src_y, src_z = VarTree.get_xyz_from_node(src_node, xs, name_to_nodeIdx, nodes_info, 1)
            end_node = nodes_info[path_cfg['end_node_idx']]
            end_x, end_y, end_z = VarTree.get_xyz_from_node(end_node, xs, name_to_nodeIdx, nodes_info, 1)
            cell_xyzs = xs[path_cfg['path_cols']].reshape((-1, 3))
            c_mat = np.concatenate((
                [[src_x, src_y, src_z]],
                cell_xyzs,
                [[end_x, end_y, end_z]]
            ), axis=0)
            path_xyzs = path_cfg['b_mat'].dot(c_mat)
            obj_dict[path_cfg['name']] = path_xyzs

            if with_path:
                vis.plot_connect(path_xyzs, color=np.array([0., 0., 1.]), opacity=1.0)

            if with_control_points:
                vis.plot_connect(c_mat, color=np.array([1., 0., 1.]), opacity=1.0)

            if with_tube:
                vis.plot_tube(path_xyzs, radius=path_cfg['radius'], color=np.array([0.8, 0.5, 0.3]), opacity=0.2)

        structor_names = {}
        for node_idx in nodes_info.keys():
            node = nodes_info[node_idx]
            if node['node_type'] != 'structor':
                continue

            pose_xyz = VarTree.get_xyz_from_node(node, xs, name_to_nodeIdx, nodes_info, 1)
            pose_xyz = np.array(pose_xyz)
            pcd_world = pose_xyz + node['shape_pcd']
            obj_dict[node['name']] = pcd_world
            structor_names[node['name']] = pose_xyz

        planeMax_node = VarTree.get_node_from_name('planeMax', name_to_nodeIdx, nodes_info)
        planeMax_xyz = VarTree.get_xyz_from_node(planeMax_node, xs, name_to_nodeIdx, nodes_info, 1)
        planeMin_node = VarTree.get_node_from_name('planeMin', name_to_nodeIdx, nodes_info)
        planeMin_xyz = VarTree.get_xyz_from_node(planeMin_node, xs, name_to_nodeIdx, nodes_info, 1)

        if with_bound:
            vis.plot_bound(
                planeMin_xyz[0], planeMin_xyz[1], planeMin_xyz[2],
                planeMax_xyz[0], planeMax_xyz[1], planeMax_xyz[2],
                color=np.array([0.5, 0.75, 1.0])
            )

        if with_structor:
            for name in structor_names.keys():
                pcd_world = obj_dict[name]
                radius = np.min(np.max(pcd_world, axis=0) - np.min(pcd_world, axis=0)) * 0.25
                vis.plot_structor(
                    xyz=structor_names[name], radius=radius, shape_xyzs=pcd_world,
                    color=np.array([0.5, 0.5, 0.5]), with_center=True
                )

    def update_node_info(self, cfg, xs, nodes_info, name_to_nodeIdx):
        node = VarTree.get_node_from_name(cfg['name'], name_to_nodeIdx, nodes_info)
        new_x, new_y, new_z = VarTree.get_xyz_from_node(node, xs, name_to_nodeIdx, nodes_info, 1)
        cfg['position'] = np.array([new_x, new_y, new_z])
        return cfg

    def save_to_npy(self, xs, var_tree, constraints, paths_cfg, file_path):
        name_to_nodeIdx = var_tree.name_to_nodeIdx
        nodes_info = var_tree.nodes_info

        record_dict = {}
        for path_idx in paths_cfg.keys():
            path_cfg = paths_cfg[path_idx]

            src_node = nodes_info[path_cfg['src_node_idx']]
            src_x, src_y, src_z = VarTree.get_xyz_from_node(src_node, xs, name_to_nodeIdx, nodes_info, 1)
            end_node = nodes_info[path_cfg['end_node_idx']]
            end_x, end_y, end_z = VarTree.get_xyz_from_node(end_node, xs, name_to_nodeIdx, nodes_info, 1)
            cell_xyzs = xs[path_cfg['path_cols']].reshape((-1, 3))
            c_mat = np.concatenate((
                [[src_x, src_y, src_z]],
                cell_xyzs,
                [[end_x, end_y, end_z]]
            ), axis=0)
            path_xyzs = path_cfg['b_mat'].dot(c_mat)
            record_dict[path_cfg['name']] = {
                'type': 'path', 'xyzs': path_xyzs, 'radius': path_cfg['radius']
            }

        for node_idx in nodes_info.keys():
            node = nodes_info[node_idx]
            if node['node_type'] != 'structor':
                continue

            pose_xyz = VarTree.get_xyz_from_node(node, xs, name_to_nodeIdx, nodes_info, 1)
            pose_xyz = np.array(pose_xyz)
            pcd_world = pose_xyz + node['shape_pcd']
            record_dict[node['name']] = {
                'type': 'structor', 'xyzs': pcd_world, 'center': pose_xyz
            }

        planeMax_node = VarTree.get_node_from_name('planeMax', name_to_nodeIdx, nodes_info)
        planeMax_xyz = VarTree.get_xyz_from_node(planeMax_node, xs, name_to_nodeIdx, nodes_info, 1)
        planeMin_node = VarTree.get_node_from_name('planeMin', name_to_nodeIdx, nodes_info)
        planeMin_xyz = VarTree.get_xyz_from_node(planeMin_node, xs, name_to_nodeIdx, nodes_info, 1)
        record_dict['planeMax'] = {'type': 'planeMax', 'xyzs': planeMax_xyz}
        record_dict['planeMin'] = {'type': 'planeMin', 'xyzs': planeMin_xyz}

        for cfg in constraints:
            if cfg['type'] == 'shape_conflict':
                info_0, info_1, threshold = cfg['info_0'], cfg['info_1'], cfg['threshold']
                info_0['xyz'] = record_dict[info_0['name']]['xyzs'][info_0['group_t_idx'], :]
                info_1['xyz'] = record_dict[info_1['name']]['xyzs'][info_1['group_t_idx'], :]

            elif cfg['type'] == 'plane_max_conflict':
                info, xyz_tag = cfg['info'], cfg['xyz_tag']
                info['xyz'] = record_dict[info['name']]['xyzs'][info['group_t_idx'], :]

            elif cfg['type'] == 'plane_min_conflict':
                info, xyz_tag = cfg['info'], cfg['xyz_tag']
                info['xyz'] = record_dict[info['name']]['xyzs'][info['group_t_idx'], :]

        npy_dict = {
            'objs': record_dict,
            'constraints': constraints
        }
        np.save(file_path, npy_dict)

    def plot_npy_record(self, file_path):
        vis = ConnectVisulizer()

        record_dict = np.load(file_path, allow_pickle=True).item()
        objs_dict: dict = record_dict['objs']
        constraints: dict = record_dict['constraints']

        planeMin_xyz, planeMax_xyz = None, None
        for name in objs_dict.keys():
            info = objs_dict[name]
            if info['type'] == 'path':
                vis.plot_connect(info['xyzs'], color=np.array([0., 0., 1.]), opacity=1.0)
                vis.plot_tube(info['xyzs'], radius=info['radius'], color=np.array([0.8, 0.5, 0.3]), opacity=0.2)

            elif info['type'] == 'structor':
                pcd_world = info['xyzs']
                radius = np.min(np.max(pcd_world, axis=0) - np.min(pcd_world, axis=0)) * 0.25
                vis.plot_structor(
                    xyz=info['center'], radius=radius, shape_xyzs=pcd_world,
                    color=np.array([0.5, 0.5, 0.5]), with_center=True
                )
            elif info['type'] == 'planeMax':
                planeMax_xyz = info['xyzs']
            elif info['type'] == 'planeMin':
                planeMin_xyz = info['xyzs']
            else:
                raise NotImplementedError

        vis.plot_bound(
            planeMin_xyz[0], planeMin_xyz[1], planeMin_xyz[2],
            planeMax_xyz[0], planeMax_xyz[1], planeMax_xyz[2],
            color=np.array([0.5, 0.75, 1.0])
        )

        for cfg in constraints:
            if cfg['type'] == 'shape_conflict':
                info_0, info_1, threshold = cfg['info_0'], cfg['info_1'], cfg['threshold']
                vis.plot_connect(np.array([info_0['xyz'], info_1['xyz']]), color=np.array([0., 1., 0.]))

            elif cfg['type'] == 'plane_max_conflict':
                info, xyz_tag = cfg['info'], cfg['xyz_tag']
                xyz = info['xyz']
                if xyz_tag == 'x':
                    xyzs = np.array([xyz, [planeMax_xyz[0], xyz[1], xyz[2]]])
                elif xyz_tag == 'y':
                    xyzs = np.array([xyz, [xyz[0], planeMax_xyz[1], xyz[2]]])
                else:
                    xyzs = np.array([xyz, [xyz[0], xyz[1], planeMax_xyz[2]]])
                vis.plot_connect(xyzs, color=np.array([1., 0.5, 0.]))

            elif cfg['type'] == 'plane_min_conflict':
                info, xyz_tag = cfg['info'], cfg['xyz_tag']
                xyz = info['xyz']
                if xyz_tag == 'x':
                    xyzs = np.array([xyz, [planeMin_xyz[0], xyz[1], xyz[2]]])
                elif xyz_tag == 'y':
                    xyzs = np.array([xyz, [xyz[0], planeMin_xyz[1], xyz[2]]])
                else:
                    xyzs = np.array([xyz, [xyz[0], xyz[1], planeMin_xyz[2]]])
                vis.plot_connect(xyzs, color=np.array([0., 0.5, 1.]))

        vis.show()


if __name__ == '__main__':
    pass
