import math
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from copy import deepcopy

from scipy import optimize
from scipy.optimize import Bounds
from sklearn.neighbors import KDTree

from scripts.version_9.app.env_utils import Shape_Utils
from scripts.version_9.springer.smoother_utils import ConnectVisulizer


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
            var_info.update({'shape_pcd': node_info['shape_pcd'], 'exclude_edges': node_info['exclude_edges']})

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

        raise NotImplementedError

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
    def cost_path_length(path_xyz):
        cost = np.sum(np.power(path_xyz[1:, :] - path_xyz[:-1, :], 2))
        return cost

    """
    @staticmethod
    def cost_path_curvature(path_xyz, k=1.0 / 3.0):
        vec01 = path_xyz[1:-1, :] - path_xyz[:-2, :]
        vec12 = path_xyz[2:, :] - path_xyz[1:-1, :]
        length0 = np.power(vec01, 1.0 / 2.0)
        length1 = np.power(vec12, 1.0 / 2.0)
        cos_theta = np.sum(vec01 * vec12, axis=1) / (length0 * length1)
        curvature = np.arccos(cos_theta) / np.min(length0, length1, axis=1)
        return np.sum(np.maximum(curvature - k, 0.0))
    """

    @staticmethod
    def penalty_shape_conflict(xyz0, xyz1, threshold, method, weight=300.0):
        dist = np.sqrt(np.sum(np.power(xyz0 - xyz1, 2.0)))
        penalty = OptimizerScipy.penalty_bound_func(dist, threshold, weight=weight, method=method)
        return penalty

    @staticmethod
    def penalty_plane_conflict(v0, v1, threshold, method, weight=300.0):
        penalty = OptimizerScipy.penalty_bound_func(v0 - v1, threshold, weight=weight, method=method)
        return penalty

    def target_func(
            self, xs, xs_init,
            paths_cfg, constraints, structor_nodes,
            name_to_nodeIdx, nodes_info,
            opt_step, penalty_weight, volume_cost_weight, debug_vis=False
    ):
        xs_new = xs + xs_init

        # ------ step 1 reconstruct env
        # ------ step 1.1 reconstruct path
        obj_dict, path_names = {}, []
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

        # ------ step 1.2 reconstruct structor
        for node in structor_nodes:
            pose_xyz = VarTree.get_xyz_from_node(node, xs_new, name_to_nodeIdx, nodes_info, 1)
            pose_xyz = np.array(pose_xyz)
            obj_dict[node['name']] = pose_xyz + node['shape_pcd']

        # self.plot_env(path_dict, structor_dict)

        # ------ step 2 compute cost
        # ------ step 2.1 compute path length cost
        path_cost = 0.0
        for path_name in path_names:
            path_xyzs = obj_dict[path_name]
            path_cost += self.cost_path_length(path_xyzs)

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

        cost = path_cost + penalty_cost * penalty_weight + volume_cost * volume_cost_weight
        # cost = path_cost + penalty_cost * penalty_weight

        if debug_vis:
            print(f"[Info] Path Cost:{path_cost} penalty_cost:{penalty_cost} "
                  f"volume_cost:{volume_cost * volume_cost_weight}")

        return cost

    @staticmethod
    def log_info(xs, info, cost_func):
        print(f"Iter:{info['iter']} Cost:{cost_func(xs)}")
        info['iter'] += 1

    def solve_problem(self, xs_init, var_tree: VarTree, paths_cfg, constraints, opt_step):
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
            'volume_cost_weight': 0.1,
            'penalty_weight': 10.0,
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
        log_problem(res.x)
        # print('------')

        opt_xs = res.x
        if res.success:
            print(f"[Debug]: OptCost:{res.fun} optXs:{opt_xs.min()}->{opt_xs.max()}")
        else:
            print(f"[Debug]: State:{res.success} StateCode:{res.status} OptCost:{res.fun} "
                  f"optXs:{opt_xs.min()}->{opt_xs.max()} msg:{res.message}")

        return opt_xs, problem(opt_xs)

    @staticmethod
    def plot_env(path_dict, structor_dict):
        vis = ConnectVisulizer()

        ramdom_colors = np.random.random(size=(len(path_dict), 3))
        for i, path_idx in enumerate(path_dict.keys()):
            path_xyzs = path_dict[path_idx]
            vis.plot_connect(path_xyzs, color=ramdom_colors[i], opacity=1.0)
        vis.show()

        ramdom_colors = np.random.random(size=(len(structor_dict), 3))
        for i, key in enumerate(structor_dict.keys()):
            pcd_world = structor_dict[key]
            vis.plot_structor(
                xyz=None, radius=None, shape_xyzs=pcd_world,
                color=ramdom_colors[i, :], with_center=False
            )

        vis.show()


class EnvParser(object):
    def __init__(self):
        self.var_tree = VarTree()

    def create_vars_tree(self, pipes_cfg, structos_cfg, bgs_cfg):
        for cfg in pipes_cfg:
            self.var_tree.add_node(cfg)

        for cfg in structos_cfg:
            self.var_tree.add_node(cfg)

        for cfg in bgs_cfg:
            self.var_tree.add_node(cfg)

    def define_path(self, paths_cfg, k=3):
        name_to_nodeIdx = self.var_tree.name_to_nodeIdx
        nodes_info = self.var_tree.nodes_info
        xs = self.var_tree.get_xs_init()

        for path_idx in paths_cfg.keys():
            path_cfg = paths_cfg[path_idx]
            xyzs = path_cfg['xyzs']
            src_name, end_name = path_cfg['src_name'], path_cfg['end_name']

            # ------ check path
            src_node = VarTree.get_node_from_name(src_name, name_to_nodeIdx, nodes_info)
            end_node = VarTree.get_node_from_name(end_name, name_to_nodeIdx, nodes_info)
            src_xyz = VarTree.get_xyz_from_node(src_node, xs, name_to_nodeIdx, nodes_info, 1)
            end_xyz = VarTree.get_xyz_from_node(end_node, xs, name_to_nodeIdx, nodes_info, 1)
            # print(src_xyz, xyzs[0], end_xyz, xyzs[-1])
            assert np.all(src_xyz == xyzs[0]) and np.all(end_xyz == xyzs[-1])

            path_cols = []
            for i, cell_xyz in enumerate(xyzs[1:-1]):
                cell_node = self.var_tree.add_node({
                    'node_type': 'cell', 'position': cell_xyz, 'name': f"{path_cfg['name']}_cell{i + 1}",
                    'pose_edge_x': {'type': 'var_value'},
                    'pose_edge_y': {'type': 'var_value'},
                    'pose_edge_z': {'type': 'var_value'}
                })
                xyz_col = VarTree.get_xyz_from_node(cell_node, xs, name_to_nodeIdx, nodes_info, return_method=2)
                path_cols.append(xyz_col)
            path_cols = np.array(path_cols).reshape(-1)

            # path_length = BSpline_utils.compute_spline_length(c_mat=xyzs, k=k)
            # sample_num = math.ceil(path_length / path_cfg['radius'] * 1.15)
            # sample_num = max(sample_num, 2)
            sample_num = 30

            b_mat = BSpline_utils.compute_Bmat_uniform(c_mat=xyzs, k=k, sample_num=sample_num)

            path_cfg.update({
                'src_node_idx': src_node['idx'], 'end_node_idx': end_node['idx'],
                'path_cols': path_cols, 'b_mat': b_mat
            })

        return paths_cfg

    def update_Bspline_mat(self, xs, paths_cfg, k=3):
        name_to_nodeIdx = self.var_tree.name_to_nodeIdx
        nodes_info = self.var_tree.nodes_info

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

            path_length = BSpline_utils.compute_spline_length(c_mat=c_mat, k=k)
            sample_num = math.ceil(path_length / path_cfg['radius'] * 1.15)
            sample_num = max(sample_num, c_mat.shape[0])
            b_mat = BSpline_utils.compute_Bmat_uniform(c_mat=c_mat, k=k, sample_num=sample_num)

            path_cfg.update({'b_mat': b_mat})

        return paths_cfg

    def get_structor_plane_constraints(self, node, node_df: pd.DataFrame, planeMax_xyz, planeMin_xyz, opt_step):
        constraints = []
        xyzs = node_df[['x', 'y', 'z']].values

        exclude_tags = []
        if 'plane_max_conflict' in node['exclude_edges'].keys():
            exclude_tags = node['exclude_edges']['plane_max_conflict']
        for xyz_tag, xyz_col in zip(['x', 'y', 'z'], [0, 1, 2]):
            if xyz_tag in exclude_tags:
                continue
            conflicts = xyzs[:, xyz_col] + node_df['radius'].values + opt_step >= planeMax_xyz[xyz_col]
            for i in np.nonzero(conflicts)[0]:
                cur_series = node_df.iloc[i]
                constraints.append({
                    'type': 'plane_max_conflict', 'xyz_tag': xyz_tag, 'threshold': cur_series['radius'],
                    'info': {
                        'group_t_idx': cur_series['group_t_idx'], 'name': cur_series['name'],
                        'world_xyz': [cur_series['x'], cur_series['y'], cur_series['z']], 'radius': cur_series['radius']
                    }
                })

        exclude_tags = []
        if 'plane_min_conflict' in node['exclude_edges'].keys():
            exclude_tags = node['exclude_edges']['plane_min_conflict']
        for xyz_tag, xyz_col in zip(['x', 'y', 'z'], [0, 1, 2]):
            if xyz_tag in exclude_tags:
                continue
            conflicts = xyzs[:, xyz_col] - node_df['radius'].values - opt_step <= planeMin_xyz[xyz_col]
            for i in np.nonzero(conflicts)[0]:
                cur_series = node_df.iloc[i]
                constraints.append({
                    'type': 'plane_min_conflict', 'xyz_tag': xyz_tag, 'threshold': cur_series['radius'],
                    'info': {
                        'group_t_idx': cur_series['group_t_idx'], 'name': cur_series['name'],
                        'world_xyz': [cur_series['x'], cur_series['y'], cur_series['z']], 'radius': cur_series['radius']
                    }
                })

        return constraints

    def get_path_plane_constraints(self, node_df: pd.DataFrame, planeMax_xyz, planeMin_xyz, opt_step):
        constraints = []
        xyzs = node_df[['x', 'y', 'z']].values

        for xyz_tag, xyz_col in zip(['x', 'y', 'z'], [0, 1, 2]):
            conflicts = xyzs[:, xyz_col] + node_df['radius'].values + opt_step >= planeMax_xyz[xyz_col]
            for i in np.nonzero(conflicts)[0]:
                cur_series = node_df.iloc[i]
                constraints.append({
                    'type': 'plane_max_conflict', 'xyz_tag': xyz_tag,
                    'info': {
                        'group_t_idx': cur_series['group_t_idx'], 'name': cur_series['name'],
                        'world_xyz': [cur_series['x'], cur_series['y'], cur_series['z']],
                        'radius': cur_series['radius']
                    },
                    'threshold': cur_series['radius']
                })

            conflicts = xyzs[:, xyz_col] - node_df['radius'].values - opt_step <= planeMin_xyz[xyz_col]
            for i in np.nonzero(conflicts)[0]:
                cur_series = node_df.iloc[i]
                constraints.append({
                    'type': 'plane_min_conflict', 'xyz_tag': xyz_tag,
                    'info': {
                        'group_t_idx': cur_series['group_t_idx'], 'name': cur_series['name'],
                        'world_xyz': [cur_series['x'], cur_series['y'], cur_series['z']],
                        'radius': cur_series['radius']
                    },
                    'threshold': cur_series['radius']
                })

        return constraints

    def get_constraints(self, xs, paths_cfg, opt_step=0.1):
        # 由于矩形阀块是个凸包，因此只要控制点不超出边界，路径就不会超出边界

        name_to_nodeIdx = self.var_tree.name_to_nodeIdx
        nodes_info = self.var_tree.nodes_info
        max_step_length = np.linalg.norm(np.array([opt_step, opt_step, opt_step]), ord=2) * 1.05

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
            pcd_array = np.concatenate([pcd_world, node['shape_pcd']], axis=1)
            sub_df = pd.DataFrame(pcd_array, columns=['x', 'y', 'z', 'shape_x', 'shape_y', 'shape_z'])
            sub_df[['radius', 'name', 'node_idx', 'df_type']] = 0.15, node['name'], node_idx, 'structor'
            sub_df['group_t_idx'] = np.arange(0, pcd_array.shape[0], 1)
            dfs.append(sub_df)
            structor_names.append(node['name'])

        # ------ step 1.2 record path data point
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
            cell_xyzs = path_cfg['b_mat'].dot(c_mat)

            sub_df = pd.DataFrame(cell_xyzs, columns=['x', 'y', 'z'])
            sub_df[['shape_x', 'shape_y', 'shape_z']] = 0.0, 0.0, 0.0
            sub_df[['radius', 'name', 'node_idx', 'df_type']] = path_cfg['radius'], path_cfg['name'], -1, 'path'
            sub_df['group_t_idx'] = np.arange(0, cell_xyzs.shape[0], 1)
            dfs.append(sub_df)

        dfs = pd.concat(dfs, axis=0, ignore_index=True)

        planeMax_node = VarTree.get_node_from_name('planeMax', name_to_nodeIdx, nodes_info)
        planeMax_xyz = VarTree.get_xyz_from_node(planeMax_node, xs, name_to_nodeIdx, nodes_info, 1)
        planeMin_node = VarTree.get_node_from_name('planeMin', name_to_nodeIdx, nodes_info)
        planeMin_xyz = VarTree.get_xyz_from_node(planeMin_node, xs, name_to_nodeIdx, nodes_info, 1)

        # ------ step 2 find conflict
        constraints = []

        '''
        # ------ step 2.1 find possible conflict between plane and path/structor
        for name in dfs['name'].unique():
            cur_df: pd.DataFrame = dfs[dfs['name'] == name]

            # ------ is structor
            if name in structor_names:
                tag_node = nodes_info[name_to_nodeIdx[name]]
                plane_constraints = self.get_structor_plane_constraints(
                    tag_node, cur_df, planeMax_xyz, planeMin_xyz, opt_step
                )
                constraints.extend(plane_constraints)
            else:
                plane_constraints = self.get_path_plane_constraints(cur_df, planeMax_xyz, planeMin_xyz, opt_step)
                constraints.extend(plane_constraints)
        '''

        # ------ step 2.2 find possible conflict between structor and path
        for name in dfs['name'].unique()[:-1]:
            cur_df: pd.DataFrame = dfs[dfs['name'] == name]
            other_df: pd.DataFrame = dfs[dfs['name'] != name]

            max_search_radius = cur_df['radius'].max() + other_df['radius'].max() + max_step_length
            cur_xyzs = cur_df[['x', 'y', 'z']].values
            other_xyzs = other_df[['x', 'y', 'z']].values

            other_tree = KDTree(other_xyzs)
            idxs_list, dists_list = other_tree.query_radius(cur_xyzs, max_search_radius, return_distance=True)
            for i, (idxs, dists) in enumerate(zip(idxs_list, dists_list)):
                if idxs.shape[0] == 0:
                    continue

                for idx, dist in zip(idxs, dists):
                    cur_series, other_series = cur_df.iloc[i], other_df.iloc[idx]
                    real_threshold = cur_series['radius'] + other_series['radius']
                    assert real_threshold > 0.0

                    scale_threshold = real_threshold + max_step_length
                    if dist >= scale_threshold:
                        continue

                    constraints.append({
                        'type': 'shape_conflict',
                        'info_0': {
                            'group_t_idx': cur_series['group_t_idx'], 'name': cur_series['name'],
                            'world_xyz': [cur_series['x'], cur_series['y'], cur_series['z']],
                            'radius': cur_series['radius']
                        },
                        'info_1': {
                            'group_t_idx': other_series['group_t_idx'], 'name': other_series['name'],
                            'world_xyz': [other_series['x'], other_series['y'], other_series['z']],
                            'radius': other_series['radius']
                        },
                        'threshold': real_threshold
                    })

            # ------ only contain rest part
            dfs = other_df

        return constraints

    def plot_env(
            self, xs, constraints, paths_cfg, with_path=False, with_structor=False,
            with_bound=False, with_constraint=False, with_control_points=False, with_tube=False
    ):
        name_to_nodeIdx = self.var_tree.name_to_nodeIdx
        nodes_info = self.var_tree.nodes_info

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
            vis.plot_bound(planeMax_xyz[0], planeMax_xyz[1], planeMax_xyz[2], color=np.array([0.5, 0.75, 1.0]))

        if with_constraint:
            for cfg in constraints:
                if cfg['type'] == 'shape_conflict':
                    info_0, info_1, threshold = cfg['info_0'], cfg['info_1'], cfg['threshold']
                    xyz0 = obj_dict[info_0['name']][info_0['group_t_idx'], :]
                    xyz1 = obj_dict[info_1['name']][info_1['group_t_idx'], :]

                    try:
                        assert np.all(np.isclose(xyz0, np.array(info_0['world_xyz'])))
                    except Exception as e:
                        print(info_0)
                        print(xyz0)
                        raise ValueError(e)

                    try:
                        assert np.all(np.isclose(xyz1, np.array(info_1['world_xyz'])))
                    except Exception as e:
                        print(info_1)
                        print(xyz1)
                        raise ValueError(e)

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

                    vis.plot_connect(xyzs, color=np.array([1., 1., 0.]))

                elif cfg['type'] == 'plane_min_conflict':
                    info, xyz_tag = cfg['info'], cfg['xyz_tag']
                    xyz = obj_dict[info['name']][info['group_t_idx'], :]

                    if xyz_tag == 'x':
                        xyzs = np.array([xyz, [planeMin_xyz[0], xyz[1], xyz[2]]])
                    elif xyz_tag == 'y':
                        xyzs = np.array([xyz, [xyz[0], planeMin_xyz[1], xyz[2]]])
                    else:
                        xyzs = np.array([xyz, [xyz[0], xyz[1], planeMin_xyz[2]]])

                    vis.plot_connect(xyzs, color=np.array([0., 1., 1.]))

        if with_structor:
            for name in structor_names.keys():
                pcd_world = obj_dict[name]
                radius = np.min(np.max(pcd_world, axis=0) - np.min(pcd_world, axis=0)) * 0.25
                vis.plot_structor(
                    xyz=structor_names[name], radius=radius, shape_xyzs=pcd_world,
                    color=np.array([0.5, 0.5, 0.5]), with_center=True
                )

        vis.show()

    def solve(self, paths_cfg, opt_step=0.1, run_count=300):
        xs_init = self.var_tree.get_xs_init()
        run_times = 0
        stop_tol = opt_step * 0.001
        last_cost = np.inf
        cur_paths_cfg = deepcopy(paths_cfg)

        while True:
            if run_times > 0:
                self.update_Bspline_mat(xs_init, cur_paths_cfg, k=3)

            constraints = self.get_constraints(xs_init, cur_paths_cfg, opt_step=opt_step)

            self.plot_env(
                xs=xs_init, constraints=constraints, paths_cfg=cur_paths_cfg, with_path=True,
                with_structor=True, with_bound=True, with_constraint=True, with_control_points=True,
                with_tube=True
            )

            optimizer = OptimizerScipy()
            opt_xs, cur_cost = optimizer.solve_problem(xs_init, self.var_tree, cur_paths_cfg, constraints, opt_step)

            if cur_cost > last_cost * 10.0:
                break

            xs_init += opt_xs
            paths_cfg = cur_paths_cfg

            # ------ debug
            name_to_nodeIdx, nodes_info = self.var_tree.name_to_nodeIdx, self.var_tree.nodes_info
            planeMax_node = VarTree.get_node_from_name('planeMax', name_to_nodeIdx, nodes_info)
            planeMax_xyz = VarTree.get_xyz_from_node(planeMax_node, xs_init, name_to_nodeIdx, nodes_info, 1)
            planeMin_node = VarTree.get_node_from_name('planeMin', name_to_nodeIdx, nodes_info)
            planeMin_xyz = VarTree.get_xyz_from_node(planeMin_node, xs_init, name_to_nodeIdx, nodes_info, 1)
            print(f"[Debug]: Iter:{run_times} min_xyz:({planeMin_xyz[0]}, {planeMin_xyz[1]}, {planeMin_xyz[2]}) -> "
                  f"({planeMax_xyz[0]}, {planeMax_xyz[1]}, {planeMax_xyz[2]}) cost:{last_cost} -> {cur_cost} \n")

            last_cost = cur_cost

            run_times += 1
            if run_times > run_count:
                break

            # if np.max(np.abs(opt_xs)) < stop_tol:
            #     break

        return xs_init, paths_cfg


if __name__ == '__main__':
    pass
