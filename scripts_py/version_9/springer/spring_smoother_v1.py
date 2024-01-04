import numpy as np
import pandas as pd
import networkx as nx
import math
import os
import json
import shutil
from functools import partial
from copy import copy, deepcopy

from scipy import optimize
from scipy.optimize import Bounds, NonlinearConstraint, LinearConstraint
from sklearn.neighbors import KDTree

from scripts_py.version_9.springer.smoother_utils import get_rotationMat
from scripts_py.version_9.springer.smoother_utils import ConnectVisulizer

'''
2023-09-23

Inner Point Method Fail: gradient 效率过低
Nonlinear Object + Nonlinear eq/ineq Constraint
Nonlinear Object + Nonlinear eq Constraint
Nonlinear Object + Nonlinear eq Constraint + Integer Define
Nonlinear Object + Nonlinear eq/ineq Constraint + Integer Define

先评估简单问题，不考虑 radius fix 约束与xyzr
UIOP
'''


class ConstraintParser(object):
    def __init__(self):
        self.nodes_info = {}
        self.xyz_to_nodeIdx = {}
        self.name_to_nodeIdx = {}

        self.paths_info = {}
        self.network = nx.Graph()

    def create_connective_graph(self, env_cfg: dict, resGroupPaths: dict):
        for groupIdx in resGroupPaths.keys():
            path_infos = resGroupPaths[groupIdx]

            for path_idx, path_info in enumerate(path_infos):
                last_node_idx = None
                path_xyzrl = path_info['path_xyzrl']

                for i, (x, y, z, radius, length) in enumerate(path_xyzrl):
                    is_connector = False

                    if i == 0:
                        name = path_info['name0']
                        info = env_cfg['pipe_cfgs'][str(groupIdx)][name]
                        if np.all(np.array([x, y, z]) == np.array(info['position'])):
                            x, y, z = info['scale_position']
                            is_connector = True

                    if i == path_xyzrl.shape[0] - 1:
                        name = path_info['name1']
                        info = env_cfg['pipe_cfgs'][str(groupIdx)][name]
                        if np.all(np.array([x, y, z]) == np.array(info['position'])):
                            x, y, z = info['scale_position']
                            is_connector = True

                    if (x, y, z) not in self.xyz_to_nodeIdx.keys():
                        cur_node_idx = len(self.nodes_info.keys())

                        if is_connector:
                            self.nodes_info[cur_node_idx] = {
                                'vertex': info['vertex'],
                                'poseEdges': info['poseEdges'],
                                'node_idx': cur_node_idx,
                                'groupIdx': groupIdx,
                                'desc': {
                                    'name': name,
                                    'position': np.array([x, y, z]),
                                    'direction': info['direction'],
                                    'radius': info['radius']
                                }
                            }
                            self.name_to_nodeIdx[name] = cur_node_idx
                            self.xyz_to_nodeIdx[(x, y, z)] = cur_node_idx

                        else:
                            name = f"cell_{cur_node_idx}"
                            self.nodes_info[cur_node_idx] = {
                                'vertex': {
                                    "type": "cell",
                                    "dim": "xyz",
                                },
                                'node_idx': cur_node_idx,
                                'groupIdx': groupIdx,
                                'desc': {
                                    'name': name,
                                    'position': np.array([x, y, z]),
                                    'radius': radius
                                }
                            }
                            self.name_to_nodeIdx[name] = cur_node_idx
                            self.xyz_to_nodeIdx[(x, y, z)] = cur_node_idx

                        self.network.add_node(cur_node_idx)

                    else:
                        cur_node_idx = self.xyz_to_nodeIdx[(x, y, z)]
                        if self.nodes_info[cur_node_idx]['vertex']['type'] == 'cell':
                            self.nodes_info[cur_node_idx]['desc']['radius'] = max(
                                self.nodes_info[cur_node_idx]['desc']['radius'], radius
                            )

                    if last_node_idx is not None:
                        self.network.add_edge(last_node_idx, cur_node_idx)
                    last_node_idx = cur_node_idx

        self.xyz_to_nodeIdx.clear()

        # ------ load shape point cloud
        shape_pcd_df = pd.read_csv(env_cfg['obstacle_path'], index_col=0)
        shape_pcd_df['x'] = shape_pcd_df['x'] * env_cfg['global_params']['scale']
        shape_pcd_df['y'] = shape_pcd_df['y'] * env_cfg['global_params']['scale']
        shape_pcd_df['z'] = shape_pcd_df['z'] * env_cfg['global_params']['scale']

        # ------ load structor node
        for structor_name in env_cfg['obstacle_cfgs'].keys():
            cur_node_idx = len(self.nodes_info.keys())
            info = env_cfg['obstacle_cfgs'][structor_name]
            if 'vertex' not in info.keys():
                continue

            desc_info = {
                'name': structor_name,
                'position': np.array(info['position']),
                'search_radius': 0.5,
            }
            if info['vertex']['type'] == "xyzr":
                # desc_info.update({
                #     'xyz_tag': None,
                #     'direction': info['scale_desc']['direction'],
                #     'radian': 0.0
                # })
                raise NotImplementedError

            shape_pcds = []
            for shape_name in info['vertex']['shape_pcd']:
                xyzs = shape_pcd_df[shape_pcd_df['tag'] == shape_name][['x', 'y', 'z']].values
                shape_pcds.append(xyzs)
            shape_pcds = np.concatenate(shape_pcds, axis=0)
            shape_pcds = shape_pcds - desc_info['position']
            desc_info.update({'shape_pcd': shape_pcds})

            self.nodes_info[cur_node_idx] = {
                'node_idx': cur_node_idx,
                'vertex': info['vertex'],
                'poseEdges': info['poseEdges'],
                'desc': desc_info
            }

        env_params = env_cfg['global_params']
        cur_node_idx = len(self.nodes_info)
        self.nodes_info[cur_node_idx] = {
            'vertex': {
                "type": "planeMax",
                "dim": "xyz",
            },
            'node_idx': cur_node_idx,
            'desc': {
                'name': 'planeMax',
                'position': np.array([env_params['envX'], env_params['envY'], env_params['envZ']]),
            }
        }
        self.name_to_nodeIdx["planeMax"] = cur_node_idx

        cur_node_idx = len(self.nodes_info)
        self.nodes_info[cur_node_idx] = {
            'vertex': {
                "type": "planeMin",
                "dim": "xyz",
            },
            'node_idx': cur_node_idx,
            'desc': {
                'name': 'planeMin',
                'position': np.array([0., 0., 0.]),
            }
        }
        self.name_to_nodeIdx["planeMin"] = cur_node_idx

    def define_path(self, env_cfg: dict, path_links):
        for groupIdx in path_links.keys():
            link_info = path_links[groupIdx]

            start_name = link_info['converge_pipe']
            start_node_idx = self.name_to_nodeIdx[start_name]

            for end_name in link_info['branch_pipes']:
                branch_info = link_info['branch_pipes'][end_name]
                end_node_idx = self.name_to_nodeIdx[end_name]

                path_node_idxs = nx.shortest_path(self.network, start_node_idx, end_node_idx)
                path_size = len(path_node_idxs)
                if path_size == 0:
                    raise ValueError('[Error]: inValid Path')

                path_idx = len(self.paths_info)
                self.paths_info[path_idx] = {
                    'groupIdx': groupIdx,
                    'start_node_idx': start_node_idx,
                    'end_node_idx': end_node_idx,
                    "startFlexRatio": 0.0,
                    "endFlexRatio": branch_info['flexRatio'],
                    "graph_node_idxs": path_node_idxs,
                }

    def create_hyper_vertex(self):
        shift_num, new_num = len(self.nodes_info), 0
        nodes_info, name_to_nodeIdx = {}, {}

        for path_idx in self.paths_info.keys():
            path_info = self.paths_info[path_idx]

            graph_node_idxs = path_info["graph_node_idxs"]
            path_size = len(graph_node_idxs)

            start_flex_num = 0
            end_flex_num = min(max(math.floor(path_size * path_info['endFlexRatio']), 6), path_size - 2)
            end_flex_num = path_size - end_flex_num

            smooth_node_idxs = []
            for i, node_idx in enumerate(graph_node_idxs):
                node: dict = self.nodes_info[node_idx]

                if node['vertex']['type'] == "cell":
                    if i < start_flex_num or i >= end_flex_num:
                        vertex_idx = shift_num + new_num
                        new_num += 1

                        new_node_info = node.copy()
                        name = f"cell_{vertex_idx}"
                        new_node_info['desc'].update({
                            'name': name,
                            'node_idx': vertex_idx
                        })
                        nodes_info[vertex_idx] = new_node_info
                        name_to_nodeIdx[name] = vertex_idx

                    else:
                        vertex_idx = node_idx
                        if vertex_idx not in nodes_info.keys():
                            nodes_info[vertex_idx] = node.copy()
                            name_to_nodeIdx[node['desc']['name']] = vertex_idx

                else:
                    vertex_idx = node_idx
                    if vertex_idx not in nodes_info.keys():
                        nodes_info[vertex_idx] = node.copy()
                        name_to_nodeIdx[node['desc']['name']] = vertex_idx

                smooth_node_idxs.append(vertex_idx)
            path_info.update({'graph_node_idxs': smooth_node_idxs})

        for node_idx in self.nodes_info.keys():
            node = self.nodes_info[node_idx]
            if node['vertex']['type'] in ['structor', 'planeMax', 'planeMin']:
                assert node['node_idx'] not in nodes_info.keys()
                nodes_info[node_idx] = node.copy()
                name_to_nodeIdx[node['desc']['name']] = node_idx

        self.nodes_info = nodes_info
        self.name_to_nodeIdx = name_to_nodeIdx

    def get_plane_constraints(self, scale):
        planeMax_node_idx = self.name_to_nodeIdx['planeMax']
        planeMax_node = self.nodes_info[planeMax_node_idx]
        planeMax_xyz = np.array(planeMax_node['desc']['position'])

        plane_constraints = []
        for node_idx in self.nodes_info.keys():
            node = self.nodes_info[node_idx]

            if node['vertex']['type'] == 'cell':
                pose, radius = node['desc']['position'], node['desc']['radius']
                for xyz_col, plane_tag in enumerate(['x', 'y', 'z']):
                    if pose[xyz_col] - radius - scale <= 0.:
                        plane_constraints.append({
                            'type': 'planeMin_conflict', 'plane_tag': plane_tag,
                            'node_idx': node_idx, 'info': f"{node['desc']['name']}.{plane_tag} - radius >= 0.0",
                            'detail': {
                                'position': node['desc']['position'],
                                'radius': node['desc']['radius'],
                            }
                        })

                    if pose[xyz_col] + radius >= planeMax_xyz[xyz_col] - scale:
                        plane_constraints.append({
                            'type': 'planeMax_conflict', 'plane_tag': plane_tag,
                            'node_idx': node_idx, 'ref_node_idx': planeMax_node_idx,
                            'info': f"{node['desc']['name']}.{plane_tag} + radius <= planeMax.{plane_tag}",
                            'detail': {
                                'position': node['desc']['position'],
                                'radius': node['desc']['radius'],
                                'planeMax': planeMax_xyz
                            }
                        })

            elif node['vertex']['type'] == 'structor':
                shape_pcd = node['desc']['shape_pcd']
                shape_idxs = np.arange(0, shape_pcd.shape[0], 1)

                if node['vertex']['dim'] == 'xyz':
                    world_shape = np.array(node['desc']['position']) + shape_pcd
                else:
                    rot_mat = get_rotationMat(node['desc']['xyz_tag'], node['desc']['radian'])
                    world_shape = np.array(node['desc']['position']) + (rot_mat.dot(shape_pcd.T)).T

                planeMin_dict = {'x': 0, 'y': 1, 'z': 2}
                planeMax_dict = {'x': 0, 'y': 1, 'z': 2}
                for edge_info in node['poseEdges']:
                    if edge_info['type'] in ['value_shift']:
                        if (edge_info['ref_objName'] == 'planeMin') and (edge_info['xyzTag'] in planeMin_dict.keys()):
                            del planeMin_dict[edge_info['xyzTag']]
                        if (edge_info['ref_objName'] == 'planeMax') and (edge_info['xyzTag'] in planeMax_dict.keys()):
                            del planeMax_dict[edge_info['xyzTag']]

                for planeTag in planeMin_dict.keys():
                    xyz_col = planeMin_dict[planeTag]
                    conflict_idxs = shape_idxs[world_shape[:, xyz_col] - scale <= 0]

                    if node['vertex']['dim'] == 'xyz':
                        for conflict_idx in conflict_idxs:
                            plane_constraints.append({
                                'type': 'planeMin_conflict', 'plane_tag': planeTag,
                                'node_idx': node_idx,
                                'shape_x': shape_pcd[conflict_idx, 0],
                                'shape_y': shape_pcd[conflict_idx, 1],
                                'shape_z': shape_pcd[conflict_idx, 2],
                                'info': f"{node['desc']['name']}.{planeTag} + shape_{planeTag} >= 0.0",
                                'detail': {
                                    'position': node['desc']['position'],
                                }
                            })
                    elif node['vertex']['dim'] == 'xyzr':
                        # for conflict_idx in conflict_idxs:
                        #     plane_constraints.append({
                        #         'type': 'planeMin_conflict', 'plane_tag': planeTag, 'node_idx': node_idx,
                        #         'shape_x': shape_pcd[conflict_idx, 0],
                        #         'shape_y': shape_pcd[conflict_idx, 1],
                        #         'shape_z': shape_pcd[conflict_idx, 2],
                        #         'info': f"({node['desc']['name']}.xyz + "
                        #                 f"rotMat_{node['desc']['xyzTag']}.dot(shape_xyz.T).T).{planeTag} >= 0.0"
                        #     })
                        raise NotImplementedError

                for planeTag in planeMax_dict.keys():
                    xyz_col = planeMax_dict[planeTag]
                    conflict_idxs = shape_idxs[world_shape[:, xyz_col] + scale >= planeMax_xyz[xyz_col]]

                    if node['vertex']['dim'] == 'xyz':
                        for conflict_idx in conflict_idxs:
                            plane_constraints.append({
                                'type': 'planeMax_conflict', 'plane_tag': planeTag,
                                'node_idx': node_idx, 'ref_node_idx': planeMax_node_idx,
                                'shape_x': shape_pcd[conflict_idx, 0],
                                'shape_y': shape_pcd[conflict_idx, 1],
                                'shape_z': shape_pcd[conflict_idx, 2],
                                'info': f"{node['desc']['name']}.{planeTag} + shape_{planeTag} <= planeMax.{planeTag}",
                                'detail': {
                                    'position': node['desc']['position'],
                                    'planeMax': planeMax_xyz
                                }
                            })
                    else:
                        # for conflict_idx in conflict_idxs:
                        #     plane_constraints.append({
                        #         'type': 'planeMax_conflict', 'plane_tag': planeTag,
                        #         'node_idx': node_idx, 'ref_node_idx': planeMax_node_idx,
                        #         'shape_x': shape_pcd[conflict_idx, 0],
                        #         'shape_y': shape_pcd[conflict_idx, 1],
                        #         'shape_z': shape_pcd[conflict_idx, 2],
                        #         'info': f"({node['desc']['name']}.xyz + "
                        #                 f"rotMat_{node['desc']['xyzTag']}.dot(shape_xyz.T).T).{planeTag} "
                        #                 f"<= planeMax.{planeTag}>"
                        #     })
                        raise NotImplementedError

        return plane_constraints

    def get_shape_constraints(self, scale):
        dfs = []
        for node_idx in self.nodes_info.keys():
            node = self.nodes_info[node_idx]
            if node['vertex']['type'] == 'structor':
                shape_pcd = node['desc']['shape_pcd']
                if node['vertex']['dim'] == "xyz":
                    world_shape = shape_pcd + np.array(node['desc']['position'])
                else:
                    rot_mat = get_rotationMat(node['desc']['xyz_tag'], node['desc']['radian'])
                    world_shape = (rot_mat.dot(shape_pcd.T)).T

                pcd_array = np.concatenate([world_shape, shape_pcd], axis=1)
                sub_df = pd.DataFrame(pcd_array, columns=['x', 'y', 'z', 'shape_x', 'shape_y', 'shape_z'])
                sub_df['search_radius'] = node['desc']['search_radius']
                sub_df['tag'] = node['desc']['name']
                sub_df['node_idx'] = node_idx
                dfs.append(sub_df)

            elif node['vertex']['type'] == 'cell':
                pose = node['desc']['position']
                group_idx = node['groupIdx']
                sub_df = pd.DataFrame({
                    'x': pose[0], 'y': pose[1], 'z': pose[2], 'shape_x': 0, 'shape_y': 0, 'shape_z': 0,
                    'node_idx': node_idx, 'search_radius': node['desc']['radius'], 'tag': f"group_{group_idx}"
                }, index=[0])
                dfs.append(sub_df)
        dfs = pd.concat(dfs, axis=0, ignore_index=True)

        shape_constraints = []
        for tag in dfs['tag'].unique()[:-1]:
            cur_df: pd.DataFrame = dfs[dfs['tag'] == tag]
            other_df: pd.DataFrame = dfs[dfs['tag'] != tag]

            max_search_radius = (cur_df['search_radius'].max() + other_df['search_radius'].max()) + scale
            cur_xyzs = cur_df[['x', 'y', 'z']].values
            other_xyzs = other_df[['x', 'y', 'z']].values

            other_tree = KDTree(other_xyzs)
            idxs_list, dists_list = other_tree.query_radius(
                cur_xyzs, max_search_radius, return_distance=True, sort_results=True
            )
            for i, (idxs, dists) in enumerate(zip(idxs_list, dists_list)):
                if idxs.shape[0] == 0:
                    continue

                for idx, dist in zip(idxs, dists):
                    real_thre = cur_df.iloc[i]['search_radius'] + other_df.iloc[idx]['search_radius']
                    scale_thre = real_thre + scale
                    if dist >= scale_thre:
                        continue

                    res = []
                    for sub_series in [cur_df.iloc[i], other_df.iloc[idx]]:
                        node = self.nodes_info[sub_series['node_idx']]
                        node_name = node['desc']['name']

                        if node['vertex']['type'] == 'cell':
                            res.append({
                                'xyz': f"{node_name}.xyz",
                                'info': {"node_idx": sub_series['node_idx'], "shape_xyz": [0., 0., 0.]}
                            })

                        else:
                            shape_x, shape_y, shape_z = sub_series[['shape_x', 'shape_y', 'shape_z']].values
                            if node['vertex']['dim'] == 'xyz':
                                res.append({
                                    'xyz': f"({node_name}.xyz + shape_xyz)",
                                    'info': {
                                        "node_idx": sub_series['node_idx'],
                                        "shape_xyz": [shape_x, shape_y, shape_z]
                                    }
                                })
                            else:
                                res.append({
                                    'xyz': f"{node_name}.xyz + rotMat_{node['desc']['xyz_tag']}.dot(shape_xyz.T).T",
                                    'info': {
                                        "node_idx": sub_series['node_idx'],
                                        "shape_xyz": [shape_x, shape_y, shape_z]
                                    }
                                })

                    cur_res, other_res = res
                    shape_constraints.append({
                        'type': 'shape_conflict',
                        'src_infos': cur_res['info'],
                        'ref_infos': other_res['info'],
                        'threshold': real_thre,
                        'info': f"({cur_res['xyz']} - {other_res['xyz']}).norm >= threshold"
                    })

            dfs = other_df

        return shape_constraints

    def get_pose_constraints(self):
        pose_constrains = []

        for node_idx in self.nodes_info.keys():
            node = self.nodes_info[node_idx]
            node_name = node['desc']['name']

            if node['vertex']['type'] in ['cell', 'planeMax', 'planeMin']:
                continue

            for edge_info in node['poseEdges']:
                ref_node = self.nodes_info[self.name_to_nodeIdx[edge_info['ref_objName']]]

                constrain_info = {
                    'node_idx': node_idx,
                    'ref_node_idx': self.name_to_nodeIdx[edge_info['ref_objName']],
                }

                if edge_info['type'] == 'value_shift':
                    if edge_info['xyzTag'] == 'x':
                        value_shift = node['desc']['position'][0] - ref_node['desc']['position'][0]
                        info = f"{node_name}.x == {edge_info['ref_objName']}.x + value_shift"
                    elif edge_info['xyzTag'] == 'y':
                        value_shift = node['desc']['position'][1] - ref_node['desc']['position'][1]
                        info = f"{node_name}.y == {edge_info['ref_objName']}.y + value_shift"
                    else:
                        value_shift = node['desc']['position'][2] - ref_node['desc']['position'][2]
                        info = f"{node_name}.z == {edge_info['ref_objName']}.z + value_shift"
                    constrain_info.update({
                        'type': 'value_shift',
                        'xyzTag': edge_info['xyzTag'],
                        'info': info,
                        'value_shift': value_shift
                    })

                elif edge_info['type'] == 'radius_fixed':
                    node_pose = node['desc']['position']
                    ref_pose = ref_node['desc']['position']

                    if edge_info['xyzTag'] == 'x':
                        fix_radius = np.sqrt(
                            np.power(node_pose[1] - ref_pose[1], 2.0) + np.power(node_pose[2] - ref_pose[2], 2.0)
                        )
                        info = f"({edge_info['ref_objName']}.xy - {node_name}.xy).norm == fix_radius"
                    elif edge_info['xyzTag'] == 'y':
                        fix_radius = np.sqrt(
                            np.power(node_pose[0] - ref_pose[0], 2.0) + np.power(node_pose[2] - ref_pose[2], 2.0)
                        )
                        info = f"({edge_info['ref_objName']}.xz - {node_name}.xz).norm == fix_radius"
                    else:
                        fix_radius = np.sqrt(
                            np.power(node_pose[0] - ref_pose[0], 2.0) + np.power(node_pose[1] - ref_pose[1], 2.0)
                        )
                        info = f"({edge_info['ref_objName']}.yz - {node_name}.yz).norm == fix_radius"

                    constrain_info.update({
                        'type': 'radius_fixed',
                        'xyzTag': edge_info['xyzTag'],
                        'info': info,
                        'fix_radius': fix_radius
                    })

                elif edge_info['type'] == 'pose_fixed':
                    pose_shift = np.array(node['desc']['position']) - np.array(ref_node['desc']['position'])

                    if ref_node['vertex']['dim'] == "xyz":
                        constrain_info.update({
                            'type': 'pose_fixed',
                            'info': f"{node_name}.xyz == {edge_info['ref_objName']}.xyz + pose_shift",
                            'pose_shift': pose_shift
                        })
                    else:
                        info = f"{node_name}.xyz == {edge_info['ref_objName']}.xyz + " \
                               f"rotMat_{edge_info['xyzTag']}.dot(pose_shift.T).T",

                        constrain_info.update({
                            'type': 'pose_fixed',
                            'info': info,
                            'pose_shift': pose_shift
                        })

                else:
                    raise NotImplementedError

                pose_constrains.append(constrain_info)

        return pose_constrains

    def get_elastic_cost(self, kSpring=1.0):
        tag_record = {}
        elastic_constraints = []

        for pathIdx in self.paths_info.keys():
            path_info = self.paths_info[pathIdx]

            path_node_idxs = path_info['graph_node_idxs']
            path_size = len(path_node_idxs)
            for i in range(path_size - 1):
                node_idx0 = path_node_idxs[i]
                node_idx1 = path_node_idxs[i + 1]

                tag = f"{min(node_idx0, node_idx1)}-{max(node_idx0, node_idx1)}"
                if tag in tag_record.keys():
                    continue
                tag_record[tag] = True

                elastic_constraints.append({
                    'type': 'elastic_cost',
                    'node_idx0': node_idx0,
                    'node_idx1': node_idx1,
                    'kSpring': kSpring,
                    'info': f"cost = (node_{node_idx0}.xyz - node_{node_idx1}.xyz)^2"
                })

        return elastic_constraints

    def get_kinematic_cost(self, edge_kSpring=3.0, vertex_kSpring=10.0):
        tag_record = {}
        kinematic_constraints = []

        for path_idx in self.paths_info:
            path_info = self.paths_info[path_idx]

            path_node_idxs = path_info['graph_node_idxs']
            path_size = len(path_node_idxs)
            for i in range(1, path_size - 1, 1):
                node_idx0 = path_node_idxs[i - 1]
                node_idx1 = path_node_idxs[i]
                node_idx2 = path_node_idxs[i + 1]

                if i == 1:
                    tag = f"{min(node_idx0, node_idx1)}-{max(node_idx0, node_idx1)}"
                    if tag in tag_record.keys():
                        continue
                    tag_record[tag] = True

                    kinematic_constraints.append({
                        'type': 'kinematic_vertex',
                        'node_idx0': node_idx0,
                        'node_idx1': node_idx1,
                        'kSpring': vertex_kSpring,
                        'vec': self.nodes_info[node_idx0]['desc']['direction'],
                        'info': f"cost = 1.0 - (node_{node_idx1}.pose - node_{node_idx0}.pose).norm() * vec.norm()"
                    })

                elif i == path_size - 2:
                    tag = f"{min(node_idx1, node_idx2)}-{max(node_idx1, node_idx2)}"
                    if tag in tag_record.keys():
                        continue
                    tag_record[tag] = True

                    kinematic_constraints.append({
                        'type': 'kinematic_vertex',
                        'node_idx0': node_idx0,
                        'node_idx1': node_idx1,
                        'kSpring': vertex_kSpring,
                        'vec': self.nodes_info[node_idx2]['desc']['direction'],
                        'info': f"cost = 1.0 - (node_{node_idx2}.pose - node_{node_idx1}.pose).norm() * vec.norm()"
                    })

                tags = np.sort([node_idx0, node_idx1, node_idx2])
                tag = f"{tags[0]}-{tags[1]}-{tags[2]}"
                if tag in tag_record.keys():
                    continue
                tag_record[tag] = True

                kinematic_constraints.append({
                    'type': 'kinematic_edge',
                    'node_idx0': node_idx0,
                    'node_idx1': node_idx1,
                    'node_idx2': node_idx2,
                    'kSpring': edge_kSpring,
                    'info': f"cost = 1.0 - (node_{node_idx1}.pose - node_{node_idx0}.pose).norm() * "
                            f"(node_{node_idx2}.pose - node_{node_idx1}.pose).norm()"
                })

        return kinematic_constraints

    def plot_path(self):
        vis = ConnectVisulizer()
        ramdom_colors = np.random.random(size=(len(self.paths_info), 3))
        for i, path_idx in enumerate(self.paths_info.keys()):
            path_info = self.paths_info[path_idx]
            xyzs = []
            for node_idx in path_info['graph_node_idxs']:
                xyzs.append(self.nodes_info[node_idx]['desc']['position'])
            vis.plot_connect(np.array(xyzs), color=ramdom_colors[i], opacity=0.5)
        vis.show()

    def plot_node(self):
        vis = ConnectVisulizer()

        random_colors = np.random.random(size=(len(self.nodes_info), 3))
        for i, node_idx in enumerate(self.nodes_info.keys()):
            node = self.nodes_info[node_idx]
            pose = np.array(node['desc']['position'])

            if node['vertex']['type'] in ['cell', 'connector']:
                vis.plot_cell(pose, node['desc']['radius'], color=random_colors[i])

            elif node['vertex']['type'] == 'structor':
                shape_pcd = node['desc']['shape_pcd']
                if node['vertex']['dim'] == 'xyz':
                    world_pcd = shape_pcd + pose
                else:
                    rot_mat = get_rotationMat(node['desc']['xyz_tag'], node['desc']['radian'])
                    world_pcd = pose + (rot_mat.dot(shape_pcd.T)).T

                radius = np.min([
                    shape_pcd[:, 0].max() - shape_pcd[:, 0].min(),
                    shape_pcd[:, 1].max() - shape_pcd[:, 1].min(),
                    shape_pcd[:, 2].max() - shape_pcd[:, 2].min(),
                ]) / 3.0
                vis.plot_structor(pose, radius=radius, shape_xyzs=world_pcd, color=random_colors[i])

        vis.show()

    def plot_constraints(self, constraints, constraint_type=None, with_path=False, with_structor=False):
        vis = ConnectVisulizer()

        if with_path:
            ramdom_colors = np.random.random(size=(len(self.paths_info), 3))
            for i, path_idx in enumerate(self.paths_info.keys()):
                path_info = self.paths_info[path_idx]
                xyzs = []
                for node_idx in path_info['graph_node_idxs']:
                    xyzs.append(self.nodes_info[node_idx]['desc']['position'])
                vis.plot_connect(np.array(xyzs), color=ramdom_colors[i], opacity=0.5)

        if with_structor:
            for i, node_idx in enumerate(self.nodes_info.keys()):
                node = self.nodes_info[node_idx]
                pose = np.array(node['desc']['position'])
                if node['vertex']['type'] == 'structor':
                    shape_pcd = node['desc']['shape_pcd']
                    if node['vertex']['dim'] == 'xyz':
                        world_pcd = shape_pcd + pose
                    else:
                        rot_mat = get_rotationMat(node['desc']['xyz_tag'], node['desc']['radian'])
                        world_pcd = pose + (rot_mat.dot(shape_pcd.T)).T
                    vis.plot_structor(
                        pose, 1.0, world_pcd, color=np.random.uniform(0.0, 1.0, (3,)), with_center=False
                    )

        ramdom_colors = np.random.random(size=(len(constraints), 3))
        for i, constraint in enumerate(constraints):
            if constraint_type is not None:
                if constraint['type'] not in constraint_type:
                    continue

            if constraint['type'] == 'elastic_cost':
                xyzs = np.array([
                    self.nodes_info[constraint['node_idx0']]['desc']['position'],
                    self.nodes_info[constraint['node_idx1']]['desc']['position'],
                ])
                vis.plot_connect(xyzs, color=ramdom_colors[i])

            elif constraint['type'] == 'kinematic_vertex':
                xyzs = np.array([
                    self.nodes_info[constraint['node_idx0']]['desc']['position'],
                    self.nodes_info[constraint['node_idx1']]['desc']['position'],
                ])
                vis.plot_connect(xyzs, color=ramdom_colors[i])

            elif constraint['type'] == 'kinematic_edge':
                xyzs = np.array([
                    self.nodes_info[constraint['node_idx0']]['desc']['position'],
                    self.nodes_info[constraint['node_idx1']]['desc']['position'],
                    self.nodes_info[constraint['node_idx2']]['desc']['position'],
                ])
                vis.plot_connect(xyzs, color=ramdom_colors[i])

            elif constraint['type'] in ['value_shift', 'radius_fixed', 'pose_fixed']:
                xyzs = np.array([
                    self.nodes_info[constraint['node_idx']]['desc']['position'],
                    self.nodes_info[constraint['ref_node_idx']]['desc']['position'],
                ])
                vis.plot_connect(xyzs, color=ramdom_colors[i])

            elif constraint['type'] in ['shape_conflict']:
                poses = []
                for info in [constraint['src_infos'], constraint['ref_infos']]:
                    node = self.nodes_info[info['node_idx']]
                    pose = np.array(node['desc']['position'])
                    if node['vertex']['type'] == 'cell':
                        poses.append(pose)
                    else:
                        if node['vertex']['dim'] == 'xyz':
                            poses.append(pose + np.array(info['shape_xyz']))
                        else:
                            shape_xyz = np.array(info['shape_xyz']).reshape((1, 3))
                            rot_mat = get_rotationMat(node['desc']['xyzTag'], node['desc']['radian'])
                            poses.append(pose + (rot_mat.dot(shape_xyz.T)).T)
                vis.plot_connect(xyzs=np.array(poses), color=ramdom_colors[i])

            elif constraint['type'] in ['planeMin_conflict', 'planeMax_conflict']:
                node = self.nodes_info[constraint['node_idx']]
                if node['vertex']['type'] == 'cell':
                    pose = np.array(node['desc']['position'])
                else:
                    pose = np.array(node['desc']['position'])
                    shape_xyz = np.array(constraint['shape_xyz'])
                    if node['vertex']['dim'] == 'xyz':
                        pose = pose + shape_xyz
                    else:
                        shape_xyz = shape_xyz.reshape((1, 3))
                        rot_mat = get_rotationMat(node['desc']['xyz_tag'], node['desc']['radian'])
                        pose = pose + (rot_mat.dot(shape_xyz.T)).T

                if constraint['type'] == 'planeMin_conflict':
                    if constraint['plane_tag'] == 'x':
                        xyzs = np.array([pose, np.array([0., pose[1], pose[2]])])
                    elif constraint['plane_tag'] == 'y':
                        xyzs = np.array([pose, np.array([pose[0], 0., pose[2]])])
                    else:
                        xyzs = np.array([pose, np.array([pose[0], pose[1], 0.])])
                else:
                    planeMax_node = self.nodes_info[constraint['ref_node_idx']]
                    planeMax_pose = planeMax_node['desc']['position']
                    if constraint['plane_tag'] == 'x':
                        xyzs = np.array([pose, np.array([planeMax_pose[0], pose[1], pose[2]])])
                    elif constraint['plane_tag'] == 'y':
                        xyzs = np.array([pose, np.array([pose[0], planeMax_pose[1], pose[2]])])
                    else:
                        xyzs = np.array([pose, np.array([pose[0], pose[1], planeMax_pose[2]])])
                vis.plot_connect(xyzs, color=(1.0, 0.0, 0.0))

        vis.show()

class Optimizer_Scipy_v1(object):

    def eq_value_shift(self, xs, src_col, ref_col, value_shift):
        src_value = self.xs_init[src_col] + xs[src_col]
        ref_value = self.xs_init[ref_col] + xs[ref_col]
        return src_value - ref_value - value_shift

    def eq_radius_fixed(self, xs, src_col0, src_col1, ref_col0, ref_col1, radius):
        src_value0 = self.xs_init[src_col0] + xs[src_col0]
        src_value1 = self.xs_init[src_col1] + xs[src_col1]
        ref_value0 = self.xs_init[ref_col0] + xs[ref_col0]
        ref_value1 = self.xs_init[ref_col1] + xs[ref_col1]
        return np.power(src_value0 - ref_value0, 2) + np.power(src_value1 - ref_value1, 2) - np.power(radius, 2)

    def ineq_shape_conflict(
            self, xs,
            x0_col, y0_col, z0_col, shape_x0, shape_y0, shape_z0,
            x1_col, y1_col, z1_col, shape_x1, shape_y1, shape_z1,
            threshold
    ):
        x0 = self.xs_init[x0_col] + xs[x0_col]
        y0 = self.xs_init[y0_col] + xs[y0_col]
        z0 = self.xs_init[z0_col] + xs[z0_col]
        x1 = self.xs_init[x1_col] + xs[x1_col]
        y1 = self.xs_init[y1_col] + xs[y1_col]
        z1 = self.xs_init[z1_col] + xs[z1_col]

        return np.power(x0 + shape_x0 - x1 - shape_x1, 2.) + \
            np.power(y0 + shape_y0 - y1 - shape_y1, 2.) + \
            np.power(z0 + shape_z0 - z1 - shape_z1, 2.) - np.power(threshold, 2.)

    def ineq_planeMin_conflict_cell(self, xs, col, radius):
        return (self.xs_init[col] + xs[col]) - radius

    def ineq_planeMax_conflict_cell(self, xs, node_col, plane_col, radius):
        plane_value = self.xs_init[plane_col] + xs[plane_col]
        node_value = self.xs_init[node_col] + xs[node_col]
        return plane_value - (node_value + radius)

    def ineq_planeMin_conflict_structor(self, xs, col, shape_value):
        return (self.xs_init[col] + xs[col]) + shape_value

    def ineq_planeMax_conflict_structor(self, xs, node_col, plane_col, shape_value):
        plane_value = self.xs_init[plane_col] + xs[plane_col]
        node_value = self.xs_init[node_col] + xs[node_col]
        return plane_value - (node_value + shape_value)

    def cost_elastic(self, xs, x0_col, y0_col, z0_col, x1_col, y1_col, z1_col):
        x0 = self.xs_init[x0_col] + xs[x0_col]
        y0 = self.xs_init[y0_col] + xs[y0_col]
        z0 = self.xs_init[z0_col] + xs[z0_col]
        x1 = self.xs_init[x1_col] + xs[x1_col]
        y1 = self.xs_init[y1_col] + xs[y1_col]
        z1 = self.xs_init[z1_col] + xs[z1_col]
        return np.power(x0-x1, 2.0) + np.power(y0-y1, 2.0) + np.power(z0-z1, 2.0)

    def cost_func(self, xs, cost_edges: list[dict]):
        cost = 0.0

        for edge in cost_edges:
            if edge['type'] == 'cost_elastic':
                cost += self.cost_elastic(
                    xs, x0_col=edge['x0_col'], y0_col=edge['y0_col'], z0_col=edge['z0_col'],
                    x1_col=edge['x1_col'], y1_col=edge['y1_col'], z1_col=edge['z1_col']
                )

        return cost

    def log_info(self, xs, info, cost_func):
        print(f"Iter:{info['iter']} Cost:{cost_func(xs)}")
        info['iter'] += 1

    def create_problem(self, nodes_info, constraint_edges, cost_edges):
        problem_nodes = {}
        var_num = 0
        self.xs_init = []
        for node_idx in nodes_info.keys():
            node = nodes_info[node_idx]
            if node['vertex']['dim'] == 'xyz':
                problem_nodes[node_idx] = {
                    'x': var_num, 'y': var_num + 1, 'z': var_num + 2
                }
                var_num += 3
                self.xs_init.extend(node['desc']['position'])

            elif node['vertex']['dim'] == 'xyzr':
                # problem_nodes[node_idx] = {
                #     'x': var_num, 'y': var_num + 1, 'z': var_num + 2, 'radian': var_num + 3
                # }
                # var_num += 4
                # self.xs_init.extend(node['desc']['position'])
                # self.xs_init.extend([node['desc']['radian']])
                raise NotImplementedError

        self.xs_init = np.array(self.xs_init)

        constraint_set = []
        for constraint in constraint_edges:
            if constraint['type'] == 'value_shift':
                xyz_tag = constraint['xyzTag']
                constraint_set.append({
                    'type': 'eq',
                    'fun': partial(
                        self.eq_value_shift,
                        src_col=problem_nodes[constraint['node_idx']][xyz_tag],
                        ref_col=problem_nodes[constraint['ref_node_idx']][xyz_tag],
                        value_shift=constraint['value_shift']),
                })

            elif constraint['type'] == 'radius_fixed':
                xyz_tag = constraint['xyzTag']
                if xyz_tag == 'x':
                    func = partial(
                            self.eq_radius_fixed,
                            src_col0=problem_nodes[constraint['node_idx']]['y'],
                            src_col1=problem_nodes[constraint['node_idx']]['z'],
                            ref_col0=problem_nodes[constraint['ref_node_idx']]['y'],
                            ref_col1=problem_nodes[constraint['ref_node_idx']]['z'],
                            radius=constraint['fix_radius'])
                elif xyz_tag == 'y':
                    func = partial(
                        self.eq_radius_fixed,
                        src_col0=problem_nodes[constraint['node_idx']]['x'],
                        src_col1=problem_nodes[constraint['node_idx']]['z'],
                        ref_col0=problem_nodes[constraint['ref_node_idx']]['x'],
                        ref_col1=problem_nodes[constraint['ref_node_idx']]['z'],
                        radius=constraint['fix_radius']
                    )
                else:
                    func = partial(
                        self.eq_radius_fixed,
                        src_col0=problem_nodes[constraint['node_idx']]['x'],
                        src_col1=problem_nodes[constraint['node_idx']]['y'],
                        ref_col0=problem_nodes[constraint['ref_node_idx']]['x'],
                        ref_col1=problem_nodes[constraint['ref_node_idx']]['y'],
                        radius=constraint['fix_radius']
                    )
                constraint_set.append({'type': 'eq', 'fun': func})

            elif constraint['type'] == 'shape_conflict':
                node_idx0 = constraint['src_infos']['node_idx']
                node_idx1 = constraint['ref_infos']['node_idx']

                if nodes_info[node_idx0]['vertex']['dim'] == 'xyzr':
                    raise NotImplementedError
                if nodes_info[node_idx1]['vertex']['dim'] == 'xyzr':
                    raise NotImplementedError

                shape_xyz0 = constraint['src_infos']['shape_xyz']
                shape_xyz1 = constraint['ref_infos']['shape_xyz']

                constraint_set.append({
                    'type': 'ineq',
                    'fun': partial(
                        self.ineq_shape_conflict,
                        x0_col=problem_nodes[node_idx0]['x'], y0_col=problem_nodes[node_idx0]['y'], z0_col=problem_nodes[node_idx0]['z'],
                        shape_x0=shape_xyz0[0], shape_y0=shape_xyz0[1], shape_z0=shape_xyz0[2],
                        x1_col=problem_nodes[node_idx1]['x'], y1_col=problem_nodes[node_idx1]['y'], z1_col=problem_nodes[node_idx1]['z'],
                        shape_x1=shape_xyz1[0], shape_y1=shape_xyz1[1], shape_z1=shape_xyz1[2],
                        threshold=constraint['threshold']
                    )
                })

            elif constraint['type'] == 'planeMin_conflict':
                node_idx = constraint['node_idx']
                node = nodes_info[node_idx]
                if node['vertex']['type'] == 'cell':
                    constraint_set.append({
                        'type': 'ineq',
                        'fun': partial(
                            self.ineq_planeMin_conflict_cell,
                            col=problem_nodes[node_idx][constraint['plane_tag']], radius=node['desc']['radius']
                        )
                    })
                else:
                    if node['vertex']['dim'] == 'xyzr':
                        raise NotImplementedError

                    constraint_set.append({
                        'type': 'ineq',
                        'fun': partial(
                            self.ineq_planeMin_conflict_structor,
                            col=problem_nodes[node_idx][constraint['plane_tag']],
                            shape_value=constraint[f"shape_{constraint['plane_tag']}"]
                        )
                    })

            elif constraint['type'] == 'planeMax_conflict':
                node_idx = constraint['node_idx']
                node = nodes_info[node_idx]
                ref_node_idx = constraint['ref_node_idx']

                if node['vertex']['type'] == 'cell':
                    constraint_set.append({
                        'type': 'ineq',
                        'fun': partial(
                            self.ineq_planeMax_conflict_cell,
                            node_col=problem_nodes[node_idx][constraint['plane_tag']],
                            plane_col=problem_nodes[ref_node_idx][constraint['plane_tag']],
                            radius=node['desc']['radius']
                        )
                    })
                else:
                    if node['vertex']['dim'] == 'xyzr':
                        raise NotImplementedError

                    constraint_set.append({
                        'type': 'ineq',
                        'fun': partial(
                            self.ineq_planeMax_conflict_structor,
                            node_col=problem_nodes[node_idx][constraint['plane_tag']],
                            plane_col=problem_nodes[ref_node_idx][constraint['plane_tag']],
                            shape_value=constraint[f"shape_{constraint['plane_tag']}"]
                        )
                    })

        cost_lists = []
        for edge in cost_edges:
            if edge['type'] == 'elastic_cost':
                node_idx0, node_idx1 = edge['node_idx0'], edge['node_idx1']
                cost_lists.append({
                    'type': 'cost_elastic',
                    'x0_col': problem_nodes[node_idx0]['x'],
                    'y0_col': problem_nodes[node_idx0]['y'],
                    'z0_col': problem_nodes[node_idx0]['z'],
                    'x1_col': problem_nodes[node_idx1]['x'],
                    'y1_col': problem_nodes[node_idx1]['y'],
                    'z1_col': problem_nodes[node_idx1]['z'],
                })

        cost_func = partial(self.cost_func, cost_edges=cost_lists)
        xs = np.zeros(self.xs_init.shape)

        lbs = np.ones(xs.shape) * -0.5
        ubs = np.ones(xs.shape) * 0.5

        print(f"[Debug]: Init Cost: {cost_func(xs)} ConstrainsNum:{len(constraint_set)} CostNum:{len(cost_edges)}")
        res = optimize.minimize(
            cost_func, xs,
            constraints=constraint_set,
            bounds=Bounds(lbs, ubs),
            tol=0.01,
            # options={'maxiter': 3, 'disp': True},
            callback=partial(self.log_info, info={'iter': 0}, cost_func=cost_func)
        )
        # print(res)

        # ------ update nodes info
        opt_xs = res.x
        print(f"[Debug]: State:{res.success} StateCode:{res.status} OptCost:{res.fun} "
              f"optXs:{opt_xs.min()}->{opt_xs.max()} msg:{res.message}")

        for node_idx in problem_nodes.keys():
            problem_node = problem_nodes[node_idx]
            node = nodes_info[node_idx]

            xyz_idx = np.array([problem_node['x'], problem_node['y'], problem_node['z']])
            node['desc']['position'] = list(self.xs_init[xyz_idx] + opt_xs[xyz_idx])

            if 'radian' in problem_node.keys():
                radian_idx = problem_node['radian']
                node['desc']['radian'] = self.xs_init[radian_idx] + opt_xs[radian_idx]

        return nodes_info

class Optimizer_Scipy_v2(object):
    def penalize_eq_value_shift(self, xs, src_col, ref_col, value_shift):
        src_value = self.xs_init[src_col] + xs[src_col]
        ref_value = self.xs_init[ref_col] + xs[ref_col]
        return np.power(src_value - ref_value - value_shift, 2.0) * 50.0

    def penalize_eq_radius_fixed(self, xs, src_col0, src_col1, ref_col0, ref_col1, radius):
        src_value0 = self.xs_init[src_col0] + xs[src_col0]
        src_value1 = self.xs_init[src_col1] + xs[src_col1]
        ref_value0 = self.xs_init[ref_col0] + xs[ref_col0]
        ref_value1 = self.xs_init[ref_col1] + xs[ref_col1]
        return np.power(
            np.sqrt(np.power(src_value0 - ref_value0, 2) + np.power(src_value1 - ref_value1, 2)) - radius, 2.0
        ) * 50.0

    def penalize_ineq_shape_conflict(
            self, xs,
            x0_col, y0_col, z0_col, shape_x0, shape_y0, shape_z0,
            x1_col, y1_col, z1_col, shape_x1, shape_y1, shape_z1,
            threshold, z_col
    ):
        x0 = self.xs_init[x0_col] + xs[x0_col]
        y0 = self.xs_init[y0_col] + xs[y0_col]
        z0 = self.xs_init[z0_col] + xs[z0_col]
        x1 = self.xs_init[x1_col] + xs[x1_col]
        y1 = self.xs_init[y1_col] + xs[y1_col]
        z1 = self.xs_init[z1_col] + xs[z1_col]

        r = np.sqrt(
            np.power(x0 + shape_x0 - x1 - shape_x1, 2.) +
            np.power(y0 + shape_y0 - y1 - shape_y1, 2.) +
            np.power(z0 + shape_z0 - z1 - shape_z1, 2.)
        ) - threshold

        return np.power(r - np.power(xs[z_col], 2.0), 2.0) * 50.0

    def penalize_ineq_planeMin_conflict_cell(self, xs, col, radius, z_col):
        r = (self.xs_init[col] + xs[col]) - radius
        return np.power(r - np.power(xs[z_col], 2.0), 2.0) * 50.0

    def penalize_ineq_planeMax_conflict_cell(self, xs, node_col, plane_col, radius, z_col):
        plane_value = self.xs_init[plane_col] + xs[plane_col]
        node_value = self.xs_init[node_col] + xs[node_col]
        r = plane_value - (node_value + radius)
        return np.power(r - np.power(xs[z_col], 2.0), 2.0) * 50.0

    def penalize_ineq_planeMin_conflict_structor(self, xs, col, shape_value, z_col):
        r = (self.xs_init[col] + xs[col]) + shape_value
        return np.power(r - np.power(xs[z_col], 2.0), 2.0) * 50.0

    def penalize_ineq_planeMax_conflict_structor(self, xs, node_col, plane_col, shape_value, z_col):
        plane_value = self.xs_init[plane_col] + xs[plane_col]
        node_value = self.xs_init[node_col] + xs[node_col]
        r = plane_value - (node_value + shape_value)
        return np.power(r - np.power(xs[z_col], 2.0), 2.0) * 50.0

    def cost_elastic(self, xs, x0_col, y0_col, z0_col, x1_col, y1_col, z1_col):
        x0 = self.xs_init[x0_col] + xs[x0_col]
        y0 = self.xs_init[y0_col] + xs[y0_col]
        z0 = self.xs_init[z0_col] + xs[z0_col]
        x1 = self.xs_init[x1_col] + xs[x1_col]
        y1 = self.xs_init[y1_col] + xs[y1_col]
        z1 = self.xs_init[z1_col] + xs[z1_col]
        return np.power(x0-x1, 2.0) + np.power(y0-y1, 2.0) + np.power(z0-z1, 2.0)

    def cost_func(self, xs, cost_edges: list[dict], constraints_edges: list[dict]):
        cost = 0.0

        for edge in cost_edges:
            if edge['type'] == 'cost_elastic':
                cost += self.cost_elastic(
                    xs, x0_col=edge['x0_col'], y0_col=edge['y0_col'], z0_col=edge['z0_col'],
                    x1_col=edge['x1_col'], y1_col=edge['y1_col'], z1_col=edge['z1_col']
                )

        for constraint in constraints_edges:
            if constraint['type'] == 'eq_value_shift':
                cost += self.penalize_eq_value_shift(xs, **constraint['param'])
            elif constraint['type'] == 'eq_radius_fixed':
                cost += self.penalize_eq_radius_fixed(xs, **constraint['param'])
            elif constraint['type'] == 'ineq_shape_conflict':
                cost += self.penalize_ineq_shape_conflict(xs, **constraint['param'])
            elif constraint['type'] == 'ineq_planeMin_conflict_cell':
                cost += self.penalize_ineq_planeMin_conflict_cell(xs, **constraint['param'])
            elif constraint['type'] == 'ineq_planeMax_conflict_cell':
                cost += self.penalize_ineq_planeMax_conflict_cell(xs, **constraint['param'])
            elif constraint['type'] == 'ineq_planeMin_conflict_structor':
                cost += self.penalize_ineq_planeMin_conflict_structor(xs, **constraint['param'])
            elif constraint['type'] == 'ineq_planeMax_conflict_structor':
                cost += self.penalize_ineq_planeMax_conflict_structor(xs, **constraint['param'])

        return cost

    def log_info(self, xs, info, cost_func):
        print(f"Iter:{info['iter']} Cost:{cost_func(xs)}")
        info['iter'] += 1

    def create_problem(self, nodes_info, constraint_edges, cost_edges):
        problem_nodes = {}
        var_num = 0
        self.xs_init = []
        for node_idx in nodes_info.keys():
            node = nodes_info[node_idx]
            if node['vertex']['dim'] == 'xyz':
                problem_nodes[node_idx] = {
                    'x': var_num, 'y': var_num + 1, 'z': var_num + 2
                }
                var_num += 3
                self.xs_init.extend(node['desc']['position'])

            elif node['vertex']['dim'] == 'xyzr':
                # problem_nodes[node_idx] = {
                #     'x': var_num, 'y': var_num + 1, 'z': var_num + 2, 'radian': var_num + 3
                # }
                # var_num += 4
                # self.xs_init.extend(node['desc']['position'])
                # self.xs_init.extend([node['desc']['radian']])
                raise NotImplementedError

        num_of_xs = len(self.xs_init)

        constraint_set = []
        for constraint in constraint_edges:
            if constraint['type'] == 'value_shift':
                xyz_tag = constraint['xyzTag']
                constraint_set.append({
                    'type': 'eq_value_shift',
                    'param': {
                        'src_col': problem_nodes[constraint['node_idx']][xyz_tag],
                        'ref_col': problem_nodes[constraint['ref_node_idx']][xyz_tag],
                        'value_shift': constraint['value_shift'],
                    }
                })

            elif constraint['type'] == 'radius_fixed':
                xyz_tag = constraint['xyzTag']
                if xyz_tag == 'x':
                    constraint_set.append({
                        'type': 'eq_radius_fixed',
                        'param': {
                            'src_col0': problem_nodes[constraint['node_idx']]['y'],
                            'src_col1': problem_nodes[constraint['node_idx']]['z'],
                            'ref_col0': problem_nodes[constraint['ref_node_idx']]['y'],
                            'ref_col1': problem_nodes[constraint['ref_node_idx']]['z'],
                            'radius': constraint['fix_radius']
                        }
                    })

                elif xyz_tag == 'y':
                    constraint_set.append({
                        'type': 'eq_radius_fixed',
                        'param': {
                            'src_col0': problem_nodes[constraint['node_idx']]['x'],
                            'src_col1': problem_nodes[constraint['node_idx']]['z'],
                            'ref_col0': problem_nodes[constraint['ref_node_idx']]['x'],
                            'ref_col1': problem_nodes[constraint['ref_node_idx']]['z'],
                            'radius': constraint['fix_radius']
                        }
                    })

                else:
                    constraint_set.append({
                        'type': 'eq_radius_fixed',
                        'param': {
                            'src_col0': problem_nodes[constraint['node_idx']]['x'],
                            'src_col1': problem_nodes[constraint['node_idx']]['y'],
                            'ref_col0': problem_nodes[constraint['ref_node_idx']]['x'],
                            'ref_col1': problem_nodes[constraint['ref_node_idx']]['y'],
                            'radius': constraint['fix_radius']
                        }
                    })

            elif constraint['type'] == 'shape_conflict':
                node_idx0 = constraint['src_infos']['node_idx']
                node_idx1 = constraint['ref_infos']['node_idx']

                if nodes_info[node_idx0]['vertex']['dim'] == 'xyzr':
                    raise NotImplementedError
                if nodes_info[node_idx1]['vertex']['dim'] == 'xyzr':
                    raise NotImplementedError

                shape_xyz0 = constraint['src_infos']['shape_xyz']
                shape_xyz1 = constraint['ref_infos']['shape_xyz']

                self.xs_init.append(0.0)
                z_idx = len(self.xs_init) - 1
                constraint_set.append({
                    'type': 'ineq_shape_conflict',
                    'param': {
                        'x0_col': problem_nodes[node_idx0]['x'],
                        'y0_col': problem_nodes[node_idx0]['y'],
                        'z0_col': problem_nodes[node_idx0]['z'],
                        'shape_x0': shape_xyz0[0],
                        'shape_y0': shape_xyz0[1],
                        'shape_z0': shape_xyz0[2],
                        'x1_col': problem_nodes[node_idx1]['x'],
                        'y1_col': problem_nodes[node_idx1]['y'],
                        'z1_col': problem_nodes[node_idx1]['z'],
                        'shape_x1': shape_xyz1[0],
                        'shape_y1': shape_xyz1[1],
                        'shape_z1': shape_xyz1[2],
                        'threshold': constraint['threshold'],
                        'z_col': z_idx
                    }
                })

            elif constraint['type'] == 'planeMin_conflict':
                node_idx = constraint['node_idx']
                node = nodes_info[node_idx]

                if node['vertex']['type'] == 'cell':
                    self.xs_init.append(np.sqrt(node['desc']['radius']))
                    z_idx = len(self.xs_init) - 1
                    constraint_set.append({
                        'type': 'ineq_planeMin_conflict_cell',
                        'param': {
                            'col': problem_nodes[node_idx][constraint['plane_tag']],
                            'radius': node['desc']['radius'],
                            'z_col': z_idx
                        }
                    })

                else:
                    if node['vertex']['dim'] == 'xyzr':
                        raise NotImplementedError

                    # self.xs_init.append(np.sqrt(node['desc']['radius']))
                    # z_idx = len(self.xs_init) - 1
                    # constraint_set.append({
                    #     'type': 'ineq_planeMin_conflict_structor',
                    #     'param': {
                    #         'col': problem_nodes[node_idx][constraint['plane_tag']],
                    #         'shape_value': constraint[f"shape_{constraint['plane_tag']}"],
                    #         'z_col': z_idx
                    #     }
                    # })
                    raise NotImplementedError

            elif constraint['type'] == 'planeMax_conflict':
                node_idx = constraint['node_idx']
                node = nodes_info[node_idx]
                ref_node_idx = constraint['ref_node_idx']

                if node['vertex']['type'] == 'cell':
                    self.xs_init.append(np.sqrt(node['desc']['radius']))
                    z_idx = len(self.xs_init) - 1
                    constraint_set.append({
                        'type': 'ineq_planeMax_conflict_cell',
                        'param': {
                            'node_col': problem_nodes[node_idx][constraint['plane_tag']],
                            'plane_col': problem_nodes[ref_node_idx][constraint['plane_tag']],
                            'radius': node['desc']['radius'],
                            'z_col': z_idx
                        }
                    })

                else:
                    if node['vertex']['dim'] == 'xyzr':
                        raise NotImplementedError

                    # constraint_set.append({
                    #     'type': 'ineq_planeMax_conflict_structor',
                    #     'param': {
                    #         'node_col': problem_nodes[node_idx][constraint['plane_tag']],
                    #         'plane_col': problem_nodes[ref_node_idx][constraint['plane_tag']],
                    #         'shape_value': constraint[f"shape_{constraint['plane_tag']}"],
                    #         'z_col': z_idx
                    #     }
                    # })
                    raise NotImplementedError

        cost_lists = []
        for edge in cost_edges:
            if edge['type'] == 'elastic_cost':
                node_idx0, node_idx1 = edge['node_idx0'], edge['node_idx1']
                cost_lists.append({
                    'type': 'cost_elastic',
                    'x0_col': problem_nodes[node_idx0]['x'],
                    'y0_col': problem_nodes[node_idx0]['y'],
                    'z0_col': problem_nodes[node_idx0]['z'],
                    'x1_col': problem_nodes[node_idx1]['x'],
                    'y1_col': problem_nodes[node_idx1]['y'],
                    'z1_col': problem_nodes[node_idx1]['z'],
                })

        self.xs_init = np.array(self.xs_init)
        xs = np.zeros(self.xs_init.shape)

        lbs = np.ones(xs.shape)
        lbs[: num_of_xs] = -0.5
        lbs[num_of_xs:] = -np.inf

        ubs = np.ones(xs.shape)
        ubs[: num_of_xs] = 0.5
        ubs[num_of_xs:] = np.inf

        cost_func = partial(self.cost_func, cost_edges=cost_lists, constraints_edges=constraint_set)
        print(f"[Debug]: Init Cost: {cost_func(xs)} ConstrainsNum:{len(constraint_set)} CostNum:{len(cost_edges)}")
        res = optimize.minimize(
            cost_func, xs,
            bounds=Bounds(lbs, ubs), tol=0.01,
            # options={'maxiter': 3, 'disp': True},
            callback=partial(self.log_info, info={'iter': 0}, cost_func=cost_func)
        )
        # print(res)

        # ------ update nodes info
        opt_xs = res.x
        print(f"[Debug]: State:{res.success} StateCode:{res.status} OptCost:{res.fun} "
              f"optXs:{opt_xs.min()}->{opt_xs.max()} msg:{res.message}")

        for node_idx in problem_nodes.keys():
            problem_node = problem_nodes[node_idx]
            node = nodes_info[node_idx]

            xyz_idx = np.array([problem_node['x'], problem_node['y'], problem_node['z']])
            node['desc']['position'] = list(self.xs_init[xyz_idx] + opt_xs[xyz_idx])

            if 'radian' in problem_node.keys():
                radian_idx = problem_node['radian']
                node['desc']['radian'] = self.xs_init[radian_idx] + opt_xs[radian_idx]

        return nodes_info

def run_optimize_scipy():
    import json

    env_cfg_file = '/home/admin123456/Desktop/work/springer_debug/springer_grid_env_cfg.json'
    with open(env_cfg_file, 'r') as f:
        env_cfg = json.load(f)

    pipe_links_file = '/home/admin123456/Desktop/work/springer_debug/pipeLink_setting.json'
    with open(pipe_links_file, 'r') as f:
        pipe_links = json.load(f)

    res_paths = np.load('/home/admin123456/Desktop/work/springer_debug/result.npy', allow_pickle=True).item()

    debug_dir = '/home/admin123456/Desktop/work/springer_debug/debug'

    parser = ConstraintParser()
    parser.create_connective_graph(env_cfg, res_paths)

    parser.define_path(env_cfg, pipe_links)
    # parser.plot_path()

    parser.create_hyper_vertex()
    # parser.plot_path()

    # parser.plot_node()

    for run_time in range(10):

        # ------ Cost Edges
        cost_edges = []

        elastic_costs = parser.get_elastic_cost()
        # parser.plot_constraints(elastic_costs)
        cost_edges.extend(elastic_costs)

        # kinematic_costs = parser.get_kinematic_cost()
        # parser.plot_constraints(kinematic_costs, constraint_type=['kinematic_vertex'])
        # cost_edges.extend(kinematic_costs)

        # ------ Constraints Edges
        constraint_edges = []

        pose_constrains = parser.get_pose_constraints()
        # parser.plot_constraints(
        #     pose_constrains, constraint_type=['radius_fixed', 'value_shift'], with_path=True, with_structor=True
        # )
        constraint_edges.extend(pose_constrains)

        # shape_constraints = parser.get_shape_constraints(scale=0.5)
        # # parser.plot_constraints(
        # #     shape_constraints, constraint_type=['shape_conflict'], with_path=True, with_structor=True
        # # )
        # constraint_edges.extend(shape_constraints)

        plane_constraints = parser.get_plane_constraints(scale=0.5)
        # parser.plot_constraints(
        #     plane_constraints, constraint_type=['planeMin_conflict', 'planeMax_conflict'],
        #     with_path=True, with_structor=True
        # )
        constraint_edges.extend(plane_constraints)

        # # ------ record
        # record_dir = os.path.join(debug_dir, "%.3d" % run_time)
        # if os.path.exists(record_dir):
        #     shutil.rmtree(record_dir)
        # os.mkdir(record_dir)
        #
        # np.save(os.path.join(record_dir, 'env.npy'), {
        #     'nodes_info': parser.nodes_info,
        #     'name_to_nodeIdx': parser.name_to_nodeIdx,
        #     'paths_info': parser.paths_info
        # })
        # np.save(os.path.join(record_dir, 'problem.npy'), {
        #     'constraint_edges': constraint_edges,
        #     'cost_edges': cost_edges,
        # })

        # ------ optimize
        model = Optimizer_Scipy_v1()
        # model = Optimizer_Scipy_v2()

        opt_node_infos = model.create_problem(deepcopy(parser.nodes_info), constraint_edges, cost_edges)
        # np.save(os.path.join(record_dir, 'opt_env.npy'), {
        #     'nodes_info': opt_node_infos
        # })

        # ------ check each change
        # for node_idx in parser.nodes_info.keys():
        #     node_raw = parser.nodes_info[node_idx]
        #     node_opt = opt_node_infos[node_idx]
        #
        #     pose_raw = node_raw['desc']['position']
        #     pose_opt = node_opt['desc']['position']
        #     if node_raw['vertex']['dim'] == 'xyz':
        #         print(f"{node_raw['desc']['name']} "
        #               f"({pose_raw[0]:.3f}, {pose_raw[1]:.3f}, {pose_raw[2]:.3f}) -> "
        #               f"({pose_opt[0]:.3f}, {pose_opt[1]:.3f}, {pose_opt[2]:.3f})")
        #     else:
        #         raise NotImplementedError

        parser.nodes_info = opt_node_infos
        planeMax_node = parser.nodes_info[parser.name_to_nodeIdx["planeMax"]]
        planeMax_pose = planeMax_node['desc']['position']
        print(f"[Debug]: Running ...... {run_time} "
              f"Vol:{planeMax_pose[0]:.1f}-{planeMax_pose[1]:.3f}-{planeMax_pose[2]:.3f} \n")

        # parser.plot_node()

    # parser.plot_node()

if __name__ == '__main__':
    run_optimize_scipy()
