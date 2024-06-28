import numpy as np
from typing import List, Dict, Literal, Tuple
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from build import mapf_pipeline
from scripts_py.version_9.mapf_pkg import conflict_select_utils
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils


class CbsNode(object):
    def __init__(self, node_id):
        self.node = mapf_pipeline.CbsNode(node_id=node_id)

    @property
    def node_id(self):
        return self.node.node_id

    @property
    def num_expanded(self):
        return self.node.num_expanded

    @property
    def num_generated(self):
        return self.node.num_generated

    @property
    def search_time_cost(self):
        return self.node.search_time_cost

    def update_group_cell(
            self, group_idx: int, task_list: List[mapf_pipeline.TaskInfo],
            grid_env: mapf_pipeline.DiscreteGridEnv, obstacle_detector: mapf_pipeline.CollisionDetector,
            dynamic_obstacles: List
    ):
        self.node.update_group_cell(
            group_idx=group_idx, group_task_tree=task_list, group_grid=grid_env,
            obstacle_detector=obstacle_detector, group_dynamic_obstacles=dynamic_obstacles
        )

    def update_constrains_map(self, group_idx: int, dynamic_obstacles: List):
        self.node.update_constrains_map(group_idx=group_idx, group_dynamic_obstacles=dynamic_obstacles)

    def update_group_path(self, group_idx: int, max_iter: int):
        return self.node.update_group_path(group_idx=group_idx, max_iter=max_iter)

    def find_inner_conflict_point2point(self):
        return self.node.find_inner_conflict_point2point()

    def is_conflict_free(self):
        return self.node.is_conflict_free()

    def compute_h_val(self):
        return self.node.compute_h_val()

    def compute_g_val(self):
        return self.node.compute_g_val()

    def get_f_val(self):
        return self.node.get_f_val()

    def copy_from_node(self, rhs: mapf_pipeline.CbsNode):
        self.node.copy_from_node(rhs)

    def get_group_path(self, group_idx: int, name: str) -> mapf_pipeline.PathResult:
        return self.node.get_group_path(group_idx=group_idx, name=name)

    def get_conflict_cells(self) -> List[mapf_pipeline.ConflictCell]:
        return self.node.get_conflict_cells()

    def get_conflict_length(self, group_idx: int):
        return self.node.get_conflict_length(group_idx=group_idx)

    def get_constrain(self, group_idx: int):
        return self.node.get_constrain(group_idx=group_idx)

    def get_conflict_size(self):
        return self.node.get_conflict_size()

    def select_conflict_cell(
            self,
            pipe_cfgs: dict,
            grid_env: mapf_pipeline.DiscreteGridEnv,
            task_infos: Dict[int, List[mapf_pipeline.TaskInfo]]
    ) -> mapf_pipeline.ConflictCell:
        group_graph = {}
        for group_idx in task_infos.keys():
            graph = nx.Graph()
            for task_info in task_infos[group_idx]:
                path: mapf_pipeline.PathResult = self.get_group_path(group_idx, task_info.task_name)
                flags = path.get_path_flags()
                step_lengths = path.get_step_length()
                for i in range(1, len(flags), 1):
                    graph.add_edge(flags[i - 1], flags[i], weight=step_lengths[i - 1])
            group_graph[group_idx] = graph

        conflict = conflict_select_utils.HeadConflictSelector.process(
            pipe_cfgs=pipe_cfgs, group_graph=group_graph,
            conflict_list=self.get_conflict_cells(), grid_env=grid_env
        )
        return conflict


class CbsSolver(object):
    def __init__(self, grid_cfg: dict, pipe_cfg: dict):
        self.solver = mapf_pipeline.CbsSolver()
        self.grid_env = mapf_pipeline.DiscreteGridEnv(
            size_of_x=grid_cfg['size_of_x'] + 1,
            size_of_y=grid_cfg['size_of_y'] + 1,
            size_of_z=grid_cfg['size_of_z'] + 1,
            x_init=grid_cfg['grid_min'][0],
            y_init=grid_cfg['grid_min'][1],
            z_init=grid_cfg['grid_min'][2],
            x_grid_length=grid_cfg['x_grid_length'],
            y_grid_length=grid_cfg['y_grid_length'],
            z_grid_length=grid_cfg['z_grid_length']
        )

        self.obstacle_detector = mapf_pipeline.CollisionDetector()
        self.obstacle_df = pd.read_csv(grid_cfg['obstacle_file'], index_col=0)
        self.obstacle_detector.update_data(self.obstacle_df.values)
        self.obstacle_detector.create_tree()

        self.pipe_cfg = pipe_cfg
        for pipe_name in self.pipe_cfg:
            pipe_info = self.pipe_cfg[pipe_name]
            discrete_xyz = pipe_info['discrete_position']
            pipe_info.update({
                'loc_flag': self.grid_env.xyz2flag(x=discrete_xyz[0], y=discrete_xyz[1], z=discrete_xyz[2])
            })

        expand_method_2d = []
        expand_method_2d.extend(mapf_pipeline.candidate_1D)
        expand_method_2d.extend(mapf_pipeline.candidate_2D)

        expand_method_3d = []
        expand_method_3d.extend(mapf_pipeline.candidate_1D)
        expand_method_3d.extend(mapf_pipeline.candidate_2D)
        expand_method_3d.extend(mapf_pipeline.candidate_3D)

        self.grid_expand_method = {
            'expand_1D': mapf_pipeline.candidate_1D,
            'expand_2D': expand_method_2d,
            'expand_3D': expand_method_3d
        }

        self.node_mapper: Dict[int, CbsNode] = {}  # 由于CbsNode是由Python构造，必须由Python自身管理
        self.task_infos: Dict[int, List[mapf_pipeline.TaskInfo]] = {}
        self.latest_node_id = 0

    def init_block_root(self, block_info: List, last_leafs_info: dict = {}, aux_constrain_df: pd.DataFrame = None):
        self.latest_node_id = 0
        self.task_infos.clear()
        for key in self.node_mapper.keys():
            del self.node_mapper[key]
        self.node_mapper.clear()

        root = CbsNode(node_id=self.latest_node_id)
        self.latest_node_id += 1

        for group_info in block_info:
            group_idx = group_info['group_idx']
            last_leaf_info = last_leafs_info.get(group_idx, {})
            task_list = []

            for name0, name1, setting in group_info['sequence']:
                if name0 in last_leaf_info.get('member_tags', []):
                    begin_marks = [(flag, 0, 0, 0) for flag in last_leaf_info['flags_dict'].keys()]
                    begin_tag = 'main'
                else:
                    pipe0_info = self.pipe_cfg[name0]
                    vec_x0, vec_y0, vec_z0 = pipe0_info['direction']
                    if not pipe0_info['is_input']:
                        vec_x0, vec_y0, vec_z0 = -vec_x0, -vec_y0, -vec_z0
                    begin_marks = [(pipe0_info['loc_flag'], int(vec_x0), int(vec_y0), int(vec_z0))]
                    begin_tag = name0

                if name1 in last_leaf_info.get('member_tags', []):
                    final_marks = [(flag, 0, 0, 0) for flag in last_leaf_info['flags_dict'].keys()]
                    final_tag = 'main'
                else:
                    pipe1_info = self.pipe_cfg[name1]
                    vec_x1, vec_y1, vec_z1 = pipe1_info['direction']
                    if pipe1_info['is_input']:
                        vec_x1, vec_y1, vec_z1 = -vec_x1, -vec_y1, -vec_z1
                    final_marks = [(pipe1_info['loc_flag'], int(vec_x1), int(vec_y1), int(vec_z1))]
                    final_tag = name1

                task_list.append(mapf_pipeline.TaskInfo(
                    task_name=setting['task_name'],
                    begin_tag=begin_tag, final_tag=final_tag, begin_marks=begin_marks, final_marks=final_marks,
                    search_radius=setting['radius'], step_scale=setting['step_scale'],
                    shrink_distance=setting['shrink_distance'], shrink_scale=setting['shrink_scale'],
                    expand_grid_cell=self.grid_expand_method[setting['expand_grid_method']],
                    with_curvature_cost=setting['with_curvature_cost'],
                    curvature_cost_weight=setting['curvature_cost_weight'],
                    use_constraint_avoid_table=setting['use_constraint_avoid_table'],
                    with_theta_star=setting['with_theta_star'],
                ))

            self.task_infos[group_idx] = task_list

        for group_idx in list(self.task_infos.keys()):
            dynamic_constrains = []

            for pipe_name in self.pipe_cfg:
                pipe_info = self.pipe_cfg[pipe_name]
                if pipe_info['group_idx'] == group_idx:
                    continue
                x, y, z = pipe_info['discrete_position']
                dynamic_constrains.append((x, y, z, pipe_info['radius']))

            for sub_idx in last_leafs_info.keys():
                if sub_idx == group_idx:
                    continue
                for flag, radius in last_leafs_info[sub_idx]['flags_dict'].items():
                    other_x, other_y, other_z = self.grid_env.flag2xyz(flag)
                    dynamic_constrains.append((other_x, other_y, other_z, radius))

            if aux_constrain_df is not None:
                aux_constrains = aux_constrain_df[aux_constrain_df['group_idx'] != group_idx]
                for ii, x, y, z, r, jj in aux_constrains.itertuples():
                    dynamic_constrains.append((x, y, z, r))

            root.update_group_cell(
                group_idx=group_idx,
                task_list=self.task_infos[group_idx],
                grid_env=self.grid_env,
                obstacle_detector=self.obstacle_detector,
                dynamic_obstacles=dynamic_constrains
            )

        return root

    def first_check(self, root: CbsNode, group_idxs: List[int], max_iter: int):
        for group_idx in group_idxs:
            self.draw_node_3D(
                root, [], self.task_infos, obstacle_df=self.obstacle_df, pipe_cfg=self.pipe_cfg,
                highlight_group_idx=group_idx
            )

            is_success = root.update_group_path(group_idx, max_iter)
            print(f"groupIdx:{group_idx} success:{is_success}")

            if is_success:
                self.draw_node_3D(
                    root, [group_idx], self.task_infos, obstacle_df=self.obstacle_df, pipe_cfg=self.pipe_cfg,
                    highlight_group_idx=group_idx
                )

    def solve_block(
            self, root: CbsNode, group_idxs: List[int], max_iter: int, max_node_limit: int
    ) -> (bool, CbsNode):
        for group_idx in group_idxs:
            is_success = root.update_group_path(group_idx, max_iter)
            print(f"groupIdx:{group_idx} success:{is_success}")
            if not is_success:
                return False, None
        # self.draw_node_3D(root, group_idxs, self.task_infos, self.pipe_cfg)

        root.find_inner_conflict_point2point()
        root.compute_h_val()
        root.compute_h_val()
        self.push_node_wrap(root)

        is_success = False
        res_node = None
        for run_times in range(max_node_limit):
            print(f"[INFO]: running step:{run_times}")

            node_wrap = self.pop_node_wrap()

            if node_wrap.is_conflict_free():
                is_success = True
                res_node = node_wrap
                print("[INFO]: Solving Success.")
                break

            child_nodes = self.extend_node_wrap(node_wrap, max_iter=max_iter)
            for child_node_wrap in child_nodes:
                self.push_node_wrap(child_node_wrap)

            if self.solver.is_openList_empty():
                break

            print()

        return is_success, res_node

    def convert_node_to_leaf_info(self, node: CbsNode, group_idxs, last_leaf_info):
        for group_idx in group_idxs:
            last_leaf_info.setdefault(group_idx, {})
            member_tags = last_leaf_info[group_idx].get('member_tags', [])
            flags_dict = last_leaf_info[group_idx].get('flags_dict', {})

            for task_info in self.task_infos[group_idx]:
                path_res = node.get_group_path(group_idx, task_info.task_name)
                radius = task_info.search_radius

                for i, flag in enumerate(path_res.get_path_flags()):
                    if flag in flags_dict.keys():
                        flags_dict[flag] = max(flags_dict[flag], radius)
                    else:
                        flags_dict[flag] = radius

                member_tags.append(task_info.begin_tag)
                member_tags.append(task_info.final_tag)

            last_leaf_info[group_idx]['member_tags'] = list(set(member_tags))
            last_leaf_info[group_idx]['flags_dict'] = flags_dict

        return last_leaf_info

    def push_node_wrap(self, node_wrap: CbsNode):
        self.solver.push_node(node_wrap.node)
        self.node_mapper[node_wrap.node.node_id] = node_wrap

    def pop_node_wrap(self):
        node = self.solver.pop_node()
        node_wrap = self.node_mapper[node.node_id]
        del self.node_mapper[node.node_id]
        return node_wrap

    def extend_node_wrap(self, node_wrap: CbsNode, max_iter: int):
        conflict = node_wrap.select_conflict_cell(self.pipe_cfg, self.grid_env, self.task_infos)

        new_constrains = [
            (conflict.idx0, conflict.x0, conflict.y0, conflict.z0, conflict.radius0),
            (conflict.idx1, conflict.x1, conflict.y1, conflict.z1, conflict.radius1)
        ]

        child_nodes = []
        for info in new_constrains:
            is_success, new_node_wrap = self.create_node_wrap(
                node_wrap=node_wrap, group_idx=info[0],
                obs_x=info[1], obs_y=info[2], obs_z=info[3], obs_radius=info[4],
                max_iter=max_iter
            )
            if not is_success:
                continue
            child_nodes.append(new_node_wrap)

        return child_nodes

    def create_node_wrap(self, node_wrap: CbsNode, group_idx, obs_x, obs_y, obs_z, obs_radius, max_iter: int):
        new_node_wrap = CbsNode(node_id=self.latest_node_id)
        self.latest_node_id += 1
        new_node_wrap.copy_from_node(node_wrap.node)

        dynamic_constraints = new_node_wrap.get_constrain(group_idx)

        # # ------ debug
        # dynamic_constrains_np = np.array(dynamic_constraints)
        # if np.any(np.sum(np.abs(dynamic_constrains_np[:, :3] - np.array([obs_x, obs_y, obs_z])), axis=1) == 0):
        #     repeat_xyzr = np.array([obs_x, obs_y, obs_z, obs_radius])
        #     print(dynamic_constrains_np)
        #     print(repeat_xyzr)
        #     # for task_info in self.task_infos[group_idx]:
        #     #     path = new_node_wrap.get_group_path(group_idx, task_info.task_name)
        #     #     xyzr = np.array(path.get_path())
        #     #     dist = np.linalg.norm(xyzr[:, :3] - repeat_xyzr[:3], ord=2, axis=1)
        #     #     require_radius = xyzr[:, -1] + obs_radius
        #     #     tmp_idx = np.argmin(dist - require_radius)
        #     #     print(xyzr[tmp_idx], dist[tmp_idx])
        #     #
        #     # print('asdadf')
        #     # flag_tmp = new_node_wrap.update_group_path(group_idx, max_iter=10000)
        #     # print(flag_tmp)
        #     raise ValueError("Repeat Loop Constrain")
        # # ------

        dynamic_constraints.append((obs_x, obs_y, obs_z, obs_radius))

        # todo 当grid的密度较大时，管道干涉的障碍点密度就越大，低效搜索的次数就越多
        # print(f'debug: {group_idx}', dynamic_constraints)

        new_node_wrap.update_constrains_map(group_idx=group_idx, dynamic_obstacles=dynamic_constraints)

        is_success = new_node_wrap.update_group_path(group_idx, max_iter)
        if not is_success:
            return False, None

        new_node_wrap.find_inner_conflict_point2point()
        new_node_wrap.compute_h_val()
        new_node_wrap.compute_g_val()
        print(f"[INFO] A star solving num_expanded:{new_node_wrap.num_expanded}, "
              f"num_generated:{new_node_wrap.num_generated}, "
              f"search_time_cost:{new_node_wrap.search_time_cost} "
              f"conflict_num:{new_node_wrap.get_conflict_size()} "
              f"constrain_size:{len(dynamic_constraints)}"
              )

        # todo CBS对于目前状况并不紧致
        # self.draw_node_3D(
        #     new_node_wrap, list(self.task_infos.keys()), self.task_infos, self.pipe_cfg,
        #     obstacle_df=self.obstacle_df,
        #     # highlight_group_idx=group_idx
        # )

        return True, new_node_wrap

    @staticmethod
    def draw_node_3D(
            node: CbsNode, group_idxs: List[int],
            task_infos: Dict[int, List[mapf_pipeline.TaskInfo]],
            pipe_cfg: Dict,
            highlight_group_idx: int = None,
            obstacle_df: pd.DataFrame = None,
            vis: VisUtils = None
    ):
        if vis is None:
            vis = VisUtils()

        colors = {}
        for i, group_idx in enumerate(group_idxs):
            colors[group_idx] = np.random.uniform(0.0, 1.0, size=(3,))

        for i, group_idx in enumerate(group_idxs):
            task_list = task_infos[group_idx]
            for task_info in task_list:
                path_res = node.get_group_path(group_idx, task_info.task_name)
                xyzr = path_res.get_path()
                xyzr = np.array(xyzr)
                line_set = np.arange(0, xyzr.shape[0], 1)
                line_set = np.insert(line_set, 0, line_set.shape[0])
                tube_mesh = VisUtils.create_tube(xyzr[:, :3], task_info.search_radius, line_set)
                line_mesh = VisUtils.create_line(xyzr[:, :3], line_set)
                vis.plot(tube_mesh, colors[group_idx], opacity=0.6)
                vis.plot(line_mesh, (0., 0., 0.), opacity=1.0)

        if obstacle_df is not None:
            mesh = VisUtils.create_point_cloud(obstacle_df[['x', 'y', 'z']].values)
            vis.plot(mesh, color=(0.3, 0.3, 0.3), opacity=1.0)

        if highlight_group_idx is not None:
            for x, y, z, radius in node.get_constrain(highlight_group_idx):
                vis.plot(
                    VisUtils.create_sphere(np.array([x, y, z]), radius=max(radius, 0.02)),
                    color=(0.0, 1.0, 0.0), opacity=1.0
                )

            for name, info in pipe_cfg.items():
                if info['group_idx'] == highlight_group_idx:
                    vis.plot(
                        VisUtils.create_sphere(info['discrete_position'], radius=info['radius']),
                        color=(1.0, 0.0, 0.0), opacity=1.0
                    )

        vis.show()

    @staticmethod
    def draw_node_2D(
            node: CbsNode, group_idxs: List[int],
            method: Literal["xy", "xz", "yz"],
            task_infos: Dict[int, List[mapf_pipeline.TaskInfo]]
    ):
        fig, ax = plt.subplots()
        for i, group_idx in enumerate(group_idxs):
            task_list = task_infos[group_idx]
            for task_info in task_list:
                path_res = node.get_group_path(group_idx, task_info.name)
                xyzr = path_res.get_path()
                xyzr = np.array(xyzr)
                if method == 'xy':
                    ax.plot(xyzr[:, 0], xyzr[:, 1])
                elif method == 'xz':
                    ax.plot(xyzr[:, 0], xyzr[:, 2])
                else:
                    ax.plot(xyzr[:, 1], xyzr[:, 2])
        plt.show()

    @staticmethod
    def save_path(
            node: CbsNode, group_idxs: List[int], task_infos: Dict[int, List[mapf_pipeline.TaskInfo]]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        res = {}
        for group_idx in group_idxs:
            task_list = task_infos[group_idx]
            for task_info in task_list:
                path_res = node.get_group_path(group_idx, task_info.task_name)
                xyzr = np.array(path_res.get_path())
                res.setdefault(group_idx, {})[task_info.task_name] = xyzr
        return res
