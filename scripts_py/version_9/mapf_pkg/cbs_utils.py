import numpy as np
from typing import List, Dict, Literal
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from build import mapf_pipeline
from scripts_py.version_9.mapf_pkg import conflict_select_utils
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils


class CbsNode(object):
    def __init__(self, node_id):
        self.node = mapf_pipeline.CbsNode(node_id=node_id)

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

    def find_inner_conflict(self):
        return self.node.find_inner_conflict()

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
                path = self.get_group_path(group_idx, task_info.name)
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
        self.task_infos = {}
        self.latest_node_id = 0

    def init_root(self, block_info: List):
        self.latest_node_id = 0
        self.task_infos.clear()
        for key in self.node_mapper.keys():
            del self.node_mapper[key]
        self.node_mapper.clear()

        root = CbsNode(node_id=self.latest_node_id)
        self.latest_node_id += 1

        for group_info in block_info:
            group_idx = group_info['group_idx']
            task_list = []
            for name0, name1, setting in group_info['sequence']:
                pipe0_info = self.pipe_cfg[name0]
                pipe1_info = self.pipe_cfg[name1]

                vec_x0, vec_y0, vec_z0 = pipe0_info['direction']
                if not pipe0_info['is_input']:
                    vec_x0, vec_y0, vec_z0 = -vec_x0, -vec_y0, -vec_z0

                vec_x1, vec_y1, vec_z1 = pipe1_info['direction']
                if not pipe1_info['is_input']:
                    vec_x1, vec_y1, vec_z1 = -vec_x1, -vec_y1, -vec_z1

                task_list.append(mapf_pipeline.TaskInfo(
                    name=setting['name'],
                    begin_loc=self.pipe_cfg[name0]['loc_flag'],
                    final_loc=self.pipe_cfg[name1]['loc_flag'],
                    vec_x0=int(vec_x0), vec_y0=int(vec_y0), vec_z0=int(vec_z0),
                    vec_x1=int(vec_x1), vec_y1=int(vec_y1), vec_z1=int(vec_z1),
                    search_radius=setting['radius'], step_scale=setting['step_scale'],
                    shrink_distance=setting['shrink_distance'], shrink_scale=setting['shrink_scale'],
                    expand_grid_cell=self.grid_expand_method[setting['expand_grid_method']],
                    with_theta_star=setting['with_theta_star'],
                    with_curvature_cost=setting['with_curvature_cost'],
                    curvature_cost_weight=setting['curvature_cost_weight']
                ))
            self.task_infos[group_idx] = task_list

        for group_idx in list(self.task_infos.keys()):
            dynamic_constrains = []
            for pipe_name in self.pipe_cfg:
                pipe_info = self.pipe_cfg[pipe_name]
                if pipe_info['group_idx'] == group_idx:
                    continue
                x, y, z = pipe_info['discrete_position']
                dynamic_constrains.append((x, y, z, 0.0))

            root.update_group_cell(
                group_idx=group_idx,
                task_list=self.task_infos[group_idx],
                grid_env=self.grid_env,
                obstacle_detector=self.obstacle_detector,
                dynamic_obstacles=dynamic_constrains
            )

        return root

    def solve(self, root: CbsNode, group_idxs: List[int], max_iter):
        for group_idx in group_idxs:
            is_success = root.update_group_path(group_idx, max_iter)
            print(f"groupIdx:{group_idx} success:{is_success}")
            if not is_success:
                return {'status': False}

        # self.draw_node_3D(root, group_idxs)

        root.find_inner_conflict()
        root.compute_h_val()
        root.compute_h_val()
        self.push_node_wrap(root)

        is_success = False
        res_node = None
        for run_times in range(1000):
            node_wrap = self.pop_node_wrap()

            if node_wrap.is_conflict_free():
                is_success = True
                res_node = node_wrap
                print("[INFO]: Solving Success.")
                break

            child_nodes = self.extend_node_wrap(node_wrap)
            for child_node_wrap in child_nodes:
                self.push_node_wrap(child_node_wrap)

            if self.solver.is_openList_empty():
                break

        return is_success, res_node

    def push_node_wrap(self, node_wrap: CbsNode):
        self.solver.push_node(node_wrap.node)
        self.node_mapper[node_wrap.node.node_id] = node_wrap

    def pop_node_wrap(self):
        node = self.solver.pop_node()
        node_wrap = self.node_mapper[node.node_id]
        del self.node_mapper[node.node_id]
        return node_wrap

    def extend_node_wrap(self, node_wrap: CbsNode):
        conflict = node_wrap.select_conflict_cell(self.pipe_cfg, self.grid_env, self.task_infos)

        new_constrains = [
            (conflict.idx0, conflict.x0, conflict.y0, conflict.z0, conflict.radius0),
            (conflict.idx1, conflict.x1, conflict.y1, conflict.z1, conflict.radius1)
        ]

        child_nodes = []
        for info in new_constrains:
            is_success, new_node_wrap = self.create_node_wrap(
                node_wrap=node_wrap, group_idx=info[0],
                obs_x=info[1], obs_y=info[2], obs_z=info[3], obs_radius=info[4]
            )
            if not is_success:
                continue
            child_nodes.append(new_node_wrap)

        return child_nodes

    def create_node_wrap(self, node_wrap: CbsNode, group_idx, obs_x, obs_y, obs_z, obs_radius, max_iter=1000):
        new_node_wrap = CbsNode(node_id=self.latest_node_id)
        self.latest_node_id += 1
        new_node_wrap.copy_from_node(node_wrap.node)

        dynamic_constraints = new_node_wrap.get_constrain(group_idx)
        dynamic_constraints.append((obs_x, obs_y, obs_z, obs_radius))
        new_node_wrap.update_constrains_map(group_idx=group_idx, dynamic_obstacles=dynamic_constraints)

        is_success = new_node_wrap.update_group_path(group_idx, max_iter)
        if not is_success:
            return False, None

        new_node_wrap.find_inner_conflict()
        new_node_wrap.compute_h_val()
        new_node_wrap.compute_g_val()

        return True, new_node_wrap

    @staticmethod
    def draw_node_3D(
            node: CbsNode, group_idxs: List[int],
            task_infos: Dict[int, List[mapf_pipeline.TaskInfo]],
            obstacle_df: pd.DataFrame = None
    ):
        vis = VisUtils()
        pipe_colors = np.random.uniform(0.0, 1.0, size=(len(group_idxs), 3))
        for i, group_idx in enumerate(group_idxs):
            task_list = task_infos[group_idx]
            for task_info in task_list:
                path_res = node.get_group_path(group_idx, task_info.name)
                xyzr = path_res.get_path()
                xyzr = np.array(xyzr)
                line_set = np.arange(0, xyzr.shape[0], 1)
                line_set = np.insert(line_set, 0, line_set.shape[0])
                mesh = VisUtils.create_tube(xyzr[:, :3], task_info.search_radius, line_set)
                vis.plot(mesh, pipe_colors[i, :], opacity=1.0)

        if obstacle_df is not None:
            mesh = VisUtils.create_point_cloud(obstacle_df[['x', 'y', 'z']].values)
            vis.plot(mesh, color=(0.3, 0.3, 0.3), opacity=1.0)

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
