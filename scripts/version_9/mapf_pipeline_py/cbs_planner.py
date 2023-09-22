import numpy as np
import pandas as pd
from typing import Dict
from scripts.visulizer import VisulizerVista
from tqdm import tqdm
from copy import copy
import os, shutil, argparse
import json

from build import mapf_pipeline


class CBS_Planner(object):
    def __init__(self, env_cfg, debug_dir):
        self.env_cfg = env_cfg
        self.debug_dir = debug_dir
        self.save_groupNodeEnv_btn = False

    def init_environment(self):
        self.instance = mapf_pipeline.Instance(
            self.env_cfg['global_params']['envX'] + 1,
            self.env_cfg['global_params']['envY'] + 1,
            self.env_cfg['global_params']['envZ'] + 1
        )
        # self.instance.info()

        self.obstacle_df = pd.read_csv(self.env_cfg['obstacle_path'], index_col=0)
        self.stepLength = 1.0

        # ------ 由于CBSNode是由Python构造，必须由Python自己管理
        self.allNodes = {}
        self.node_id = 0

        # ------ group config
        self.group_cfg = {}
        for group_idx_str in self.env_cfg['pipe_cfgs'].keys():
            group_pipes = self.env_cfg['pipe_cfgs'][group_idx_str]
            self.group_cfg[int(group_idx_str)] = group_pipes

        self.cbs_planner = mapf_pipeline.CBSSolver()

    def load_task_tree(self, group_cfg, task_tree):
        record = {}
        for group_idx_str in task_tree:
            group_idx = int(group_idx_str)
            task_list = task_tree[group_idx_str]

            record[group_idx] = []
            pipes_info = group_cfg[group_idx]

            for task in task_list:
                name_i = task['name0']
                name_j = task['name1']

                info_i = pipes_info[name_i]
                info_j = pipes_info[name_j]

                record[group_idx].append({
                    'name0': name_i,
                    'loc0': self.instance.linearizeCoordinate(tuple(info_i['position'])),
                    'xyz0': info_i['position'],
                    'name1': task['name1'],
                    'loc1': self.instance.linearizeCoordinate(tuple(info_j['position'])),
                    'xyz1': info_j['position'],
                    'radius': task['radius'],
                })
        return record

    def solve(self, save_file: str):
        print("Init Setting")
        self.group_idxs = list(self.group_cfg.keys())

        # for group_idx in self.group_idxs:
        #     group_pipes_info = self.group_cfg[group_idx]
        #     for pipe_name in group_pipes_info.keys():
        #         pipe_info = group_pipes_info[pipe_name]
        #         print(
        #             f'GroupIdx:{group_idx} PipeName:{pipe_name} '
        #             f'xyz:{pipe_info["position"]} radius:{pipe_info["radius"]:.2f} '
        #         )
        # print()

        # --- 1. init root cbsNode
        root = mapf_pipeline.CBSNode(self.stepLength)

        for group_idx in self.group_idxs:
            # --- 1.1 init search engines
            self.cbs_planner.addSearchEngine(group_idx, with_AnyAngle=False, with_OrientCost=True)

            # --- 1.2 init root for group agent
            root.add_GroupAgent(group_idx)

        # --- 1.3 init taskTree
        self.group_task_record = self.load_task_tree(self.group_cfg, self.env_cfg['task_tree'])

        for group_idx in self.group_task_record.keys():
            for task in self.group_task_record[group_idx]:
                print(
                    f'GroupIdx:{group_idx}, {task["name0"]}({task["xyz0"]}) -> '
                    f'{task["name1"]}({task["xyz1"]}) radius:{task["radius"]}')
                root.addTask_to_GroupAgent(
                    group_idx, f"{task['name0']}->{task['name1']}",
                    task['loc0'], task['radius'], task['loc1'], task['radius']
                )
        print()

        # --- 1.4 init solver obstacle tree
        for _, row in self.obstacle_df.iterrows():
            self.cbs_planner.add_obstacle(row.x, row.y, row.z, row.radius)

        # --- 1.4 init agent constrains
        for groupIdx_i in self.group_idxs:
            constrains = []

            xyzr_i = []
            for name_i in self.group_cfg[groupIdx_i].keys():
                info = self.group_cfg[groupIdx_i][name_i]
                xyzr_i.append([
                    info['position'][0], info['position'][1], info['position'][2], info['radius'],
                ])
            xyzr_i = np.array(xyzr_i)

            # ------ 1.4.1 add pipe terminal sphere constraint
            # for groupIdx_j in self.group_idxs:
            #     if groupIdx_i == groupIdx_j:
            #         continue
            #
            #     for pipeName in self.group_cfg[groupIdx_j].keys():
            #         pipe_info = self.group_cfg[groupIdx_j][pipeName]
            #         pipe_position = np.array(pipe_info['position']).reshape((1, 3))
            #         pipe_radius = pipe_info['radius']
            #
            #         # ------ 1.4.1 add pipe terminal sphere constraint
            #         dist = np.linalg.norm(xyzr_i[:, :3] - pipe_position, axis=1, ord=2) - xyzr_i[:, 3] - 0.1
            #         dist = np.min(np.minimum(dist, pipe_radius))
            #
            #         constrains.append((
            #             pipe_info['position'][0], pipe_info['position'][1], pipe_info['position'][2], dist
            #         ))

            root.update_Constrains(groupIdx_i, constrains)

            # ------ 1.4.2 todo I forgot why I do that:
            for pipe_name in self.group_cfg[groupIdx_i].keys():
                pipe_info = self.group_cfg[groupIdx_i][pipe_name]
                grid_x, grid_y, grid_z = np.array(pipe_info['position'])
                radius = pipe_info['radius']

                xmin, ymin, zmin = grid_x - radius, grid_y - radius, grid_z - radius
                xmax, ymax, zmax = grid_x + radius, grid_y + radius, grid_z + radius
                root.add_rectangleExcludeArea(xmin, ymin, zmin, xmax, ymax, zmax)

        # root.info(with_constrainInfo=True)

        print("Starting Solving ...")
        # --- 1.5 compute all agent path / init first root
        for group_idx in self.group_idxs:
            success = self.cbs_planner.update_GroupAgentPath(group_idx, root, self.instance)
            print("groupIdx:%d success:%d" % (group_idx, success))

            if not success:
                print("[Debug]: Conflict Exist in Start Or End Pos")
                return {'status': False}

        # --- 1.4 find all the conflict and compute cost and heuristics
        root.depth = 0
        root.findFirstPipeConflict()
        root.compute_Heuristics()
        root.compute_Gval()
        # print(f'[Debug]: h_val:{root.h_val} g_val:{root.g_val}')
        # self.print_pathGraph(root, groupIdxs=self.groupIdxs, with_confict=True)

        # --- 1.5 push node into list
        self.pushNode(root)

        success = False
        for run_times in range(300):
            node = self.popNode()

            if self.cbs_planner.isGoal(node):
                success = True
                result_node = node
                break

            childNodes = self.extendNode(node)
            for child_node in childNodes:
                self.pushNode(child_node)

            if self.cbs_planner.is_openList_empty():
                print("[DEBUG]: Out of Resource !!!")
                break

            print(f'Running {run_times} ......')

        if success:
            self.print_pathGraph(result_node, groupIdxs=self.group_idxs, with_confict=True)

            res_dict = self.extractPath(result_node)
            np.save(save_file, res_dict)

            return {'status': True}

        else:
            print('[DEBUG]: Fail Find Any Solution')
            return {'status': False}

    def pushNode(self, node):
        node.node_id = self.node_id
        self.node_id += 1
        self.cbs_planner.pushNode(node)
        self.allNodes[node.node_id] = node

    def popNode(self):
        node = self.cbs_planner.popNode()
        del self.allNodes[node.node_id]
        return node

    def extendNode(self, node):
        select_conflict = node.firstConflict
        select_conflict.conflictExtend()
        # print(f'[DEBUG]: Constrain1 groupIdx:{select_conflict.groupIdx1} '
        #       f'x:{select_conflict.conflict1_x} y:{select_conflict.conflict1_y} z:{select_conflict.conflict1_z} '
        #       f'radius:{select_conflict.conflict1_radius}')
        # print(f'[DEBUG]: Constrain2 groupIdx:{select_conflict.groupIdx2} '
        #       f'x:{select_conflict.conflict2_x} y:{select_conflict.conflict2_y} z:{select_conflict.conflict2_z} '
        #       f'radius:{select_conflict.conflict2_radius}')

        new_constrains = [
            (select_conflict.groupIdx1, select_conflict.constrain1),
            (select_conflict.groupIdx2, select_conflict.constrain2)
        ]
        childNodes = []
        for groupIdx, constrain in new_constrains:
            success, new_node = self.createCBSNode(node, groupIdx, constrain)
            if not success:
                continue

            childNodes.append(new_node)

        return childNodes

    def extendNode_deepFirst(self, node):
        select_conflict = node.firstConflict
        select_conflict.conflictExtend()

        new_constrains = [
            (select_conflict.groupIdx1, select_conflict.constrain1),
            (select_conflict.groupIdx2, select_conflict.constrain2)
        ]

        for groupIdx, constrain in new_constrains:
            success, new_node = self.createCBSNode(node, groupIdx, constrain)
            if success:
                return [new_node]

        return []

    def createCBSNode(self, node, groupIdx, new_constrain):
        old_constrains = node.getConstrains(groupIdx)
        new_constraints = copy(old_constrains)
        new_constraints.append(new_constrain)

        childNode = mapf_pipeline.CBSNode(self.stepLength)
        childNode.copy(node)

        childNode.update_Constrains(groupIdx, new_constraints)
        success = self.cbs_planner.update_GroupAgentPath(groupIdx, childNode, self.instance)

        if not success:
            print(f"Fail to find groupIdx:{groupIdx} path")
            # for constraint in old_constrains:
            #     print(f'old: {constraint[0]:.1f}, {constraint[1]:.1f}, {constraint[2]:.1f}, radius:{constraint[3]:.1f}')
            # print(f'new: {new_constrain[0]:.1f}, {new_constrain[1]:.1f}, {new_constrain[2]:.1f}, radius:{constraint[3]:.1f}')
            # self.printDetailPath(
            #     node,
            #     compare_node=None,
            #     groupIdxs=self.groupIdxs,
            #     constraints=old_constrains,
            #     target_groupIdx=groupIdx,
            #     specify_constraints=[new_constrain]
            #     # specify_constraints=None
            # )

            return False, None

        childNode.depth = node.depth + 1
        childNode.findFirstPipeConflict()
        childNode.compute_Heuristics()
        childNode.compute_Gval()

        ### ------ debug vis
        # print(f'[DEBUG]: groupIdx:{groupIdx}')
        # self.printDetailPath(
        #     childNode,
        #     compare_node=node,
        #     groupIdxs=self.groupIdxs,
        #     # groupIdxs=[groupIdx],
        #     constraints=old_constrains,
        #     target_groupIdx=groupIdx,
        #     specify_constraints=[new_constrain]
        # )
        # self.save_groupNodeEnvironment(childNode, groupIdx, new_constraints)
        # self.print_pathGraph(childNode, groupIdxs=self.groupIdxs, with_confict=True)

        return True, childNode

    def print_pathGraph(self, node, groupIdxs, with_confict=True):
        vis = VisulizerVista()

        random_colors = np.random.uniform(0.0, 1.0, size=(50, 3))
        for groupIdx in groupIdxs:
            resPath_list = node.getGroupAgentResPath(groupIdx)
            for res_path in resPath_list:
                path_xyzrl = np.array(res_path)
                radius = path_xyzrl[0, 3]

                if path_xyzrl.shape[0] > 1:
                    tube_mesh = VisulizerVista.create_tube(path_xyzrl[:, :3], radius=radius)
                    line_mesh = VisulizerVista.create_line(path_xyzrl[:, :3])

                    # if target_groupIdx is not None and target_groupIdx == groupIdx:
                    #     vis.plot(tube_mesh, color=tuple(random_colors[groupIdx]), opacity=1.0)
                    # else:
                    vis.plot(tube_mesh, color=tuple(random_colors[groupIdx]), opacity=0.65)
                    vis.plot(line_mesh, color=(1, 0, 0))

        obstacle_xyzs = self.obstacle_df[self.obstacle_df['tag'] != 'wall'][['x', 'y', 'z']].values
        obstacle_mesh = VisulizerVista.create_pointCloud(obstacle_xyzs)
        vis.plot(obstacle_mesh, (0.5, 0.5, 0.5))

        if node.isConflict and with_confict:
            conflict = node.firstConflict

            print(f'[DEBUG]: Insert Conflict1 groupIdx:{conflict.groupIdx1} '
                  f'x:{conflict.conflict1_x} y:{conflict.conflict1_y} z:{conflict.conflict1_z} '
                  f'radius:{conflict.conflict1_radius}')
            conflict1_mesh = vis.create_sphere(
                np.array([conflict.conflict1_x, conflict.conflict1_y, conflict.conflict1_z]), conflict.conflict1_radius
            )
            vis.plot(conflict1_mesh, (0.0, 1.0, 0.0))

            print(f'[DEBUG]: Insert Conflict2 groupIdx:{conflict.groupIdx2} '
                  f'x:{conflict.conflict2_x} y:{conflict.conflict2_y} z:{conflict.conflict2_z} '
                  f'radius:{conflict.conflict2_radius}')
            conflict2_mesh = vis.create_sphere(
                np.array([conflict.conflict2_x, conflict.conflict2_y, conflict.conflict2_z]), conflict.conflict2_radius
            )
            vis.plot(conflict2_mesh, (0.0, 0.0, 1.0))

        vis.show()

    def save_groupNodeEnvironment_on_click(self):
        self.save_groupNodeEnv_btn = True

    def save_groupNodeEnvironment(self, node, groupIdx, constraints):
        if self.save_groupNodeEnv_btn:
            save_setting = {
                'num_of_x': self.instance.num_of_x,
                'num_of_y': self.instance.num_of_y,
                'num_of_z': self.instance.num_of_z,
                'taskTree': self.group_task_record[groupIdx],
                'constraints': constraints,
                'obstacle_file': self.env_cfg['scaleObstaclePath'],
                'computed_resPaths': node.getGroupAgentResPath(groupIdx)
            }
            np.save(os.path.join(self.debug_dir, f'record_{node.node_id}.npy'), save_setting)

            print('[DEBUG] Saving Environment')
            self.save_groupNodeEnv_btn = False

    def printDetailPath(
            self, node, groupIdxs, compare_node=None, target_groupIdx=None, constraints=None, specify_constraints=None
    ):
        vis = VisulizerVista()
        vis.add_KeyPressEvent(key='s', callback=self.save_groupNodeEnvironment_on_click)

        random_colors = np.random.uniform(0.0, 1.0, size=(50, 3))
        if target_groupIdx is not None:
            random_colors[target_groupIdx, :] = np.array([1., 0., 0.])

        for groupIdx in groupIdxs:
            resPath_list = node.getGroupAgentResPath(groupIdx)
            for res_path in resPath_list:
                path_xyzrl = np.array(res_path)
                radius = path_xyzrl[0, 3]

                if path_xyzrl.shape[0] > 1:
                    tube_mesh = VisulizerVista.create_tube(path_xyzrl[:, :3], radius=radius)
                    line_mesh = VisulizerVista.create_line(path_xyzrl[:, :3])
                    vis.plot(tube_mesh, color=tuple(random_colors[groupIdx]), opacity=0.65)
                    vis.plot(line_mesh, color=(1, 0, 0))

        obstacle_xyzs = self.obstacle_df[self.obstacle_df['tag'] != 'wall'][['x', 'y', 'z']].values
        obstacle_mesh = VisulizerVista.create_pointCloud(obstacle_xyzs)
        vis.plot(obstacle_mesh, (0.5, 0.5, 0.5))

        if constraints is not None:
            for x, y, z, radius in constraints:
                constraint_mesh = vis.create_sphere(np.array([x, y, z]), radius)
                vis.plot(constraint_mesh, (1.0, 0.0, 0.0), opacity=0.85)

        if specify_constraints is not None:
            for x, y, z, radius in specify_constraints:
                constraint_mesh = vis.create_sphere(np.array([x, y, z]), radius)
                vis.plot(constraint_mesh, (0.0, 0.0, 1.0), opacity=1.0)

        if compare_node is not None:
            resPath_list = compare_node.getGroupAgentResPath(target_groupIdx)
            for res_path in resPath_list:
                path_xyzrl = np.array(res_path)
                radius = path_xyzrl[0, 3]

                if path_xyzrl.shape[0] > 0:
                    tube_mesh = VisulizerVista.create_tube(path_xyzrl[:, :3], radius=radius)
                    # line_mesh = VisulizerVista.create_line(path_xyzrl[:, :3])
                    vis.plot(tube_mesh, color=(0., 1., 0.), opacity=0.65)
                    # vis.plot(line_mesh, color=(1, 0, 0))

        vis.show()

    def extractPath(self, node):
        res = {}
        for groupIdx in self.group_idxs:
            res[groupIdx] = []

            res_path_list = node.getGroupAgentResPath(groupIdx)
            for i, res_path in enumerate(res_path_list):
                path_xyzrl = np.array(res_path)
                if path_xyzrl.shape[0] > 1:
                    info = self.group_task_record[groupIdx][i]
                    info.update({'path_xyzrl': path_xyzrl})
                    res[groupIdx].append(info)

        return res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, help="the name of config json file",
        default="/home/admin123456/Desktop/work/example1/grid_env_cfg.json"
    )
    parser.add_argument("--save_file", type=str, help="project directory", default="result.npy")
    args = parser.parse_args()
    return args

def custon_main():
    args = parse_args()

    with open(args.config_file) as f:
        env_cfg = json.load(f)

    project_dir = env_cfg['project_dir']
    debug_dir = os.path.join(project_dir, 'debug')
    if not os.path.exists(debug_dir):
        os.mkdir(debug_dir)

    save_file = os.path.join(project_dir, args.save_file)

    cbs_planner = CBS_Planner(env_cfg, debug_dir=debug_dir)
    cbs_planner.init_environment()
    cbs_planner.solve(save_file=save_file)

if __name__ == '__main__':
    custon_main()
