import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict
import math
import os, argparse, shutil
import json
from tqdm import tqdm

from build import mapf_pipeline
from scripts_py.visulizer import VisulizerVista


class FlexPathSmoother(object):
    def __init__(
            self,
            env_config,
            optimize_setting,
            with_elasticBand=True,
            with_kinematicEdge=False,
            with_obstacleEdge=False,
            with_pipeConflictEdge=False
    ):
        self.env_config = env_config
        self.optimize_setting = optimize_setting

        self.nodesInfo = {}
        self.pathInfos = {}
        self.network = nx.Graph()

        self.flexNodeInfos = {}
        self.groupFlexNodeIdxs = {}
        self.pose2NodeIdx = {}

        self.flexSmoother_runner = mapf_pipeline.FlexSmootherXYZ_Runner()
        self.flexSmoother_runner.initOptimizer()

        self.with_elasticBand = with_elasticBand
        self.with_kinematicEdge = with_kinematicEdge
        self.with_obstacleEdge = with_obstacleEdge
        self.with_pipeConflictEdge = with_pipeConflictEdge

    def init_environment(self):
        self.obstacle_df = pd.read_csv(self.env_config['obstacle_path'], index_col=0)

        self.groupConfig = {}
        for group_idx_str in self.env_config['pipe_cfgs'].keys():
            group_pipes = self.env_config['pipe_cfgs'][group_idx_str]
            for name in group_pipes:
                group_pipes[name]['scale_position'] = np.round(
                    group_pipes[name]['scale_position'], decimals=1
                )
            self.groupConfig[int(group_idx_str)] = group_pipes

        # ------ shift shell obstacle pointCloud
        shell_names = list(self.obstacle_df['tag'].unique())
        for group_idx in self.groupConfig.keys():
            pipe_infos = self.groupConfig[group_idx]
            for name in pipe_infos.keys():
                pipe_info = pipe_infos[name]

                shell_name = f'{name}_shell'
                if shell_name in shell_names:
                    shift_xyz = np.array(pipe_info['scale_position']) - np.array(pipe_info['position'])
                    idxs = self.obstacle_df.index[self.obstacle_df['tag'] == shell_name]
                    self.obstacle_df.loc[idxs, ['x', 'y', 'z']] += shift_xyz

        # ------ create obstacle tree
        for _, row in self.obstacle_df.iterrows():
            self.flexSmoother_runner.add_obstacle(row.x, row.y, row.z, row.radius)

    def createWholeNetwork(self, resGroupPaths: Dict):
        for groupIdx in resGroupPaths.keys():
            path_infos = resGroupPaths[groupIdx]

            for pathIdx, pathInfo in enumerate(path_infos):
                last_node_idx = None
                path_xyzrl = pathInfo['path_xyzrl']

                for i, (x, y, z, radius, length) in enumerate(path_xyzrl):

                    # ------ recover correct start and end point
                    if i == 0:
                        start_name = pathInfo['name0']
                        info = self.groupConfig[groupIdx][start_name]
                        grid_position = np.array(info['position'])
                        if np.all(grid_position == np.array([x, y, z])):
                            x, y, z = np.array(info['scale_position'])

                    elif i == path_xyzrl.shape[0] - 1:
                        end_name = pathInfo['name1']
                        info = self.groupConfig[groupIdx][end_name]
                        grid_position = np.array(info['position'])
                        if np.all(grid_position == np.array([x, y, z])):
                            x, y, z = np.array(info['scale_position'])

                    # ------ record points
                    if (x, y, z) not in self.pose2NodeIdx.keys():
                        current_node_idx = len(self.pose2NodeIdx.keys())
                        self.network.add_node(
                            current_node_idx,  # **{"x": x, "y": y, "z": z, "radius": radius}
                        )
                        self.nodesInfo[current_node_idx] = {
                            "pose": np.array([x, y, z]),
                            "radius": radius,
                            "groupIdx": groupIdx,
                            # "pathIdx": [pathIdx],
                        }
                        self.pose2NodeIdx[(x, y, z)] = current_node_idx

                    else:
                        current_node_idx = self.pose2NodeIdx[(x, y, z)]
                        self.nodesInfo[current_node_idx]["radius"] = max(
                            self.nodesInfo[current_node_idx]["radius"], radius
                        )
                        # self.nodesInfo[current_node_idx]['pathIdx'].append(pathIdx)

                    if last_node_idx is not None:
                        self.network.add_edge(last_node_idx, current_node_idx)

                    last_node_idx = current_node_idx

    def definePath(self, path_links):
        for groupIdx in path_links.keys():
            link_nfo = path_links[groupIdx]
            start_name = link_nfo['converge_pipe']

            start_pos = self.groupConfig[groupIdx][start_name]['scale_position']
            start_node_idx = self.pose2NodeIdx[(start_pos[0], start_pos[1], start_pos[2])]
            start_dire = self.groupConfig[groupIdx][start_name]['direction']
            start_radius = self.groupConfig[groupIdx][start_name]['radius']

            for end_name in link_nfo['branch_pipes']:
                branch_info = link_nfo['branch_pipes'][end_name]

                end_pos = self.groupConfig[groupIdx][end_name]['scale_position']
                end_node_idx = self.pose2NodeIdx[(end_pos[0], end_pos[1], end_pos[2])]
                end_dire = self.groupConfig[groupIdx][end_name]['direction']
                end_radius = self.groupConfig[groupIdx][end_name]['radius']

                nx_path_idxs = nx.shortest_path(self.network, start_node_idx, end_node_idx)
                path_size = len(nx_path_idxs)
                if path_size == 0:
                    print('[Waring]: inValid Path')
                    return

                path_idx = len(self.pathInfos)
                self.pathInfos[path_idx] = {
                    "groupIdx": groupIdx,
                    "nx_pathIdxs": nx_path_idxs,
                    "startName": start_name,
                    "endName": end_name,
                    "startDire": start_dire,
                    "endDire": end_dire,
                    "startFlexRatio": 0.0,
                    "endFlexRatio": branch_info['flexRatio'],
                    "startRadius": start_radius,
                    "endRadius": end_radius
                }

    def create_flexGraphRecord(self):
        self.flexNodeInfos.clear()
        self.groupFlexNodeIdxs.clear()

        shift_num = len(self.nodesInfo)
        new_num = 0

        for path_idx in self.pathInfos.keys():
            path_info = self.pathInfos[path_idx]
            group_idx = path_info['groupIdx']

            path_node_idxs = path_info["nx_pathIdxs"]
            path_size = len(path_node_idxs)
            # start_flex_num = math.ceil(path_size * pathInfo['startFlexRatio'])
            start_flex_num = 0

            end_flex_num = min(max(math.floor(path_size * path_info['endFlexRatio']), 6), path_size - 2)
            end_flex_num = path_size - end_flex_num

            flex_path_node_idxs = []
            for i, node_idx in enumerate(path_node_idxs):
                node_info: Dict = self.nodesInfo[node_idx]

                if i == 0 or i == path_size - 1:
                    fixed = True
                    flex_node_idx = node_idx
                    flex_node_info = node_info.copy()
                    flex_node_info.update({'fixed': fixed})

                elif i <= start_flex_num or i >= end_flex_num:
                    fixed = False
                    flex_node_idx = new_num + shift_num
                    new_num += 1

                    flex_node_info = node_info.copy()
                    flex_node_info.update({
                        'fixed': fixed,
                        'radius': path_info['endRadius']
                    })

                else:
                    fixed = False
                    flex_node_idx = node_idx
                    flex_node_info = node_info.copy()
                    flex_node_info.update({'fixed': fixed})

                if flex_node_idx not in self.flexNodeInfos.keys():
                    self.flexNodeInfos[flex_node_idx] = flex_node_info

                flex_path_node_idxs.append(flex_node_idx)

                if group_idx not in self.groupFlexNodeIdxs.keys():
                    self.groupFlexNodeIdxs[group_idx] = []
                self.groupFlexNodeIdxs[group_idx].append(flex_node_idx)

            path_info.update({'flex_pathIdxs': flex_path_node_idxs})

        for groupIdx in self.groupFlexNodeIdxs.keys():
            self.groupFlexNodeIdxs[groupIdx] = list(set(self.groupFlexNodeIdxs[groupIdx]))

        # for pathIdx in self.pathInfos.keys():
        #     pathInfo = self.pathInfos[pathIdx]
        #     print(pathIdx)
        #     print(pathInfo['nx_pathIdxs'])
        #     print(pathInfo['flex_pathIdxs'])
        #     print()

    def reconstruct_vertex_graph(self):
        for flexNodeIdx in self.flexNodeInfos.keys():
            flexNodeInfo = self.flexNodeInfos[flexNodeIdx]
            x, y, z = flexNodeInfo['pose']
            radius = flexNodeInfo['radius']
            fixed = flexNodeInfo['fixed']
            self.flexSmoother_runner.add_graphNode(flexNodeIdx, x, y, z, radius, fixed)
            success = self.flexSmoother_runner.add_vertex(flexNodeIdx)
            assert success

    def updateNodeMap_to_flexInfos(self):
        self.flexSmoother_runner.updateNodeMap_Vertex()
        for flexNodeIdx in self.flexNodeInfos.keys():
            nx_graphNode = self.flexSmoother_runner.graphNode_map[flexNodeIdx]
            self.flexNodeInfos[flexNodeIdx].update({
                'pose': np.array([nx_graphNode.x, nx_graphNode.y, nx_graphNode.z])
            })

    def add_elasticBand(self, kSpring=1.0, weight=1.0):
        record_dict = {}
        for pathIdx in self.pathInfos:
            path_info = self.pathInfos[pathIdx]

            flex_path_idxs = path_info['flex_pathIdxs']
            path_size = len(flex_path_idxs)

            for i in range(path_size - 1):
                node_idx0 = flex_path_idxs[i]
                node_idx1 = flex_path_idxs[i + 1]

                tag = f'{min(node_idx0, node_idx1)}-{max(node_idx0, node_idx1)}'
                if tag in record_dict.keys():
                    continue
                record_dict[tag] = True

                status = self.flexSmoother_runner.add_elasticBand(node_idx0, node_idx1, kSpring=kSpring, weight=weight)
                if not status:
                    return status

        return True

    def add_kinematicEdge(self, edge_kSpring=3.0, vertex_kSpring=10.0, weight=1.0):
        record_dict = {}

        for pathIdx in self.pathInfos:
            path_info = self.pathInfos[pathIdx]

            flex_path_idxs = path_info['flex_pathIdxs']
            path_size = len(flex_path_idxs)

            for i in range(1, path_size - 1, 1):
                node_idx0 = flex_path_idxs[i - 1]
                node_idx1 = flex_path_idxs[i]
                node_idx2 = flex_path_idxs[i + 1]

                tags = [node_idx0, node_idx1, node_idx2]
                tags = sorted(tags)
                tag = f'{tags[0]}-{tags[1]}-{tags[2]}'
                if tag in record_dict.keys():
                    continue
                record_dict[tag] = True

                if i == 1:
                    vec_i, vec_j, vec_k = path_info['startDire']
                    status = self.flexSmoother_runner.add_kinematicVertexEdge(
                        node_idx0, node_idx1, vec_i, vec_j, vec_k, kSpring=vertex_kSpring, weight=weight
                    )
                    if not status:
                        return status

                elif i == path_size - 2:
                    vec_i, vec_j, vec_k = path_info['endDire']
                    status = self.flexSmoother_runner.add_kinematicVertexEdge(
                        node_idx1, node_idx2, vec_i, vec_j, vec_k, kSpring=vertex_kSpring, weight=weight
                    )
                    if not status:
                        return status

                status = self.flexSmoother_runner.add_kinematicEdge(
                    node_idx0, node_idx1, node_idx2, kSpring=edge_kSpring, weight=weight
                )
                if not status:
                    return status

        return True

    def add_obstacleEdge(self, searchScale=1.5, repleScale=1.2, kSpring=100.0, weight=1.0):
        for flexNodeIdx in self.flexNodeInfos.keys():
            status = self.flexSmoother_runner.add_obstacleEdge(
                flexNodeIdx,
                searchScale=searchScale, repleScale=repleScale,
                kSpring=kSpring, weight=weight
            )
            if not status:
                return status
        return True

    def add_pipeConflictEdge(self, searchScale=1.5, repleScale=1.2, kSpring=100.0, weight=1.0):
        record_dict = {}
        for path_idx in self.pathInfos:
            path_info = self.pathInfos[path_idx]
            group_idx = path_info['groupIdx']

            for flex_nodeIdx in path_info['flex_pathIdxs']:
                tag = f'{flex_nodeIdx}'
                if tag in record_dict.keys():
                    continue
                record_dict[tag] = True

                status = self.flexSmoother_runner.add_pipeConflictEdge(
                    flex_nodeIdx, group_idx, searchScale=searchScale, repleScale=repleScale,
                    kSpring=kSpring, weight=weight
                )
                if not status:
                    return status
        return True

    def reconstruct_edges_graph(self):
        if self.with_elasticBand:
            success = self.add_elasticBand(
                kSpring=self.optimize_setting["elasticBand_kSpring"],
                weight=self.optimize_setting["elasticBand_weight"]
            )
            assert success

        if self.with_kinematicEdge:
            success = self.add_kinematicEdge(
                edge_kSpring=self.optimize_setting["kinematicEdge_kSpring"],
                vertex_kSpring=self.optimize_setting["kinematicVertex_kSpring"],
                weight=self.optimize_setting["kinematic_weight"]
            )
            assert success

        if self.with_obstacleEdge:
            success = self.add_obstacleEdge(
                searchScale=self.optimize_setting["obstacle_searchScale"],
                repleScale=self.optimize_setting["obstacle_repleScale"],
                kSpring=self.optimize_setting["obstacle_kSpring"],
                weight=self.optimize_setting["obstacle_weight"]
            )
            assert success

        if self.with_pipeConflictEdge:
            success = self.add_pipeConflictEdge(
                searchScale=self.optimize_setting["pipeConflict_searchScale"],
                repleScale=self.optimize_setting["pipeConflict_repleScale"],
                kSpring=self.optimize_setting["pipeConflict_kSpring"],
                weight=self.optimize_setting["pipeConflict_weight"]
            )
            assert success

    def optimize(self, outer_times=300, verbose=False):
        self.flexSmoother_runner.clear_graph()

        for outer_i in range(outer_times):
            print(f'[DEBUG]: Running Iteration {outer_i} ......')

            if self.flexSmoother_runner.is_g2o_graph_empty():
                # ------ Step 1 Create flex graph
                self.reconstruct_vertex_graph()

                # ------ Step 2 Update group loc tree
                for groupIdx in self.groupFlexNodeIdxs.keys():
                    self.flexSmoother_runner.updateGroupTrees(groupIdx, self.groupFlexNodeIdxs[groupIdx])

                # ------ Step 3 Create G2o optimize graph
                self.reconstruct_edges_graph()

                # self.flexSmoother_runner.info()

                # ------ Step 4 optimize
                self.flexSmoother_runner.optimizeGraph(
                    self.optimize_setting["inner_optimize_times"], verbose
                )

                # ------ Step 5 Update Vertex to Node
                self.updateNodeMap_to_flexInfos()

                # ------ Step 6 Clear Graph
                self.flexSmoother_runner.clear_graph()

                # ------ Step 7 Clear vertex memory in FlexSmootherXYZ_Runner
                self.flexSmoother_runner.clear_graphNodeMap()

            else:
                print('[DEBUG]: G2o Graph is not empty')

        self.flexSmoother_runner.updateNodeMap_Vertex()
        self.plotFlexGraphEnv()

    def output_result(self, save_dir):
        rescale = 1.0 / self.env_config["global_params"]["scale"]

        for path_idx in self.pathInfos.keys():
            path_info = self.pathInfos[path_idx]

            path_dir = os.path.join(save_dir, "path_%d" % path_idx)
            if os.path.exists(path_dir):
                shutil.rmtree(path_dir)
            os.mkdir(path_dir)

            path_xyzrs = []
            for flexNodeIdx in path_info['flex_pathIdxs']:
                flexNodeInfo = self.flexNodeInfos[flexNodeIdx]
                x, y, z = flexNodeInfo['pose']
                radius = flexNodeInfo['radius']
                path_xyzrs.append([x, y, z, radius])
            path_xyzrs = np.array(path_xyzrs)
            path_xyzrs = path_xyzrs * rescale

            path_csv = os.path.join(path_dir, "xyzs.csv")
            pd.DataFrame(path_xyzrs[:, :3], columns=['x', 'y', 'z']).to_csv(path_csv, header=True, index=False)

            radius_csv = os.path.join(path_dir, "radius.csv")
            pd.DataFrame(path_xyzrs[:, 3:4], columns=['radius']).to_csv(radius_csv, header=True, index=False)

            setting = {
                "groupIdx": path_info["groupIdx"],
                "startName": path_info["startName"],
                "endName": path_info["endName"],
                "startDire": path_info["startDire"],
                "endDire": path_info["endDire"],
                "startFlexRatio": path_info["startFlexRatio"],
                "endFlexRatio": path_info["endFlexRatio"],
                "startRadius": path_info["startRadius"] * rescale,
                "endRadius": path_info["endRadius"] * rescale,
                "path_file": path_csv
            }

            setting_file = os.path.join(path_dir, 'setting.json')
            with open(setting_file, 'w') as f:
                json.dump(setting, f, indent=4)

    def plotGroupPointCloudEnv(self):
        vis = VisulizerVista()

        groupIdxs = list(self.groupConfig.keys())
        random_colors = np.random.uniform(0.0, 1.0, size=(len(groupIdxs), 3))

        group_pointClouds = {}
        for nodeIdx in self.network.nodes(data=False):
            nodeInfo = self.nodesInfo[nodeIdx]
            groupIdx = nodeInfo['groupIdx']
            if groupIdx not in group_pointClouds.keys():
                group_pointClouds[groupIdx] = []
            group_pointClouds[groupIdx].append(nodeInfo['pose'])

        for groupIdx in groupIdxs:
            pcd_mesh = VisulizerVista.create_pointCloud(np.array(group_pointClouds[groupIdx]))
            vis.plot(pcd_mesh, color=random_colors[groupIdx, :])

        # obstacle_xyzs = self.obstacle_df[self.obstacle_df['tag'] != 'wall'][['x', 'y', 'z']].values
        # obstacle_mesh = VisulizerVista.create_pointCloud(obstacle_xyzs)
        # vis.plot(obstacle_mesh, (0.5, 0.5, 0.5))

        vis.show()

    def plotPathEnv(self, pathIdxs=None):
        if pathIdxs is None:
            pathIdxs = list(self.pathInfos.keys())

        random_colors = np.random.uniform(0.0, 1.0, size=(len(pathIdxs), 3))
        vis = VisulizerVista()
        for i, pathIdx in enumerate(pathIdxs):
            pathInfo = self.pathInfos[pathIdx]

            path_xyzs = []
            for nodeIdx in pathInfo['nx_pathIdxs']:
                nodeInfo = self.nodesInfo[nodeIdx]
                path_xyzs.append(nodeInfo['pose'])
            path_xyzs = np.array(path_xyzs)

            line_mesh = VisulizerVista.create_line(path_xyzs)
            vis.plot(line_mesh, color=random_colors[i, :], opacity=0.8)

        obstacle_xyzs = self.obstacle_df[self.obstacle_df['tag'] != 'wall'][['x', 'y', 'z']].values
        obstacle_mesh = VisulizerVista.create_pointCloud(obstacle_xyzs)
        vis.plot(obstacle_mesh, (0.5, 0.5, 0.5))

        vis.show()

    def plotFlexGraphEnv(self):
        pathIdxs = list(self.pathInfos.keys())
        random_colors = np.random.uniform(0.0, 1.0, size=(len(pathIdxs), 3))
        vis = VisulizerVista()
        for i, pathIdx in enumerate(pathIdxs):
            pathInfo = self.pathInfos[pathIdx]

            path_xyzrs = []
            for flexNodeIdx in pathInfo['flex_pathIdxs']:
                flexNodeInfo = self.flexNodeInfos[flexNodeIdx]
                x, y, z = flexNodeInfo['pose']
                radius = flexNodeInfo['radius']
                path_xyzrs.append([x, y, z, radius])
            path_xyzrs = np.array(path_xyzrs)

            tube_mesh = VisulizerVista.create_complex_tube(
                path_xyzrs[:, :3], capping=True, radius=None, scalars=path_xyzrs[:, 3]
            )
            line_mesh = VisulizerVista.create_line(path_xyzrs[:, :3])

            # vis.plot(tube_mesh, color=random_colors[i, :], opacity=0.65)
            vis.plot(line_mesh, color=random_colors[i, :], opacity=1.0)

        obstacle_xyzs = self.obstacle_df[self.obstacle_df['tag'] != 'wall'][['x', 'y', 'z']].values
        obstacle_mesh = VisulizerVista.create_pointCloud(obstacle_xyzs)
        vis.plot(obstacle_mesh, (0.5, 0.5, 0.5))

        vis.show()

    def plotNetwork_NX(self):
        nx.draw(self.network, with_labels=True)
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="the name of config json file",
                        default="/home/admin123456/Desktop/work/example8/grid_env_cfg.json"
                        )
    parser.add_argument("--path_result_file", type=str, help="path result", default="result.npy")
    parser.add_argument("--pipe_setting_file", type=str, help="pipe optimize setting", default="pipeLink_setting.json")
    parser.add_argument("--optimize_setting_file", type=str, help="", default="optimize_setting.json")
    parser.add_argument("--optimize_times", type=int, help="", default=100)
    parser.add_argument("--verbose", type=int, help="", default=0)
    args = parser.parse_args()
    return args


def custon_main():
    args = parse_args()

    with open(args.config_file) as f:
        env_cfg = json.load(f)

    result_file = os.path.join(env_cfg['project_dir'], args.path_result_file)
    result_pipes: Dict = np.load(result_file, allow_pickle=True).item()

    optimize_setting_file = os.path.join(env_cfg['project_dir'], args.optimize_setting_file)
    with open(optimize_setting_file, 'r') as f:
        optimize_setting = json.load(f)

    pipe_setting_file = os.path.join(env_cfg['project_dir'], args.pipe_setting_file)
    with open(pipe_setting_file) as f:
        path_links = json.load(f)

    for groupIdx in list(path_links.keys()):
        path_links[int(groupIdx)] = path_links[groupIdx]
        del path_links[groupIdx]

    smoother = FlexPathSmoother(
        env_cfg,
        optimize_setting=optimize_setting,
        with_elasticBand=True,
        with_kinematicEdge=True,
        with_obstacleEdge=True,
        with_pipeConflictEdge=True,
    )
    smoother.init_environment()

    # result_pipes = {
    #     0: result_pipes[0],
    #     1: result_pipes[1],
    #     2: result_pipes[2],
    #     3: result_pipes[3],
    #     4: result_pipes[4],
    #     5: result_pipes[5]
    # }
    # path_links = {
    #     0: path_links[0],
    #     1: path_links[1],
    #     2: path_links[2],
    #     3: path_links[3],
    #     4: path_links[4],
    #     5: path_links[5]
    # }

    smoother.createWholeNetwork(result_pipes)
    # smoother.plotGroupPointCloudEnv()

    smoother.definePath(path_links)

    smoother.create_flexGraphRecord()

    smoother.optimize(outer_times=args.optimize_times, verbose=args.verbose)

    optimize_dir = os.path.join(env_cfg['project_dir'], 'smoother_result')
    if not os.path.exists(optimize_dir):
        os.mkdir(optimize_dir)

    smoother.output_result(optimize_dir)


if __name__ == '__main__':
    custon_main()
