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
from scripts.visulizer import VisulizerVista

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
        self.obstacle_df = pd.read_csv(self.env_config['scaleObstaclePath'], index_col=0)

        self.groupConfig = {}
        for groupIdx in self.env_config['pipeConfig'].keys():
            groupPipes = self.env_config['pipeConfig'][groupIdx]
            for name in groupPipes:
                groupPipes[name]['scale_desc']['scale_position'] = np.round(
                    groupPipes[name]['scale_desc']['scale_position'], decimals=1
                )
            self.groupConfig[int(groupIdx)] = groupPipes

        ### ------ shift shell obstacle pointCloud
        for groupIdx in self.groupConfig.keys():
            pipeInfos = self.groupConfig[groupIdx]
            for name in pipeInfos.keys():
                pipeInfo = pipeInfos[name]
                scaleInfo = pipeInfo['scale_desc']

                shift_xyz = np.array(scaleInfo['scale_position']) - np.array(scaleInfo['grid_position'])
                shell_name = f'{name}_shell'
                idxs = self.obstacle_df.index[self.obstacle_df['tag'] == shell_name]
                self.obstacle_df.loc[idxs, ['x', 'y', 'z']] += shift_xyz

        ### ------ create obstacle tree
        for _, row in self.obstacle_df.iterrows():
            self.flexSmoother_runner.add_obstacle(row.x, row.y, row.z, row.radius)

    def createWholeNetwork(self, resGroupPaths: Dict):
        for groupIdx in resGroupPaths.keys():
            pathInfos = resGroupPaths[groupIdx]

            for pathIdx, pathInfo in enumerate(pathInfos):
                last_nodeIdx = None
                path_xyzrl = pathInfo['path_xyzrl']

                for i, (x, y, z, radius, length) in enumerate(path_xyzrl):

                    ### ------ recover correct start and end point
                    if i == 0:
                        startName = pathInfo['name0']
                        scaleInfo = self.groupConfig[groupIdx][startName]['scale_desc']
                        grid_position = np.array(scaleInfo['grid_position'])
                        if np.all(grid_position == np.array([x, y, z])):
                            x, y, z = np.array(scaleInfo['scale_position'])

                    elif i == path_xyzrl.shape[0] - 1:
                        endName = pathInfo['name1']
                        scaleInfo = self.groupConfig[groupIdx][endName]['scale_desc']
                        grid_position = np.array(scaleInfo['grid_position'])
                        if np.all(grid_position == np.array([x, y, z])):
                            x, y, z = np.array(scaleInfo['scale_position'])

                    ### ------ record points
                    if (x, y, z) not in self.pose2NodeIdx.keys():
                        current_nodeIdx = len(self.pose2NodeIdx.keys())
                        self.network.add_node(
                            current_nodeIdx,
                            # **{"x": x, "y": y, "z": z, "radius": radius}
                        )
                        self.nodesInfo[current_nodeIdx] = {
                            "pose": np.array([x, y, z]),
                            "radius": radius,
                            "groupIdx": groupIdx,
                            # "pathIdx": [pathIdx],
                        }
                        self.pose2NodeIdx[(x, y, z)] = current_nodeIdx

                    else:
                        current_nodeIdx = self.pose2NodeIdx[(x, y, z)]
                        self.nodesInfo[current_nodeIdx]["radius"] = max(
                            self.nodesInfo[current_nodeIdx]["radius"], radius
                        )
                        # self.nodesInfo[current_nodeIdx]['pathIdx'].append(pathIdx)

                    if last_nodeIdx is not None:
                        self.network.add_edge(last_nodeIdx, current_nodeIdx)

                    last_nodeIdx = current_nodeIdx

    def definePath(self, path_links):
        for groupIdx in path_links.keys():
            linkInfo = path_links[groupIdx]
            startName = linkInfo['converge_pipe']

            startPos = self.groupConfig[groupIdx][startName]['scale_desc']['scale_position']
            startNodeIdx = self.pose2NodeIdx[(startPos[0], startPos[1], startPos[2])]
            startDire = self.groupConfig[groupIdx][startName]['desc']['direction']
            startRadius = self.groupConfig[groupIdx][startName]['scale_desc']['scale_radius']

            for endName in linkInfo['branch_pipes']:
                branchInfo = linkInfo['branch_pipes'][endName]

                endPos = self.groupConfig[groupIdx][endName]['scale_desc']['scale_position']
                endNodeIdx = self.pose2NodeIdx[(endPos[0], endPos[1], endPos[2])]
                endDire = self.groupConfig[groupIdx][endName]['desc']['direction']
                endRadius = self.groupConfig[groupIdx][endName]['scale_desc']['scale_radius']

                nx_pathNodeIdxs = nx.shortest_path(self.network, startNodeIdx, endNodeIdx)
                path_size = len(nx_pathNodeIdxs)
                if path_size == 0:
                    print('[Waring]: inValid Path')
                    return

                pathIdx = len(self.pathInfos)
                self.pathInfos[pathIdx] = {
                    "groupIdx": groupIdx,
                    "nx_pathIdxs": nx_pathNodeIdxs,
                    "startName": startName,
                    "endName": endName,
                    "startDire": startDire,
                    "endDire": endDire,
                    "startFlexRatio": 0.0,
                    "endFlexRatio": branchInfo['flexRatio'],
                    "startRadius": startRadius,
                    "endRadius": endRadius
                }

    def create_flexGraphRecord(self):
        self.flexNodeInfos.clear()
        self.groupFlexNodeIdxs.clear()

        shift_num = len(self.nodesInfo)
        new_num = 0

        for pathIdx in self.pathInfos.keys():
            pathInfo = self.pathInfos[pathIdx]
            groupIdx = pathInfo['groupIdx']

            nx_pathNodeIdxs = pathInfo["nx_pathIdxs"]
            path_size = len(nx_pathNodeIdxs)
            # startFlexNum = math.ceil(path_size * pathInfo['startFlexRatio'])
            startFlexNum = 0

            endFlexNum = min(max(math.floor(path_size * pathInfo['endFlexRatio']), 6), path_size - 2)
            endFlexNum = path_size - endFlexNum

            flex_pathNodeIdxs = []
            for i, nodeIdx in enumerate(nx_pathNodeIdxs):
                nodeInfo: Dict = self.nodesInfo[nodeIdx]

                if i == 0 or i == path_size - 1:
                    fixed = True
                    flexNodeIdx = nodeIdx
                    flexNodeInfo = nodeInfo.copy()
                    flexNodeInfo.update({'fixed': fixed})

                elif i <= startFlexNum or i >= endFlexNum:
                    fixed = False
                    flexNodeIdx = new_num + shift_num
                    new_num += 1

                    flexNodeInfo = nodeInfo.copy()
                    flexNodeInfo.update({
                        'fixed': fixed,
                        'radius': pathInfo['endRadius']
                    })

                else:
                    fixed = False
                    flexNodeIdx = nodeIdx
                    flexNodeInfo = nodeInfo.copy()
                    flexNodeInfo.update({'fixed': fixed})

                if flexNodeIdx not in self.flexNodeInfos.keys():
                    self.flexNodeInfos[flexNodeIdx] = flexNodeInfo

                flex_pathNodeIdxs.append(flexNodeIdx)

                if groupIdx not in self.groupFlexNodeIdxs.keys():
                    self.groupFlexNodeIdxs[groupIdx] = []
                self.groupFlexNodeIdxs[groupIdx].append(flexNodeIdx)

            pathInfo.update({'flex_pathIdxs': flex_pathNodeIdxs})

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
            pathInfo = self.pathInfos[pathIdx]

            flex_pathIdxs = pathInfo['flex_pathIdxs']
            path_size = len(flex_pathIdxs)

            for i in range(path_size - 1):
                flex_nodeIdx0 = flex_pathIdxs[i]
                flex_nodeIdx1 = flex_pathIdxs[i + 1]

                tag = f'{min(flex_nodeIdx0, flex_nodeIdx1)}-{max(flex_nodeIdx0, flex_nodeIdx1)}'
                if tag in record_dict.keys():
                    continue
                record_dict[tag] = True

                status = self.flexSmoother_runner.add_elasticBand(flex_nodeIdx0, flex_nodeIdx1, kSpring=kSpring, weight=weight)
                if not status:
                    return status

        return True

    def add_kinematicEdge(self, edge_kSpring=3.0, vertex_kSpring=10.0, weight=1.0):
        record_dict = {}

        for pathIdx in self.pathInfos:
            pathInfo = self.pathInfos[pathIdx]

            flex_pathIdxs = pathInfo['flex_pathIdxs']
            path_size = len(flex_pathIdxs)

            for i in range(1, path_size - 1, 1):
                flex_nodeIdx0 = flex_pathIdxs[i - 1]
                flex_nodeIdx1 = flex_pathIdxs[i]
                flex_nodeIdx2 = flex_pathIdxs[i + 1]

                tags = [flex_nodeIdx0, flex_nodeIdx1, flex_nodeIdx2]
                tags = sorted(tags)
                tag = f'{tags[0]}-{tags[1]}-{tags[2]}'
                if tag in record_dict.keys():
                    continue
                record_dict[tag] = True

                if i == 1:
                    vec_i, vec_j, vec_k = pathInfo['startDire']
                    status = self.flexSmoother_runner.add_kinematicVertexEdge(
                        flex_nodeIdx0, flex_nodeIdx1, vec_i, vec_j, vec_k, kSpring=vertex_kSpring, weight=weight
                    )
                    if not status:
                        return status

                elif i == path_size - 2:
                    vec_i, vec_j, vec_k = pathInfo['endDire']
                    status = self.flexSmoother_runner.add_kinematicVertexEdge(
                        flex_nodeIdx1, flex_nodeIdx2, vec_i, vec_j, vec_k, kSpring=vertex_kSpring, weight=weight
                    )
                    if not status:
                        return status

                status = self.flexSmoother_runner.add_kinematicEdge(
                    flex_nodeIdx0, flex_nodeIdx1, flex_nodeIdx2, kSpring=edge_kSpring, weight=weight
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
        for pathIdx in self.pathInfos:
            pathInfo = self.pathInfos[pathIdx]
            groupIdx = pathInfo['groupIdx']

            for flex_nodeIdx in pathInfo['flex_pathIdxs']:
                tag = f'{flex_nodeIdx}'
                if tag in record_dict.keys():
                    continue
                record_dict[tag] = True

                status = self.flexSmoother_runner.add_pipeConflictEdge(
                    flex_nodeIdx, groupIdx, searchScale=searchScale, repleScale=repleScale,
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
            print(f'[DEBUG]: Runing Iteration {outer_i} ......')

            if self.flexSmoother_runner.is_g2o_graph_empty():
                ### ------ Step 1 Create flex graph
                self.reconstruct_vertex_graph()

                ### ------ Step 2 Update group loc tree
                for groupIdx in self.groupFlexNodeIdxs.keys():
                    self.flexSmoother_runner.updateGroupTrees(groupIdx, self.groupFlexNodeIdxs[groupIdx])

                ### ------ Step 3 Create G2o optimize graph
                self.reconstruct_edges_graph()

                # self.flexSmoother_runner.info()

                ### ------ Step 4 optimize
                self.flexSmoother_runner.optimizeGraph(
                    self.optimize_setting["inner_optimize_times"], verbose
                )

                ### ------ Step 5 Update Vertex to Node
                self.updateNodeMap_to_flexInfos()

                ### ------ Step 6 Clear Graph
                self.flexSmoother_runner.clear_graph()

                ### ------ Step 7 Clear vertex memory in FlexSmootherXYZ_Runner
                self.flexSmoother_runner.clear_graphNodeMap()

            else:
                print('[DEBUG]: G2o Graph is not empty')

        self.flexSmoother_runner.updateNodeMap_Vertex()
        self.plotFlexGraphEnv()

    def output_result(self, save_dir):
        rescale = 1.0 / self.env_config["global_params"]["grid_scale"]

        for pathIdx in self.pathInfos.keys():
            pathInfo = self.pathInfos[pathIdx]

            path_dir = os.path.join(save_dir, "path_%d" % pathIdx)
            if os.path.exists(path_dir):
                shutil.rmtree(path_dir)
            os.mkdir(path_dir)

            path_xyzrs = []
            for flexNodeIdx in pathInfo['flex_pathIdxs']:
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
                "groupIdx": pathInfo["groupIdx"],
                "startName": pathInfo["startName"],
                "endName": pathInfo["endName"],
                "startDire": pathInfo["startDire"],
                "endDire": pathInfo["endDire"],
                "startFlexRatio": pathInfo["startFlexRatio"],
                "endFlexRatio": pathInfo["endFlexRatio"],
                "startRadius": pathInfo["startRadius"] * rescale,
                "endRadius": pathInfo["endRadius"] * rescale,
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

            vis.plot(tube_mesh, color=random_colors[i, :], opacity=0.65)
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
    parser.add_argument("--proj_dire", type=str, help="project directory", default="")
    parser.add_argument("--config_file", type=str, help="the name of config json file", default="envGridConfig.json")
    parser.add_argument("--path_result_file", type=str, help="path result", default="result.npy")
    parser.add_argument("--pipe_setting_file", type=str, help="pipe optimize setting", default="pipeLink_setting.json")
    parser.add_argument("--optimize_setting_file", type=str, help="", default="optimize_setting.json")
    parser.add_argument("--optimize_times", type=int, help="", default=400)
    parser.add_argument("--verbose", type=int, help="", default=0)
    args = parser.parse_args()
    return args

def debug_run():
    with open('/home/admin123456/Desktop/work/example3/envGridConfig.json') as f:
        env_config = json.load(f)

    result_file = '/home/admin123456/Desktop/work/example3/result.npy'
    result_pipes: Dict = np.load(result_file, allow_pickle=True).item()

    pipeSetting_file = '/home/admin123456/Desktop/work/example3/pipeLink_setting.json'
    with open(pipeSetting_file) as f:
        path_links = json.load(f)
    for groupIdx in list(path_links.keys()):
        path_links[int(groupIdx)] = path_links[groupIdx]
        del path_links[groupIdx]

    optimize_setting_file = '/home/admin123456/Desktop/work/example3/optimize_setting.json'
    with open(optimize_setting_file, 'r') as f:
        optimize_setting = json.load(f)

    smoother = FlexPathSmoother(
        env_config,
        optimize_setting=optimize_setting,
        with_elasticBand=True,
        with_kinematicEdge=True,
        with_obstacleEdge=True,
        with_pipeConflictEdge=False,
    )
    smoother.init_environment()

    for info in result_pipes[2]:
        print(info['path_xyzrl'])

    smoother.createWholeNetwork(result_pipes)
    smoother.plotGroupPointCloudEnv()

    # smoother.definePath(path_links)
    # smoother.plotPathEnv()

    # smoother.create_flexGraphRecord()
    #
    # for pathIdx in smoother.pathInfos:
    #     print(smoother.pathInfos[pathIdx])

    # smoother.optimize(outer_times=400, verbose=False)

    # # smoother.output_result('/home/admin123456/Desktop/work/application/debug')

def custon_main():
    args = parse_args()

    if not os.path.exists(args.proj_dire):
        print(f"[WARNING]: Project isn't Exist {args.proj_dire}")
        return

    if not args.config_file.endswith('.json'):
        print(f"[WARNING]: Config Files isn't JSON Format {args.config_file}")
        return

    config_file = os.path.join(args.proj_dire, args.config_file)
    if not os.path.exists(config_file):
        print(f"[WARNING]: Config File isn't Exist {config_file}")
        return

    result_file = os.path.join(args.proj_dire, args.path_result_file)
    if not result_file.endswith('.npy'):
        print(f"[WARNING]: save Files must be npy Format {result_file}")
        return

    if not os.path.exists(result_file):
        print(f"[WARNING]: Result File isn't Exist {config_file}")
        return

    pipeSetting_file = os.path.join(args.proj_dire, args.pipe_setting_file)
    if not pipeSetting_file.endswith('.json'):
        print(f"[WARNING]: Pipe Setting Files isn't JSON Format {pipeSetting_file}")
        return

    if not os.path.exists(pipeSetting_file):
        print(f"[WARNING]: Pipe setting File isn't Exist {pipeSetting_file}")
        return

    ### ---------------------------------------------
    with open(config_file) as f:
        env_config = json.load(f)

    result_pipes: Dict = np.load(result_file, allow_pickle=True).item()

    optimize_setting_file = os.path.join(args.proj_dire, args.optimize_setting_file)
    with open(optimize_setting_file, 'r') as f:
        optimize_setting = json.load(f)

    smoother = FlexPathSmoother(
        env_config,
        optimize_setting=optimize_setting,
        with_elasticBand=True,
        with_kinematicEdge=True,
        with_obstacleEdge=True,
        with_pipeConflictEdge=True,
    )
    smoother.init_environment()
    smoother.createWholeNetwork(result_pipes)
    # smoother.plotGroupPointCloudEnv()

    with open(pipeSetting_file) as f:
        path_links = json.load(f)

    for groupIdx in list(path_links.keys()):
        path_links[int(groupIdx)] = path_links[groupIdx]
        del path_links[groupIdx]

    smoother.definePath(path_links)

    smoother.create_flexGraphRecord()

    smoother.optimize(outer_times=args.optimize_times, verbose=args.verbose)

    optimize_dir = os.path.join(args.proj_dire, 'smoother_result')
    if not os.path.exists(optimize_dir):
        os.mkdir(optimize_dir)

    smoother.output_result(optimize_dir)

if __name__ == '__main__':
    # debug_run()
    custon_main()