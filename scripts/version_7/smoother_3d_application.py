import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import json
import os

from scripts.visulizer import VisulizerVista

from build import mapf_pipeline

grid_json_file = '/home/quan/Desktop/tempary/application_pipe/cond.json'
with open(grid_json_file, 'r') as f:
    env_config = json.load(f)

obs_df = pd.read_csv(env_config['obstacleSavePath'], index_col=0)

pathRes = np.load(os.path.join(env_config['projectDir'], 'res.npy'), allow_pickle=True).item()

group_keys = [0, 1, 2, 3, 4]
# group_keys = [0, 1, 2]

groupAgentConfig = {}
for pipeConfig in env_config['pipeConfig']:
    groupIdx = pipeConfig['groupIdx']
    if groupIdx not in group_keys:
        continue

    groupAgentConfig[groupIdx] = {}
    for pipe in pipeConfig['pipe']:
        groupAgentConfig[groupIdx].update({pipe['name']: pipe})
    groupAgentConfig[groupIdx].update({'grid_radius': pipeConfig['grid_radius']})

groupAgentLinks = {
    0: [
        ### elasticBand_weight=0.1 kinematic_weight=10.0,
        {"start": 'p', 'end': 'p1', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
        {"start": 'p', 'end': 'M1', 'startFlexRatio': 0.0, 'endFlexRatio': 0.5},
        {"start": 'p', 'end': 'p_valve', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}
    ],
    1: [
        {"start": 'B', 'end': 'M3', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
        {"start": 'B', 'end': 'B_valve', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}
    ],
    2: [
        {"start": 'T_valve', 'end': 'T', 'startFlexRatio': 0.4, 'endFlexRatio': 0.0},
        {"start": 'A2T', 'end': 'T', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}
    ],
    3: [
        {"start": 'A_valve', 'end': 'A2valve_01', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
        {"start": 'A_valve', 'end': 'A2valve_02', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}
    ],
    4: [
        ### TODO 这里出现问题的原因在于使用了 多进口多出口情景
        {"start": 'valve_01', 'end': 'A', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
        {"start": 'valve_02', 'end': 'A', 'startFlexRatio': 0.35, 'endFlexRatio': 0.0},
        {"start": 'valve_03','end': 'A', 'startFlexRatio': 0.35, 'endFlexRatio': 0.0},
        {"start": 'M2','end': 'A', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}

        # {"start": 'valve_03','end': 'M2', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}
    ]
}

smoother = mapf_pipeline.SmootherXYZG2O()
smoother.initOptimizer()

xmin=0.0
ymin=0.0
zmin=0.0
xmax=env_config['grid_x']
ymax=env_config['grid_y']
zmax=env_config['grid_z']
smoother.setBoundary(xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax)

for idx, row in obs_df.iterrows():
    smoother.insertStaticObs(row.x, row.y, row.z, row.radius, 0.0, 0.0)

for groupIdx in group_keys:
    path_xyzrls = pathRes[groupIdx]
    for path_xyzrl in path_xyzrls:
        path_xyzr = []
        for xyzrl in path_xyzrl:
            path_xyzr.append((xyzrl[0], xyzrl[1], xyzrl[2], xyzrl[3]))
        
        smoother.add_Path(groupIdx, path_xyzr)

        # path_xyzr_resample = []
        # for (x, y, z, r, l) in mapf_pipeline.sampleDetailPath(path_xyzr, 0.5):
        #     path_xyzr_resample.append((x, y, z, r))
        # smoother.add_Path(groupIdx, path_xyzr_resample)

res_config = {}

for groupIdx in group_keys:
    pipeConfig = groupAgentConfig[groupIdx]

    res_config[groupIdx] = {}

    for pathIdx, link in enumerate(groupAgentLinks[groupIdx]):
        start_info = pipeConfig[link['start']]
        end_info = pipeConfig[link['end']]

        success = smoother.add_OptimizePath(
            groupIdx=groupIdx, pathIdx=pathIdx,
            start_x=start_info['grid_position'][0], 
            start_y=start_info['grid_position'][1], 
            start_z=start_info['grid_position'][2],
            end_x=end_info['grid_position'][0], 
            end_y=end_info['grid_position'][1], 
            end_z=end_info['grid_position'][2],
            startDire=(start_info['alpha'], start_info['theta']),
            endDire=(end_info['alpha'], end_info['theta']),
            startFlexRatio=link['startFlexRatio'], 
            endFlexRatio=link['endFlexRatio']
        )
        print(link, success)

        res_config[groupIdx][pathIdx] = {
            'grid_radius': groupAgentConfig[groupIdx]['grid_radius'],
            'startDire': (start_info['alpha'], start_info['theta']),
            'endDire':(end_info['alpha'], end_info['theta'])
        }
        res_config[groupIdx][pathIdx].update(link)

### Show PathIdxs
# for groupIdx in group_keys:
#     groupPath = smoother.groupMap[groupIdx]
#     for pathIdx in groupPath.graphPathMap.keys():
#         print('PathIdxs:', groupPath.graphPathMap[pathIdx])

for pipeConfig in env_config['pipeConfig']:
    if pipeConfig['groupIdx'] in group_keys:
        smoother.setMaxRadius(pipeConfig['groupIdx'] ,pipeConfig['grid_radius'])
        # smoother.setFlexible_percentage(pipeConfig['groupIdx'], 0.0)

def run_smooth(
    smoother,
    elasticBand_weight=0.0,
    kinematic_weight=0.0,
    obstacle_weight=0.0,
    pipeConflict_weight=0.0,
    boundary_weight=0.0,
    run_times=10
):
    ### Step 2.1 Build Graph 
    success = smoother.build_graph(
        elasticBand_weight=elasticBand_weight,
        kinematic_weight=kinematic_weight,
        obstacle_weight=obstacle_weight,
        pipeConflict_weight=pipeConflict_weight,
        boundary_weight=boundary_weight
    )
    # print("build Graph:", success)
    # smoother.info()

    ### Step 2.2 Optimize
    smoother.optimizeGraph(run_times, False)

    ### Step 3.3 Update Vertex to Node
    smoother.update2groupVertex()
    smoother.updateGroupTrees()

    # ### Step 2.4 Clear Graph
    smoother.clear_graph()

    # smoother.loss_report(
    #     groupIdx=4, 
    #     pathIdx=0,
    #     elasticBand_weight=1.0,
    #     kinematic_weight=0.0,
    #     obstacle_weight=0.0,
    #     pipeConflict_weight=0.0,
    #     boundary_weight=0.0
    # )
    # print()

### ------------------------------------------
smoother.updateGroupTrees()
for outer_i in range(500):
    run_smooth(
        smoother=smoother,
        elasticBand_weight=1.0,
        kinematic_weight=10.0,
        obstacle_weight=30.0,
        pipeConflict_weight=30.0,
        boundary_weight=0.0,
        run_times=10
    )

    ### ------ Debug Vis path
    if outer_i % 500 == 0:
        '''
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim3d(-1, xmax+1)
        ax.set_ylim3d(-1, ymax+1)
        ax.set_zlim3d(-1, zmax+1)

        colors = np.random.uniform(0.0, 1.0, (5, 3))
        for groupIdx in group_keys:
            groupPath = smoother.groupMap[groupIdx]

            for pathIdx in groupPath.graphPathMap.keys():
                nodeIdxs_path = groupPath.graphPathMap[pathIdx]

                path_xyz = []
                for idx, nodeIdx in enumerate(nodeIdxs_path):
                    node = groupPath.graphNodeMap[nodeIdx]
                    path_xyz.append([node.x, node.y, node.z])
                    
                    if (node.fixed):
                        dz = math.sin(node.theta)
                        dl = math.cos(node.theta)
                        dx = dl * math.cos(node.alpha)
                        dy = dl * math.sin(node.alpha)
                        ax.quiver(
                            node.x, node.y, node.z, 
                            dx, dy, dz, 
                            length=5.0, normalize=True, color='r'
                        )

                path_xyz = np.array(path_xyz)
                ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], '*-', c=colors[groupIdx])
        
        plt.show()
        '''

        vis = VisulizerVista()
        obstacls_df = obs_df[obs_df['tag'] == 'Obstacle']
        obstacle_mesh = vis.create_pointCloud(obstacls_df[['x', 'y', 'z']].values)
        vis.plot(obstacle_mesh, (0.0, 1.0, 0.0))

        colors = np.random.uniform(0.0, 1.0, (5, 3))
        for groupIdx in group_keys:
            groupPath = smoother.groupMap[groupIdx]

            for pathIdx in groupPath.graphPathMap.keys():
                nodeIdxs_path = groupPath.graphPathMap[pathIdx]

                path_xyz = []
                for idx, nodeIdx in enumerate(nodeIdxs_path):
                    node = groupPath.graphNodeMap[nodeIdx]
                    path_xyz.append([node.x, node.y, node.z])
                path_xyz = np.array(path_xyz)
                
                resInfo = res_config[groupIdx][pathIdx]
                tube_mesh = vis.create_tube(path_xyz, radius=resInfo['grid_radius'])
                vis.plot(tube_mesh, color=tuple(colors[groupIdx]))

        vis.ploter.add_axes(line_width=5)
        vis.show()

    print("Runing Iteration %d ......" % outer_i)

### ------ Save Result
def polar2vec(polarVec, length=1.0):
    dz = length * math.sin(polarVec[1])
    dl = length * math.cos(polarVec[1])
    dx = dl * math.cos(polarVec[0])
    dy = dl * math.sin(polarVec[0])
    return np.array([dx, dy, dz])

for groupIdx in group_keys:
    groupPath = smoother.groupMap[groupIdx]

    for pathIdx in groupPath.graphPathMap.keys():
        nodeIdxs_path = groupPath.graphPathMap[pathIdx]

        path_xyzr = []
        for idx, nodeIdx in enumerate(nodeIdxs_path):
            node = groupPath.graphNodeMap[nodeIdx]
            path_xyzr.append([node.x, node.y, node.z, node.radius])
        path_xyzr = np.array(path_xyzr)
        
        resInfo = res_config[groupIdx][pathIdx]

        # start_vec = polar2vec(resInfo['startDire'])
        # interplot_xyz = path_xyzr[0, :3] + start_vec * 0.2
        # start_interplot_xyzr = np.array([interplot_xyz[0], interplot_xyz[1], interplot_xyz[2], path_xyzr[0, 3]])

        # end_vec = polar2vec(resInfo['endDire'])
        # interplot_xyz = path_xyzr[-1, :3] - end_vec * 0.2
        # end_interplot_xyzr = np.array([interplot_xyz[0], interplot_xyz[1], interplot_xyz[2], path_xyzr[-1, 3]])

        # path_xyzr = np.concatenate([
        #     path_xyzr[0, :].reshape((1, -1)),
        #     start_interplot_xyzr.reshape((1, -1)),
        #     path_xyzr[1:-1, :],
        #     end_interplot_xyzr.reshape((1, -1)),
        #     path_xyzr[-1, :].reshape((1, -1)),
        # ])

        resInfo['path_xyzr'] = path_xyzr

np.save(os.path.join(env_config['projectDir'], 'resPath_config.npy'), res_config)

### ------ Debug Vis 
vis = VisulizerVista()
obstacle_mesh = vis.create_pointCloud(obs_df[['x', 'y', 'z']].values)
vis.plot(obstacle_mesh, (0.0, 1.0, 0.0))

colors = np.random.uniform(0.0, 1.0, (5, 3))
for groupIdx in group_keys:
    res_info = res_config[groupIdx]
        
    for pathIdx in res_info.keys():
        path_info = res_info[pathIdx]
        path_xyzr = path_info['path_xyzr']
        tube_mesh = vis.create_tube(path_xyzr[:, :3], radius=path_info['grid_radius'])
        vis.plot(tube_mesh, color=tuple(colors[groupIdx]))

vis.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.set_xlim3d(-1, env_config['grid_x'] + 1)
# ax.set_ylim3d(-1, env_config['grid_y'] + 1)
# ax.set_zlim3d(-1, env_config['grid_z'] + 1)

# for groupIdx in res_config.keys():
#     res_info = res_config[groupIdx]

#     for pathIdx in res_info.keys():
#         path_info = res_info[pathIdx]
#         path_xyzr = path_info['path_xyzr']
#         ax.plot(path_xyzr[:, 0], path_xyzr[:, 1], path_xyzr[:, 2], '*-', c=colors[groupIdx])
        
#         alpha0, theta0 = path_info['startDire']
#         dz = math.sin(theta0)
#         dx = math.cos(theta0) * math.cos(alpha0)
#         dy = math.cos(theta0) * math.sin(alpha0)
#         ax.quiver(
#             path_xyzr[0, 0], path_xyzr[0, 1], path_xyzr[0, 2], 
#             dx, dy, dz, 
#             length=5.0, normalize=True, color='r'
#         )

#         alpha0, theta0 = path_info['endDire']
#         dz = math.sin(theta0)
#         dx = math.cos(theta0) * math.cos(alpha0)
#         dy = math.cos(theta0) * math.sin(alpha0)
#         ax.quiver(
#             path_xyzr[-1, 0], path_xyzr[-1, 1], path_xyzr[-1, 2], 
#             dx, dy, dz, 
#             length=5.0, normalize=True, color='r'
#         )

# plt.show()
