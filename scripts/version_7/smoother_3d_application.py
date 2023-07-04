import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import json

from scripts.visulizer import VisulizerVista

from build import mapf_pipeline

grid_json_file = '/home/quan/Desktop/MAPF_Pipeline/scripts/version_7/app_dir/grid_env_cfg.json'
with open(grid_json_file, 'r') as f:
    env_config = json.load(f)

obs_df = pd.read_csv(env_config['static_grid_obs_pcd'], index_col=0)
wall_obs_df = pd.read_csv(env_config['wall_obs_pcd'], index_col=0)
obs_df = pd.concat([obs_df, wall_obs_df], axis=0, ignore_index=True)

pathRes = np.load('/home/quan/Desktop/MAPF_Pipeline/scripts/version_7/app_dir/res.npy', allow_pickle=True).item()

group_keys = [0, 1, 2, 3, 4]
# group_keys = [4]

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
        {"start": 'p', 'end': 'p1', 'startFlexRatio': 0.0, 'endFlexRatio': 0.2},
        {"start": 'p', 'end': 'M1', 'startFlexRatio': 0.0, 'endFlexRatio': 0.4},
        {"start": 'p', 'end': 'p_valve', 'startFlexRatio': 0.0, 'endFlexRatio': 0.2}
    ],
    1: [
        {"start": 'B_valve', 'end': 'M3', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
        {"start": 'B_valve', 'end': 'B', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}
    ],
    2: [
        {"start": 'T_valve', 'end': 'T', 'startFlexRatio': 0.2, 'endFlexRatio': 0.0},
        {"start": 'A2T', 'end': 'T', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}
    ],
    3: [
        {"start": 'A_valve', 'end': 'A2valve_01', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
        {"start": 'A_valve', 'end': 'A2valve_02', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}
    ],
    4: [
        # {"start": 'valve_01', 'end': 'A', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
        # {"start": 'valve_02', 'end': 'A', 'startFlexRatio': 0.25, 'endFlexRatio': 0.0},
        # {"start": 'valve_03','end': 'A', 'startFlexRatio': 0.45, 'endFlexRatio': 0.0},
        # {"start": 'valve_03','end': 'M2', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}

        {"start": 'valve_01', 'end': 'valve_02', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
        {"start": 'valve_02', 'end': 'A', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0},
        {"start": 'valve_03','end': 'A', 'startFlexRatio': 0.45, 'endFlexRatio': 0.0},
        {"start": 'valve_03','end': 'M2', 'startFlexRatio': 0.0, 'endFlexRatio': 0.0}
    ]
}

smoother = mapf_pipeline.SmootherXYZG2O()
smoother.initOptimizer()

xmin=0.0
ymin=0.0
zmin=0.0
xmax=63.0
ymax=63.0
zmax=63.0
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

for outer_i in range(50):
    # smoother.elasticBand_targetLength = 0.1
    smoother.elasticBand_minLength = 0.5

    run_smooth(
        smoother=smoother,
        elasticBand_weight=1.0,
        kinematic_weight=5.0,
        obstacle_weight=0.0,
        pipeConflict_weight=0.0,
        boundary_weight=0.0,
        run_times=10
    )

    ### ------ Debug Vis path
    if outer_i % 100 == 0:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim3d(-1, 63)
        ax.set_ylim3d(-1, 63)
        ax.set_zlim3d(-1, 63)

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
    
    print("Runing Iteration %d ......" % outer_i)

### ------ Save Result
for groupIdx in group_keys:
    groupPath = smoother.groupMap[groupIdx]

    for pathIdx in groupPath.graphPathMap.keys():
        nodeIdxs_path = groupPath.graphPathMap[pathIdx]

        path_xyzr = []
        for idx, nodeIdx in enumerate(nodeIdxs_path):
            node = groupPath.graphNodeMap[nodeIdx]
            path_xyzr.append([node.x, node.y, node.z, node.radius])
        path_xyzr = np.array(path_xyzr)

        res_config[groupIdx][pathIdx]['path_xyzr'] = path_xyzr
np.save('/home/quan/Desktop/MAPF_Pipeline/scripts/version_7/app_dir/resPath_config.npy', res_config)

### ------ Debug Vis 
# vis = VisulizerVista()
# obstacle_mesh = vis.create_pointCloud(obs_df[['x', 'y', 'z']].values)
# vis.plot(obstacle_mesh, (0.0, 1.0, 0.0))

# colors = np.random.uniform(0.0, 1.0, (5, 3))
# for groupIdx in group_keys:
#     res_info = res_config[groupIdx]
        
#     for pathIdx in res_info.keys():
#         path_info = res_info[pathIdx]
#         path_xyzr = path_info['path_xyzr']
#         tube_mesh = vis.create_tube(path_xyzr[:, :3], radius=path_info['grid_radius'])
#         vis.plot(tube_mesh, color=tuple(colors[groupIdx]))

# vis.show()

