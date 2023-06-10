import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import json

from scripts.visulizer import VisulizerVista

from build import mapf_pipeline

grid_json_file = '/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/grid_env_cfg.json'
with open(grid_json_file, 'r') as f:
    env_config = json.load(f)

obs_df = pd.read_csv(env_config['static_grid_obs_pcd'], index_col=0)
wall_obs_df = pd.read_csv(env_config['wall_obs_pcd'], index_col=0)
obs_df = pd.concat([obs_df, wall_obs_df], axis=0, ignore_index=True)

pathRes = np.load('/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/res.npy', allow_pickle=True).item()

group_keys = [0, 1, 2, 3, 4]

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
        {"start": 'p', 'end': 'p1'},
        {"start": 'p', 'end': 'M1'},
        {"start": 'p', 'end': 'p_valve'}
    ],
    1: [
        {"start": 'B_valve', 'end': 'M3'},
        {"start": 'B_valve', 'end': 'B'}
    ],
    2: [
        {"start": 'T_valve', 'end': 'T'},
        {"start": 'A2T', 'end': 'T'}
    ],
    3: [
        {"start": 'A_valve', 'end': 'A2valve_01'},
        {"start": 'A_valve', 'end': 'A2valve_02'}
    ],
    4: [
        {"start": 'valve_01', 'end': 'A'},
        {"start": 'valve_02', 'end': 'A'},
        {"start": 'valve_03','end': 'A'},
        {"start": 'valve_03','end': 'M2'}
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
        
for groupIdx in group_keys:
    pipeConfig = groupAgentConfig[groupIdx]
    for pathIdx, link in enumerate(groupAgentLinks[groupIdx]):
        start_info = pipeConfig[link['start']]
        end_info = pipeConfig[link['end']]

        success = smoother.add_OptimizePath(
            groupIdx, pathIdx,
            start_info['grid_position'][0], start_info['grid_position'][1], start_info['grid_position'][2],
            end_info['grid_position'][0], end_info['grid_position'][1], end_info['grid_position'][2],
            (start_info['alpha'], start_info['theta']),
            (end_info['alpha'], end_info['theta'])
        )
        print(link, success)

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
    elasticBand_weight,
    kinematic_weight,
    obstacle_weight,
    pipeConflict_weight,
    boundary_weight,
    run_times
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

    # if outer_i == 0:
    #     smoother.info()
    #     smoother.loss_info(
    #         elasticBand_weight=0.0,
    #         kinematic_weight=0.0,
    #         obstacle_weight=0.0,
    #         pipeConflict_weight=0.0,
    #         boundary_weight=1.0
    #     )
    #     print()

    ### Step 2.2 Optimize
    smoother.optimizeGraph(run_times, False)

    # smoother.loss_info(
    #     elasticBand_weight=0.0,
    #     kinematic_weight=0.0,
    #     obstacle_weight=0.0,
    #     pipeConflict_weight=0.0,
    #     boundary_weight=1.0
    # )

    ### Step 3.3 Update Vertex to Node
    smoother.update2groupVertex()

    # ### Step 2.4 Clear Graph
    smoother.clear_graph()

for outer_i in range(100):

    # for _ in range(1):
    #     run_smooth(
    #         smoother=smoother,
    #         elasticBand_weight=1.0,
    #         kinematic_weight=0.0,
    #         obstacle_weight=1.0,
    #         pipeConflict_weight=0.0,
    #         boundary_weight=0.1,
    #         run_times=100
    #     )
    #     run_smooth(
    #         smoother=smoother,
    #         elasticBand_weight=0.1,
    #         kinematic_weight=5.0,
    #         obstacle_weight=1.0,
    #         pipeConflict_weight=0.0,
    #         boundary_weight=0.1,
    #         run_times=20
    #     )
    # run_smooth(
    #     smoother=smoother,
    #     elasticBand_weight=1.0,
    #     kinematic_weight=10.0,
    #     obstacle_weight=1.0,
    #     pipeConflict_weight=0.0,
    #     boundary_weight=0.1,
    #     run_times=2000
    # )
    # run_smooth(
    #     smoother=smoother,
    #     elasticBand_weight=1.0,
    #     kinematic_weight=0.0,
    #     obstacle_weight=1.0,
    #     pipeConflict_weight=0.0,
    #     boundary_weight=0.1,
    #     run_times=500
    # )
    run_smooth(
        smoother=smoother,
        elasticBand_weight=1.0,
        kinematic_weight=8.0,
        obstacle_weight=1.0,
        pipeConflict_weight=1.0,
        boundary_weight=1.0,
        run_times=500
    )

    ### ------ Debug Vis path
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

    ### ------ Debug Vis 
    vis = VisulizerVista()
    obstacle_mesh = vis.create_pointCloud(obs_df[['x', 'y', 'z']].values)
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
            tube_mesh = vis.create_tube(path_xyz[:, :3], radius=groupAgentConfig[groupIdx]['grid_radius'])
            vis.plot(tube_mesh, color=tuple(colors[groupIdx]))

    vis.show()

