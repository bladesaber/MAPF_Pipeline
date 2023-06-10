import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import math

from build import mapf_pipeline
from scripts.version_6.cbs_planner import CBS_Planner

grid_json_file = '/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/grid_env_cfg.json'
with open(grid_json_file, 'r') as f:
    env_config = json.load(f)

obs_df = pd.read_csv(env_config['static_grid_obs_pcd'], index_col=0)
wall_obs_df = pd.read_csv(env_config['wall_obs_pcd'], index_col=0)
obs_df = pd.concat([obs_df, wall_obs_df], axis=0, ignore_index=True)

### Step1 ------ auto compute path
instance = mapf_pipeline.Instance(env_config['x'], env_config['y'], env_config['z'])
cbs_planner = CBS_Planner(
    config=env_config, 
    instance=instance, 
    staticObs_df=obs_df
)
res = cbs_planner.solve()
if res['status']:
    result = res['res']
    np.save(
        '/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/res.npy', result
    )

# ### Step2
# group_res = np.load('/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/res.npy', allow_pickle=True).item()

# group_keys = []
# group_config = {}
# for pipeConfig in env_config['pipeConfig']:
#     groupIdx = pipeConfig['groupIdx']
#     group_keys.append(groupIdx)

#     group_config[groupIdx] = {}
#     for pipe in pipeConfig['pipe']:
#         group_config[groupIdx].update({pipe['name']: pipe})

# group_keys = [0]

# linkPaths = {
#     0: [
#         {"start": 'p', 'end': 'p1'},
#         {"start": 'p', 'end': 'M1'},
#         {"start": 'p', 'end': 'p_valve'}
#     ],
#     1: [
#         {"start": 'B_valve', 'end': 'M3'},
#         {"start": 'B_valve', 'end': 'B'}
#     ],
#     2: [
#         {"start": 'T_valve', 'end': 'T'},
#         {"start": 'A2T', 'end': 'T'}
#     ],
#     3: [
#         {"start": 'A_valve', 'end': 'A2valve_01'},
#         {"start": 'A_valve', 'end': 'A2valve_02'}
#     ],
#     4: [
#         {"start": 'valve_01', 'end': 'A'},
#         {"start": 'valve_02', 'end': 'A'},
#         {"start": 'valve_03','end': 'A'},
#         {"start": 'valve_03','end': 'M2'}
#     ]
# }

# smoother = mapf_pipeline.SmootherXYZG2O()
# smoother.initOptimizer()

# for groupIdx in group_keys:
#     path_xyzrls = group_res[groupIdx]
#     for path_xyzrl in path_xyzrls:
#         path_xyzr = []
#         for xyzrl in path_xyzrl:
#             path_xyzr.append((xyzrl[0], xyzrl[1], xyzrl[2], xyzrl[3]))

#         smoother.add_Path(groupIdx, path_xyzr)

# for groupIdx in group_keys:
#     pipeConfig = group_config[groupIdx]
#     for pathIdx, link in enumerate(linkPaths[groupIdx]):

#         start_info = pipeConfig[link['start']]
#         end_info = pipeConfig[link['end']]

#         success = smoother.add_OptimizePath(
#             groupIdx, pathIdx,
#             start_info['grid_position'][0], start_info['grid_position'][1], start_info['grid_position'][2],
#             end_info['grid_position'][0], end_info['grid_position'][1], end_info['grid_position'][2],
#             (start_info['alpha'], start_info['theta']),
#             (end_info['alpha'], end_info['theta'])
#         )
#         print(link, success)

# for pipeConfig in env_config['pipeConfig']:
#     if pipeConfig['groupIdx'] in group_keys:
#         smoother.setMaxRadius(pipeConfig['groupIdx'] ,pipeConfig['grid_radius'])
# smoother.setBoundary(
#     xmin=0, ymin=0, zmin=0, 
#     xmax=env_config['x']-1, ymax=env_config['y']-1, zmax=env_config['z']-1
# )

# for idx, row in obs_df.iterrows():
#     smoother.insertStaticObs(row.x, row.y, row.z, row.radius, 0.0, 0.0)

# colors = np.random.uniform(0.0, 1.0, (len(group_keys), 3))
# for outer_i in range(3):

#     ### Step 2.1 Build Graph 
#     success = smoother.build_graph(
#         elasticBand_weight=1.0,
#         kinematic_weight=0.02,
#         obstacle_weight=1.0,
#         pipeConflict_weight=0.0,
#         boundary_weight=0.0
#     )
#     # print("build Graph:", success)

#     ### Step 2.2 Optimize
#     smoother.optimizeGraph(5, False)

#     ### Step 3.3 Update Vertex to Node
#     smoother.update2groupVertex()

#     # ### Step 2.4 Clear Graph
#     smoother.clear_graph()

#     ### Debug Vis
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.set_xlim3d(-1, env_config['x']+1)
#     ax.set_ylim3d(-1, env_config['y']+1)
#     ax.set_zlim3d(-1, env_config['z']+1)

#     for group_key in smoother.groupMap.keys():
#         groupPath = smoother.groupMap[group_key]

#         for pathIdx in groupPath.graphPathMap.keys():
#             nodeIdxs_path = groupPath.graphPathMap[pathIdx]

#             path_xyz = []
#             for idx, nodeIdx in enumerate(nodeIdxs_path):
#                 node = groupPath.graphNodeMap[nodeIdx]
#                 path_xyz.append([node.x, node.y, node.z])

#                 if (node.fixed):
#                     dz = math.sin(node.theta)
#                     dl = math.cos(node.theta)
#                     dx = dl * math.cos(node.alpha)
#                     dy = dl * math.sin(node.alpha)
#                     ax.quiver(
#                         node.x, node.y, node.z, 
#                         dx, dy, dz, 
#                         length=1.0, normalize=True, color='r'
#                     )

#             path_xyz = np.array(path_xyz)
#             ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], '*-', c=colors[groupIdx])
    
#     plt.show()

