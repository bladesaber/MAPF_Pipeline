import numpy as np
import pandas as pd
import math
import json
import os
import matplotlib.pyplot as plt

from scripts_py.visulizer import VisulizerVista
from scripts_py.version_6 import o3d_helper

grid_json_file = '/home/quan/Desktop/MAPF_Pipeline/scripts_py/version_6/app_dir/grid_env_cfg.json'
with open(grid_json_file, 'r') as f:
    env_config = json.load(f)

res_config = np.load(
    '/home/quan/Desktop/MAPF_Pipeline/scripts_py/version_6/app_dir/resPath_config.npy',
    allow_pickle=True
).item()

vis = VisulizerVista()

obs_df = pd.read_csv(env_config['static_grid_obs_pcd'], index_col=0)
obstacle_mesh = vis.create_pointCloud(obs_df[['x', 'y', 'z']].values)
vis.plot(obstacle_mesh, (1.0, 0.5, 0.25), opacity=1.0)

wall_obs_df = pd.read_csv(env_config['wall_obs_pcd'], index_col=0)
obstacle_mesh = vis.create_pointCloud(wall_obs_df[['x', 'y', 'z']].values)
vis.plot(obstacle_mesh, (0.5, 0.5, 0.5), opacity=0.5)

# for file in os.listdir(env_config['mesh_dir']):
#     path = os.path.join(env_config['mesh_dir'], file)
#     mesh = vis.read_file(path)
#     vis.plot(mesh, (0.5, 0.5, 0.5), opacity=0.5)

# colors = np.random.uniform(0.0, 1.0, (10, 3))
colors = np.array([
    [1.0, 0.0, 0.0], # P
    [0.0, 0.0, 1.0], # B
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0]
])

for groupIdx in res_config.keys():
    res_info = res_config[groupIdx]
        
    for pathIdx in res_info.keys():
        path_info = res_info[pathIdx]
        path_xyzr = path_info['path_xyzr']
        tube_mesh = vis.create_tube(path_xyzr[:, :3], radius=path_info['grid_radius'])
        vis.plot(tube_mesh, color=tuple(colors[groupIdx]))

# for pipeConfig in env_config['pipeConfig']:
#     for pipe in pipeConfig['pipe']:
#         # mesh = vis.create_sphere(np.array([
#         #     pipe['grid_position'][0], pipe['grid_position'][1], pipe['grid_position'][2]
#         # ]), radius=pipeConfig['grid_radius'])
#         mesh = vis.create_box(np.array([
#             pipe['grid_position'][0], pipe['grid_position'][1], pipe['grid_position'][2]
#         ]), pipeConfig['grid_radius'] * 2.0)
#         vis.plot(mesh, color=tuple(colors[pipeConfig['groupIdx']]))

vis.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim3d(-1, 63)
ax.set_ylim3d(-1, 63)
ax.set_zlim3d(-1, 63)

for groupIdx in res_config.keys():
    res_info = res_config[groupIdx]

    for pathIdx in res_info.keys():
        path_info = res_info[pathIdx]
        path_xyzr = path_info['path_xyzr']
        ax.plot(path_xyzr[:, 0], path_xyzr[:, 1], path_xyzr[:, 2], '*-', c=colors[groupIdx])
        
        alpha0, theta0 = path_info['startDire']
        dz = math.sin(theta0)
        dx = math.cos(theta0) * math.cos(alpha0)
        dy = math.cos(theta0) * math.sin(alpha0)
        ax.quiver(
            path_xyzr[0, 0], path_xyzr[0, 1], path_xyzr[0, 2], 
            dx, dy, dz, 
            length=5.0, normalize=True, color='r'
        )

        alpha0, theta0 = path_info['endDire']
        dz = math.sin(theta0)
        dx = math.cos(theta0) * math.cos(alpha0)
        dy = math.cos(theta0) * math.sin(alpha0)
        ax.quiver(
            path_xyzr[-1, 0], path_xyzr[-1, 1], path_xyzr[-1, 2], 
            dx, dy, dz, 
            length=5.0, normalize=True, color='r'
        )

plt.show()