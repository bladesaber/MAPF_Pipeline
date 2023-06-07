import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import open3d as o3d

from scripts.version_6.env_cfg import pipe_config, obstacle_config
from scripts.version_6 import static_obstacle

vis = o3d.visualization.Visualizer()
vis.create_window(height=720, width=960)

obs_pcds = []
for cfg in obstacle_config:
    if cfg['type'] == 'Z_support':
        mesh = static_obstacle.create_Z_support_mesh(
            cfg['position'][0], cfg['position'][1], cfg['position'][2],
            cfg['radius'], cfg['height']
        )

    elif cfg['type'] == 'Z_screw':
        mesh = static_obstacle.create_Z_screw_mesh(
            cfg['position'][0], cfg['position'][1], cfg['position'][2],
            cfg['radius'], cfg['height']
        )

    elif cfg['type'] == 'X_valve':
        mesh = static_obstacle.create_X_valve_mesh(
            cfg['position'][0], cfg['position'][1], cfg['position'][2],
            cfg['radius'], cfg['height']
        )
    elif cfg['type'] == 'Z_valve':
        mesh = static_obstacle.create_Z_valve_mesh(
            cfg['position'][0], cfg['position'][1], cfg['position'][2],
            cfg['radius'], cfg['height']
        )
    
    pcd:o3d.geometry.PointCloud = mesh.sample_points_poisson_disk(number_of_points=cfg['sample_num'])
    pcd_np = np.asarray(pcd.points)
    obs_pcds.append(pcd_np)

    # vis.add_geometry(mesh)

obs_pcds = np.concatenate(obs_pcds, axis=0)
obs_df = pd.DataFrame(obs_pcds, columns=['x', 'y', 'z'])
obs_df['radius'] = 0.0
obs_df.to_csv('/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/obs_df.csv')

obs_pcd = o3d.geometry.PointCloud()
obs_pcd.points = o3d.utility.Vector3dVector(obs_df[['x', 'y', 'z']].values)
vis.add_geometry(obs_pcd)

colors = np.random.uniform(0.0, 1.0, size=(len(pipe_config), 3))
for groupIdx in pipe_config.keys():
    cfgs = pipe_config[groupIdx]
    for cfg in cfgs:
        mesh = static_obstacle.create_create_sphere(
            cfg['position'][0], cfg['position'][1], cfg['position'][2], cfg['radius']
        )
        mesh.paint_uniform_color(colors[groupIdx])
        vis.add_geometry(mesh)

vis.run()
vis.destroy_window()