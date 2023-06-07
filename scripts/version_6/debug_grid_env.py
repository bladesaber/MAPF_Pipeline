import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import open3d as o3d

from scripts.version_6.env_cfg import pipe_config, obstacle_config, env_config
from scripts.version_6 import static_obstacle

vis = o3d.visualization.Visualizer()
vis.create_window(height=720, width=960)

obs_df = pd.read_csv('/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/obs_df.csv', index_col=0)
obs_df = obs_df / env_config['resolution']

obs_pcd = o3d.geometry.PointCloud()
obs_pcd.points = o3d.utility.Vector3dVector(obs_df[['x', 'y', 'z']].values)
vis.add_geometry(obs_pcd)

colors = np.random.uniform(0.0, 1.0, size=(len(pipe_config), 3))
for groupIdx in pipe_config.keys():
    cfgs = pipe_config[groupIdx]
    for cfg in cfgs:
        cfg['position'] = tuple((np.array(cfg['position']) // env_config['resolution']).astype(np.int))
        cfg['radius'] = cfg['radius'] / env_config['resolution']
        
        print(cfg)

        mesh = static_obstacle.create_create_sphere(
            cfg['position'][0], cfg['position'][1], cfg['position'][2], cfg['radius']
        )
        mesh.paint_uniform_color(colors[groupIdx])
        vis.add_geometry(mesh)

obs_df.to_csv('/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/obs_grid_df.csv')
np.save('/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/grid_cfg', pipe_config)

vis.run()
vis.destroy_window()