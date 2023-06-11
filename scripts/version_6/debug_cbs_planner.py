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
