import numpy as np
import pandas as pd

from build import mapf_pipeline
from scripts.version_6.cbs_planner import CBS_Planner
from scripts.version_6.env_cfg import env_config

group_config = np.load('/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/grid_cfg.npy', allow_pickle=True).item()
obs_df = pd.read_csv('/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/obs_grid_df.csv', index_col=0)

instance = mapf_pipeline.Instance(env_config['x'], env_config['y'], env_config['z'])
env_config['group'] = group_config
cbs_planner = CBS_Planner(config=env_config, instance=instance, staticObs_df=obs_df)
cbs_planner.solve()

