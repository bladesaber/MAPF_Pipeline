import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import math
import os

from build import mapf_pipeline
from scripts_py.version_6.cbs_planner import CBS_Planner

cond_json_file = '/home/quan/Desktop/tempary/application_pipe/cond.json'
with open(cond_json_file, 'r') as f:
    env_config = json.load(f)

obs_df = pd.read_csv(env_config['obstacleSavePath'], index_col=0)
print('[DEBUG]: Obstacle Num:', obs_df.shape[0])

### Step1 ------ auto compute path
instance = mapf_pipeline.Instance(env_config['grid_x'], env_config['grid_y'], env_config['grid_z'])
cbs_planner = CBS_Planner(
    config=env_config, 
    instance=instance, 
    staticObs_df=obs_df
)
res = cbs_planner.solve()
if res['status']:
    result = res['res']
    np.save(os.path.join(env_config['projectDir'], 'res.npy'), result)
else:
    print("Not Found")