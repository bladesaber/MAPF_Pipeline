import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from build import mapf_pipeline
from scripts.version_6.cbs_planner import CBS_Planner

grid_json_file = '/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/grid_env_cfg.json'
with open(grid_json_file, 'r') as f:
    env_config = json.load(f)

obs_df = pd.read_csv(env_config['static_grid_obs_pcd'], index_col=0)

### ------ auto compute path
instance = mapf_pipeline.Instance(env_config['x'], env_config['y'], env_config['z'])
cbs_planner = CBS_Planner(
    config=env_config, 
    instance=instance, 
    staticObs_df=obs_df
)
res = cbs_planner.solve()
# if res['status']:
#     result = res['res']
#     np.save(
#         '/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/res.npy', result
#     )

# ## ------ regular group Path
# res = np.load('/home/quan/Desktop/MAPF_Pipeline/scripts/version_6/app_dir/res.npy', allow_pickle=True).item()
# groupIdx_list = res.keys()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.grid(True)
# ax.set_xlim3d(0, 85)
# ax.set_ylim3d(0, 85)
# ax.set_zlim3d(0, 85)

# for groupIdx in groupIdx_list:
#     groupAgents = res[groupIdx]

#     for obj in groupAgents:
#         pathIdx = obj['pathIdx']
#         path_xyzrl = np.array(obj['path_xyzrl'])

#         print('[INFO]: (%d, %d, %d) -> (%d, %d, %d)' % (
#             path_xyzrl[0, 0], path_xyzrl[0, 1], path_xyzrl[0, 2],
#             path_xyzrl[-1, 0], path_xyzrl[-1, 1], path_xyzrl[-1, 2]
#         ))

#         ax.plot(path_xyzrl[:, 0], path_xyzrl[:, 1], path_xyzrl[:, 2], '*-', )

# plt.show()

