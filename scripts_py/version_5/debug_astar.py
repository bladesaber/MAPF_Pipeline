import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from build import mapf_pipeline

cond_params = {
    'y': 30,
    'x': 30,
    'z': 1,
    'num_of_groups': 1,

    'save_dir': '/home/quan/Desktop/MAPF_Pipeline/scripts_py/application',

    'use_obs': True,
}

agentsInfo = {
    'groupIdx': 0,
    'agentIdx': 0,
    
    # 'startPos': np.array([28, 26, 0]),
    # 'endPos': np.array([0, 2, 0]),

    'startPos': np.array([0, 2, 0]),
    'endPos': np.array([28, 26, 0]),
    
    'start_paddingDire': (0., 0., 1.),
    'end_paddingDire': (0.0, -1.0, 0.0),
    'radius': 1.0,
}

obs_df = pd.read_csv("/home/quan/Desktop/MAPF_Pipeline/scripts_py/application/obs.csv", index_col=0)
obs_df = obs_df[obs_df['z'] == 1.]
obs_df.drop_duplicates(subset=['x', 'y'], inplace=True)

constrains = []
for idx, row in obs_df.iterrows():
    constrains.append((
    row.x, row.y, row.z, row.radius
))

# plt.scatter(obs_df['x'].values, obs_df['y'].values)
# plt.show()

instance = mapf_pipeline.Instance(cond_params['x'], cond_params['y'], cond_params['z'])
model = mapf_pipeline.AngleAStar(1.0)
path = model.findPath(
    constrains, 
    instance,
    tuple(agentsInfo['startPos']),
    tuple(agentsInfo['endPos'])
)
path_xyz = []
for loc in path:
    (x, y, z) = instance.getCoordinate(loc)
    path_xyz.append([x, y, z])
path_xyz = np.array(path_xyz)

plt.plot(path_xyz[:, 0], path_xyz[:, 1])
plt.scatter(obs_df['x'].values, obs_df['y'].values)

plt.scatter([path_xyz[0, 0]], [path_xyz[0, 1]], c='r')
plt.scatter([path_xyz[-1, 0]], [path_xyz[-1, 1]], c='g')

xs, ys = np.meshgrid(np.arange(0, cond_params['x'], 1), np.arange(0, cond_params['y'], 1))
xys = np.concatenate((xs[..., np.newaxis], ys[..., np.newaxis]), axis=-1)
xys = xys.reshape((-1, 2))
plt.scatter(xys[:, 0], xys[:, 1], s=1.0)

plt.show()
