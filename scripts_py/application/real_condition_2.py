import numpy as np
import pandas as pd
import os
import open3d as o3d
from scripts_py.visulizer import VisulizerVista

x_shift = 35.
y_shift = 35.
z_shift = 0.

agentsInfo = {
    0:{
        'groupIdx': 0,
        'agentIdx': 0,
        'startPos': np.array([
            35. + x_shift, 
            30.0 + y_shift, 
            30.0 + z_shift
        ]),
        'endPos': np.array([
            -35. + x_shift, 
            -30.0 + y_shift, 
            15.0 + z_shift
        ]),
        'start_paddingDire': (0., 0., 1.),
        'end_paddingDire': (0.0, -1.0, 0.0),
        'radius': 2.5,
    },
    # 1:{
    #     'groupIdx': 0,
    #     'agentIdx': 1,
    #     'startPos': np.array([
    #         -35 + x_shift, 
    #         30.0 + y_shift, 
    #         15.0 + z_shift
    #     ]),
    #     'endPos': np.array([
    #         -35 + x_shift, 
    #         -30.0 + y_shift, 
    #         15.0 + z_shift
    #     ]),
    #     'start_paddingDire': (-1., 0., 0.),
    #     'end_paddingDire': (0.0, -1.0, 0.0),
    #     'radius': 2.5,
    # },
    # 2:{
    #     'groupIdx': 0,
    #     'agentIdx': 2,
    #     'startPos': np.array([
    #         -15 + x_shift, 
    #         5.0 + y_shift, 
    #         30.0 + z_shift
    #     ]),
    #     'endPos': np.array([
    #         -35 + x_shift, 
    #         -30.0 + y_shift, 
    #         15.0 + z_shift
    #     ]),
    #     'start_paddingDire': (0., 0., 1.),
    #     'end_paddingDire': (0.0, -1.0, 0.0),
    #     'radius': 2.5,
    # },
}

## ------ create fake obstacle mesh
# obs_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=12, height=40.0)
# obs_mesh.translate(np.array([35., 35., 20.]), relative=False)
# obs_mesh.compute_vertex_normals()
# o3d.io.write_triangle_mesh('/home/quan/Desktop/MAPF_Pipeline/scripts_py/application/replace.STL', obs_mesh)

### ------ visualize fake condition
# random_colors = np.random.uniform(0.0, 1.0, size=(4, 3))
# vis = VisulizerVista()
# for agentIdx in agentsInfo.keys():
#     agentInfo = agentsInfo[agentIdx]

#     mesh = vis.create_sphere(
#         xyz=np.array(agentInfo['startPos']), radius=agentInfo['radius']
#     )
#     vis.plot(mesh, tuple(random_colors[agentInfo['groupIdx']]))
#     mesh = vis.create_sphere(
#         xyz=np.array(agentInfo['endPos']), radius=agentInfo['radius']
#     )
#     vis.plot(mesh, tuple(random_colors[agentInfo['groupIdx']]))

# stl_file = '/home/quan/Desktop/MAPF_Pipeline/scripts_py/application/replace.STL'
# obs_mesh = vis.read_file(stl_file)
# vis.plot(obs_mesh, (0.0, 1.0, 0.0))
# vis.show()

### ------ make grid
resolution = 2.5

# for agentIdx in agentsInfo.keys():
#     agentInfo = agentsInfo[agentIdx]
#     agentInfo['startPos'] = (agentInfo['startPos'] / resolution).astype(int)
#     agentInfo['endPos'] = (agentInfo['endPos'] / resolution).astype(int)
#     agentInfo['radius'] = agentInfo['radius'] / resolution

#     print('AgentIdx:%d GroupIdx:%d' % (agentInfo['agentIdx'], agentInfo['groupIdx']))
#     print('  startPos:', agentInfo['startPos'])
#     print('  endPos:', agentInfo['endPos'])
#     print('  radius:', agentInfo['radius'])

# np.save(
#     '/home/quan/Desktop/MAPF_Pipeline/scripts_py/application/real_condition2_map',
#     agentsInfo
# )

# ### ------ make obstacle pcd
# mesh = o3d.io.read_triangle_mesh('/home/quan/Desktop/MAPF_Pipeline/scripts_py/application/replace.STL')
# mesh.compute_vertex_normals()

# pcd = mesh.sample_points_poisson_disk(2000)
# pcd_np = np.asarray(pcd.points)

# reso = 1.0
# pcd_df = pd.DataFrame(pcd_np, columns=['x', 'y', 'z'])
# pcd_df['x'] = np.round(pcd_df['x'] / reso, decimals=0) * reso
# pcd_df['y'] = np.round(pcd_df['y'] / reso, decimals=0) * reso
# pcd_df['z'] = np.round(pcd_df['z'] / reso, decimals=0) * reso
# pcd_df['tag'] = pcd_df['x'].astype(str) + pcd_df['y'].astype(str) + pcd_df['z'].astype(str)
# pcd_df.drop_duplicates(subset=['tag'], inplace=True)
# pcd_df['radius'] = 0.0

# pcd_df['x'] = np.round(pcd_df['x'] / resolution, decimals=0)
# pcd_df['y'] = np.round(pcd_df['y'] / resolution, decimals=0)
# pcd_df['z'] = np.round(pcd_df['z'] / resolution, decimals=0)

# pcd_df[['x', 'y', 'z', 'radius']].to_csv(
#     '/home/quan/Desktop/MAPF_Pipeline/scripts_py/application/obs.csv'
# )

### ------ visulaize simulation run condition
# agentsInfo = np.load(
#     '/home/quan/Desktop/MAPF_Pipeline/scripts_py/application/real_condition2_map.npy', allow_pickle=True
# ).item()

# pcd_df = pd.read_csv('/home/quan/Desktop/MAPF_Pipeline/scripts_py/application/obs.csv', index_col=0)

# vis = VisulizerVista()

# for agentIdx in agentsInfo.keys():
#     agentInfo = agentsInfo[agentIdx]

#     print('AgentIdx:%d GroupIdx:%d' % (agentInfo['agentIdx'], agentInfo['groupIdx']))
#     print('  startPos:', agentInfo['startPos'])
#     print('  endPos:', agentInfo['endPos'])
#     print('  radius:', agentInfo['radius'])

#     mesh = vis.create_sphere(
#         xyz=np.array(agentInfo['startPos']), radius=agentInfo['radius']
#     )
#     vis.plot(mesh, (1.0, 0.0, 0.0))
#     mesh = vis.create_sphere(
#         xyz=np.array(agentInfo['endPos']), radius=agentInfo['radius']
#     )
#     vis.plot(mesh, (1.0, 0.0, 0.0))

# obs_mesh = vis.create_pointCloud(pcd_df[['x', 'y', 'z']].values)
# vis.plot(obs_mesh, (0.0, 1.0, 0.0))

# vis.show()

### ----------------  Auto Compute
from scripts_py.version_4.cbs import CBSSolver
from build import mapf_pipeline

cond_params = {
    'y': 30,
    'x': 30,
    'z': 15,
    'num_of_groups': 1,

    'save_dir': '/home/quan/Desktop/MAPF_Pipeline/scripts_py/application',

    'use_obs': True,
    'stl_file': '/home/quan/Desktop/MAPF_Pipeline/scripts_py/application/replace.STL',
    'csv_file': '/home/quan/Desktop/MAPF_Pipeline/scripts_py/application/obs.csv',
}

### ------ planning
print("Start Creating Map......")
agentInfos = np.load(
    '/home/quan/Desktop/MAPF_Pipeline/scripts_py/application/real_condition2_map.npy',
    allow_pickle=True
).item()

for key in agentInfos.keys():
    agentInfo = agentInfos[key]
    print(agentInfo)

planner = CBSSolver(cond_params, agentInfos)

if cond_params['use_obs']:
    staticObs_df = pd.read_csv(cond_params['csv_file'], index_col=0)
else:
    staticObs_df = pd.DataFrame()

success_node = planner.solve(staticObs_df=staticObs_df)

if success_node is not None:
    if cond_params['use_obs']:
        obs_mesh = VisulizerVista.create_pointCloud(staticObs_df[['x', 'y', 'z']].values)
        planner.print_NodeGraph(success_node, obs_mesh=obs_mesh)
    else:
        planner.print_NodeGraph(success_node)

    # for agentIdx in success_node.agentMap.keys():
    #     agent = success_node.agentMap[agentIdx]

    #     detailPath = agent.getDetailPath()
    #     agentInfos[agentIdx]['detailPath'] = detailPath
        
    # np.save(os.path.join(cond_params['save_dir'], 'agentInfo'), agentInfos)

# ### -------- smoothing
# from scripts_py import vis_utils
# import matplotlib.pyplot as plt

# smoother = mapf_pipeline.RandomStep_Smoother(
#     xmin = 0.0, xmax = cond_params['x'] + 2.0,
#     ymin = 0.0, ymax = cond_params['y'] + 2.0,
#     zmin = 0.0, zmax = cond_params['z'] + 2.0,
#     stepReso = 0.025
# )
# smoother.wSmoothness = 1.5
# smoother.wCurvature = 0.15
# smoother.wGoupPairObs = 1.0
# smoother.wStaticObs = 1.0

# if cond_params['use_obs']:
#     smoother.wStaticObs = 1.0

#     obs_df:pd.DataFrame = pd.read_csv(cond_params['csv_file'], index_col=0)
#     for idx, row in obs_df.iterrows():
#         smoother.insertStaticObs(row.x, row.y, row.z, row.radius)

# agentInfos = np.load(os.path.join(cond_params['save_dir'], 'agentInfo')+'.npy', allow_pickle=True).item()

# oldPaths = {}
# for agentIdx in agentInfos.keys():
#     paddingPath = smoother.paddingPath(
#         agentInfos[agentIdx]['detailPath'], 
#         agentInfos[agentIdx]['start_paddingDire'],
#         agentInfos[agentIdx]['end_paddingDire'],
#         x_shift=1.0, y_shift=1.0, z_shift=1.0
#     )
#     paddingPath = smoother.detailSamplePath(paddingPath, 0.35)

#     agentInfos[agentIdx]['paddingPath'] = paddingPath
    
#     smoother.addDetailPath(
#         groupIdx=agentInfos[agentIdx]['groupIdx'], 
#         pathIdx=agentIdx,
#         radius=agentInfos[agentIdx]['radius'],
#         detailPath=agentInfos[agentIdx]['paddingPath']
#     )

#     oldPaths[agentIdx] = np.array(agentInfos[agentIdx]['paddingPath'])[:, :3]

# ### ---------- Just For Debug
# newPaths = {}
# for _ in range(2):
#     smoother.smoothPath(updateTimes=30)

#     ### -------- vis debug
#     ax = vis_utils.create_Graph3D(
#         xmax=cond_params['x'] + 4.0, 
#         ymax=cond_params['y'] + 4.0, 
#         zmax=cond_params['z'] + 4.0
#     )

#     for groupIdx in smoother.groupMap.keys():
#         groupInfo = smoother.groupMap[groupIdx]
#         groupPath = groupInfo.path

#         for agentIdx in groupPath.pathIdxs_set:
#             pathXYZR = np.array(groupPath.extractPath(agentIdx))
#             newPaths[agentIdx] = pathXYZR[:, :3]

#     for agentIdx in oldPaths.keys():
#         vis_utils.plot_Path3D(ax, newPaths[agentIdx][:, :3], color='r')
#         vis_utils.plot_Path3D(ax, oldPaths[agentIdx][:, :3], color='b')
        
#     plt.show()

#     oldPaths = newPaths
# ### --------------------------------------------------------
# # smoother.smoothPath(updateTimes=50)

# for agentIdx in agentInfos.keys():
#     groupIdx = agentInfos[agentIdx]['groupIdx']
#     groupInfo = smoother.groupMap[groupIdx]
#     groupPath = groupInfo.path

#     pathXYZR = groupPath.extractPath(agentIdx)
#     agentInfos[agentIdx]['smoothXYZR'] = np.array(pathXYZR)

# random_colors = np.random.uniform(0.0, 1.0, size=(cond_params['num_of_groups'], 3))
# vis = VisulizerVista()
# for agentIdx in agentInfos.keys():
#     mainXYZ = agentInfos[agentIdx]['smoothXYZR'][:, :3]

#     startDire = np.array(agentInfos[agentIdx]['start_paddingDire'])
#     startXYZ = np.array([
#         mainXYZ[0] + startDire * 5.,
#         mainXYZ[0] + startDire * 4.,
#         mainXYZ[0] + startDire * 3.,
#         mainXYZ[0] + startDire * 2.,
#         mainXYZ[0] + startDire * 1.,
#     ])
#     endDire = np.array(agentInfos[agentIdx]['end_paddingDire'])
#     endXYZ = np.array([
#         mainXYZ[-1] + endDire * 1.,
#         mainXYZ[-1] + endDire * 2.,
#         mainXYZ[-1] + endDire * 3.,
#         mainXYZ[-1] + endDire * 4.,
#         mainXYZ[-1] + endDire * 5.,
#     ])

#     mainXYZ = np.concatenate([startXYZ, mainXYZ, endXYZ], axis=0)

#     tube_mesh = vis.create_tube(
#         mainXYZ, 
#         radius=agentInfos[agentIdx]['radius']
#     )
#     vis.plot(tube_mesh, color=tuple(random_colors[agentInfos[agentIdx]['groupIdx']]))
    
# if cond_params['use_obs']:
#     obs_mesh = vis.create_pointCloud(obs_df[['x', 'y', 'z']].values)
#     vis.plot(obs_mesh, (0.0, 1.0, 0.0))

# vis.show()