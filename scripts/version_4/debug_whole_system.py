import numpy as np
import pandas as pd
import os

from scripts.version_4.map_creator import MapGen
from scripts.version_4.cbs import CBSSolver
from build import mapf_pipeline
from scripts.visulizer import VisulizerVista
from scripts import vis_utils
import matplotlib.pyplot as plt

cond_params = {
    'y': 15,
    'x': 15,
    'z': 15,
    'num_of_groups': 3,
    'radius_choices': [
        0.45, 
        0.85, 
        # 1.25
    ],
    'pipe_type': [
        'one2one', 
        'more2one'
    ],
    'pipe_choice': [2, 3],

    'save_path': '/home/quan/Desktop/MAPF_Pipeline/scripts/version_4/map',
    "load": True,

    'save_dir': '/home/quan/Desktop/MAPF_Pipeline/scripts/version_4/',

    'stl_file': '/home/quan/Desktop/MAPF_Pipeline/scripts/version_4/obs.stl',
    'csv_file': '/home/quan/Desktop/MAPF_Pipeline/scripts/version_4/obs.csv',
    'use_obs': True,
}

def planner_find_Path():
    print("Start Creating Map......")
    map = MapGen(cond_params)
    if cond_params['load']:
        map.load()
    else:
        map.generateMap(cond_params['num_of_groups'])
        map.save()

    agentInfos = map.agentInfos
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
            planner.print_NodeGraph(success_node, obs_file=cond_params['stl_file'])
        else:
            planner.print_NodeGraph(success_node)

        for agentIdx in success_node.agentMap.keys():
            agent = success_node.agentMap[agentIdx]

            detailPath = agent.getDetailPath()
            agentInfos[agentIdx]['detailPath'] = detailPath
        
        np.save(os.path.join(cond_params['save_dir'], 'agentInfo'), agentInfos)

def smoothPath():
    smoother = mapf_pipeline.RandomStep_Smoother(
        xmin = 0.0, xmax = cond_params['x'] + 2.0,
        ymin = 0.0, ymax = cond_params['y'] + 2.0,
        zmin = 0.0, zmax = cond_params['z'] + 2.0,
        stepReso = 0.01
    )
    smoother.wSmoothness = 1.0
    smoother.wCurvature = 0.0
    smoother.wGoupPairObs = 0.0
    smoother.wStaticObs = 0.0

    agentInfos = np.load(
        os.path.join(cond_params['save_dir'], 'agentInfo')+'.npy', allow_pickle=True
    ).item()

    # agentInfos = {0: agentInfos[0]}

    oldPaths = {}
    for agentIdx in agentInfos.keys():
        paddingPath = smoother.paddingPath(
            agentInfos[agentIdx]['detailPath'], 
            agentInfos[agentIdx]['start_paddingDire'],
            agentInfos[agentIdx]['end_paddingDire'],
            x_shift=1.0, y_shift=1.0, z_shift=1.0
        )
        paddingPath = smoother.detailSamplePath(paddingPath, 0.35)

        agentInfos[agentIdx]['paddingPath'] = paddingPath
            
        smoother.addDetailPath(
            groupIdx=agentInfos[agentIdx]['groupIdx'], 
            pathIdx=agentIdx,
            radius=agentInfos[agentIdx]['radius'],
            detailPath=agentInfos[agentIdx]['paddingPath']
        )

        oldPaths[agentIdx] = np.array(agentInfos[agentIdx]['paddingPath'])[:, :3]

    ### ---------- Just For Debug
    newPaths = {}
    for _ in range(3):
        smoother.smoothPath(updateTimes=30)

        ### -------- vis debug
        ax = vis_utils.create_Graph3D(
            xmax=cond_params['x'] + 4.0, 
            ymax=cond_params['y'] + 4.0, 
            zmax=cond_params['z'] + 4.0
        )

        for groupIdx in smoother.groupMap.keys():
            groupInfo = smoother.groupMap[groupIdx]
            groupPath = groupInfo.path

            for agentIdx in groupPath.pathIdxs_set:
                pathXYZR = np.array(groupPath.extractPath(agentIdx))
                newPaths[agentIdx] = pathXYZR[:, :3]

        for agentIdx in oldPaths.keys():
            vis_utils.plot_Path3D(ax, newPaths[agentIdx][:, :3], color='r')
            vis_utils.plot_Path3D(ax, oldPaths[agentIdx][:, :3], color='b')
        
        plt.show()

        oldPaths = newPaths
    ### --------------------------------------------------------
    # smoother.smoothPath(updateTimes=50)

    for agentIdx in agentInfos.keys():
        groupIdx = agentInfos[agentIdx]['groupIdx']
        groupInfo = smoother.groupMap[groupIdx]
        groupPath = groupInfo.path

        pathXYZR = groupPath.extractPath(agentIdx)
        agentInfos[agentIdx]['smoothXYZR'] = np.array(pathXYZR)

    random_colors = np.random.uniform(0.0, 1.0, size=(cond_params['num_of_groups'], 3))
    vis = VisulizerVista()
    for agentIdx in agentInfos.keys():
        mainXYZ = agentInfos[agentIdx]['smoothXYZR'][:, :3]

        startDire = np.array(agentInfos[agentIdx]['start_paddingDire'])
        startXYZ = np.array([
            mainXYZ[0] + startDire * 5.,
            mainXYZ[0] + startDire * 4.,
            mainXYZ[0] + startDire * 3.,
            mainXYZ[0] + startDire * 2.,
            mainXYZ[0] + startDire * 1.,
        ])
        endDire = np.array(agentInfos[agentIdx]['end_paddingDire'])
        endXYZ = np.array([
            mainXYZ[-1] + endDire * 1.,
            mainXYZ[-1] + endDire * 2.,
            mainXYZ[-1] + endDire * 3.,
            mainXYZ[-1] + endDire * 4.,
            mainXYZ[-1] + endDire * 5.,
        ])

        mainXYZ = np.concatenate([startXYZ, mainXYZ, endXYZ], axis=0)

        tube_mesh = vis.create_tube(
            mainXYZ, 
            radius=agentInfos[agentIdx]['radius']
        )
        vis.plot(tube_mesh, color=tuple(random_colors[agentInfos[agentIdx]['groupIdx']]))
    vis.show()

if __name__ == '__main__':
    # planner_find_Path()
    smoothPath()
