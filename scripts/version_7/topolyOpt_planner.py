import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn.neighbors import KDTree
from typing import Dict
import os
import shutil
import math

import pyvista
import pymeshfix

from scripts.visulizer import VisulizerVista
from scripts.version_7.mesh_helper import MeshHelper

class TopolyOpt_Helper(object):
    def __init__(self):
        self.meshHelper = MeshHelper()

    def expansion_mesh(self, resPaths:Dict, expansionDist, env_config, outputDir):
        obs_df = pd.read_csv(env_config['static_grid_obs_pcd'], index_col=0)
        wall_obs_df = pd.read_csv(env_config['wall_obs_pcd'], index_col=0)
        obs_df = pd.concat([obs_df, wall_obs_df], axis=0, ignore_index=True)

        obs_tree = KDTree(obs_df[['x', 'y', 'z']].values)
        
        groupIdxs = resPaths.keys()
        # groupIdxs = [0]

        ### ------ create path tree
        groupPaths = {}
        for groupKey in groupIdxs:
            groupInfo = {'paths': {}}

            path_xyz = []
            for pathIdx in resPaths[groupKey].keys():
                path_xyz.append(resPaths[groupKey][pathIdx]['path_xyzr'][:, :3])
                groupInfo['paths'][pathIdx] = resPaths[groupKey][pathIdx]

            group_xyzs = np.concatenate(path_xyz, axis=0)
            groupInfo['tree_xyz'] = KDTree(group_xyzs)
            groupInfo['group_xyzs'] = group_xyzs

            groupPaths[groupKey] = groupInfo

        # self.compute_groupTubeStl(groupPaths, obs_tree, expansionDist, outputDir)
        self.compute_groupBoxStl(
            groupPaths, obs_tree, expansionDist,
            xlims=[0, env_config['x'] - 1],
            ylims=[0, env_config['y'] - 1],
            zlims=[0, env_config['z'] - 1],
        )

    def compute_groupBoxStl(self, groupPaths, obs_tree, expansionDist, xlims, ylims, zlims):
        groupIdxs = groupPaths.keys()

        for groupKey in groupIdxs:
            groupInfo = groupPaths[groupKey]

            xyzs = []
            for (x, y, z) in groupInfo['group_xyzs']:
                xs = np.arange(
                    np.maximum(np.minimum(x - expansionDist, xlims[1]), xlims[0]), 
                    np.maximum(np.minimum(x + expansionDist, xlims[1]), xlims[0]),
                    1.0
                )
                ys = np.arange(
                    np.maximum(np.minimum(y - expansionDist, ylims[1]), ylims[0]), 
                    np.maximum(np.minimum(y + expansionDist, ylims[1]), ylims[0]),
                    1.0
                )
                zs = np.arange(
                    np.maximum(np.minimum(z - expansionDist, zlims[1]), zlims[0]), 
                    np.maximum(np.minimum(z + expansionDist, zlims[1]), zlims[0]),
                    1.0
                )

                xs, ys, zs = np.meshgrid(xs, ys, zs)
                cell_xyzs = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1)
                cell_xyzs = cell_xyzs.reshape((-1, 3))
                xyzs.append(cell_xyzs)
            
            xyzs = np.concatenate(xyzs, axis=0)
            xyzs = pd.DataFrame(xyzs).drop_duplicates().values

            obs_dists = obs_tree.query(xyzs, k=1, return_distance=True)
            xyzs = xyzs[obs_dists > 0.55]

            select_bool = np.ones((xyzs.shape[0], ))
            ref_dists = groupInfo['tree_xyz'].query(xyzs, k=1, return_distance=True).reshape(-1)
            for sub_groupKey in groupIdxs:
                if groupKey == sub_groupKey:
                    continue

                sub_groupInfo = groupPaths[sub_groupKey]
                pipe_dists = sub_groupInfo['tree_xyz'].query(xyzs, k=1, return_distance=True).reshape(-1)

                select_bool[pipe_dists < ref_dists] = 0.0

            xyzs = xyzs[select_bool==1.0]

            print(xyzs.shape)

    def compute_groupTubeStl(self, groupPaths, obs_tree, expansionDist, outputDir):
        groupIdxs = groupPaths.keys()

        for groupKey in groupIdxs:
            groupInfo = groupPaths[groupKey]

            for pathIdx in groupInfo['paths'].keys():
                pathInfo = groupInfo['paths'][pathIdx]
                
                pathRadius, index = obs_tree.query(pathInfo['path_xyzr'][1:-1, :3], k=1, return_distance=True)
                pathRadius = np.minimum(pathRadius, expansionDist)

                for sub_groupKey in groupIdxs:
                    if groupKey == sub_groupKey:
                        continue
                    sub_groupInfo = groupPaths[sub_groupKey]

                    dist, index = sub_groupInfo['tree_xyz'].query(
                        pathInfo['path_xyzr'][1:-1, :3], k=1, return_distance=True
                    )
                    dist = dist / 2.0
                    pathRadius = np.minimum(pathRadius, dist)
                
                pathRadius = np.concatenate([
                    np.array([[pathInfo['grid_radius']]]),
                    pathRadius,
                    np.array([[pathInfo['grid_radius']]])
                ], axis=0)
                pathInfo['expandRadius'] = pathRadius

        groupSetting = {}
        for idx, groupKey in enumerate(groupIdxs):
            groupInfo = groupPaths[groupKey]
            groupSetting[groupKey] = {}

            group_dir = os.path.join(outputDir, 'group_%d'%groupKey)
            if os.path.exists(group_dir):
                shutil.rmtree(group_dir)
            os.mkdir(group_dir)

            for pathIdx in groupInfo['paths'].keys():
                settingDict = {}

                pathInfo = groupInfo['paths'][pathIdx]
                path_xyz = pathInfo['path_xyzr'][:, :3]
                
                startVec = self.polar2vec(pathInfo['startDire'])
                startlet = path_xyz[0, :] - startVec

                endVec = self.polar2vec(pathInfo['endDire'])
                endlet = path_xyz[-1, :] + endVec

                path_xyz = np.concatenate([
                    startlet.reshape((1, -1)),
                    path_xyz,
                    endlet.reshape((1, -1))
                ], axis=0)
                expandRadius = np.concatenate([
                    np.array([[pathInfo['path_xyzr'][0, -1]]]),
                    pathInfo['expandRadius'],
                    np.array([[pathInfo['path_xyzr'][-1, -1]]])
                ], axis=0)

                tube_mesh = VisulizerVista.create_complex_tube(
                    path_xyz, radius=None, capping=True, scalars=expandRadius
                )

                save_path = os.path.join(group_dir, 'path_%d.stl'%pathIdx)
                tube_mesh.save(save_path, binary=True)
                # pyvista.save_meshio(save_path, tube_mesh)

                ### TODO for toposet
                settingDict['startPoint'] = [path_xyz[1, 0], path_xyz[1, 1], path_xyz[1, 2]]
                settingDict['pathStartPoint'] = [path_xyz[0, 0], path_xyz[0, 1], path_xyz[0, 2]]
                settingDict['startDire'] = pathInfo['startDire']
                settingDict['endPoint'] = [path_xyz[-2, 0], path_xyz[-2, 1], path_xyz[-2, 2]]
                settingDict['pathEndPoint'] = [path_xyz[-1, 0], path_xyz[-1, 1], path_xyz[-1, 2]]
                settingDict['endDire'] = pathInfo['endDire']
                settingDict['startRadius'] = pathInfo['path_xyzr'][1, -1]
                settingDict['endRadius'] = pathInfo['path_xyzr'][-2, -1]
                settingDict['STL_path'] = save_path

                inside_point = path_xyz[int(path_xyz.shape[0] * 0.5), :3]
                settingDict['inside_point']=[inside_point[0], inside_point[1], inside_point[2]]

                groupSetting[groupKey][pathIdx] = settingDict

        with open(os.path.join(outputDir, 'groupSetting.json'), 'w') as f:
            json.dump(groupSetting, f)

    def polar2vec(self, polarVec, length=1.0):
        dz = length * math.sin(polarVec[1])
        dl = length * math.cos(polarVec[1])
        dx = dl * math.cos(polarVec[0])
        dy = dl * math.sin(polarVec[0])
        return np.array([dx, dy, dz])

    def debugSearchField(self, groupPaths, idx=None):
        vis = VisulizerVista()

        if idx is None:
            groupIdxs = groupPaths.keys()
        else:
            groupIdxs = [idx]

        random_colors = np.random.uniform(0.0, 1.0, size=(len(groupIdxs), 3))
        for idx, groupKey in enumerate(groupIdxs):
            groupInfo = groupPaths[groupKey]

            for pathIdx in groupInfo['paths'].keys():
                pathInfo = groupInfo['paths'][pathIdx]
                stl_path = pathInfo['STL_path']

                mesh = VisulizerVista.read_file(stl_path)
                vis.plot(mesh, color=random_colors[idx], opacity=1.0, style='wireframe')

        vis.show()

def main():
    grid_json_file = '/home/quan/Desktop/MAPF_Pipeline/scripts/version_7/app_dir/grid_env_cfg.json'
    with open(grid_json_file, 'r') as f:
        env_config = json.load(f)
    
    resPaths = np.load(
        '/home/quan/Desktop/MAPF_Pipeline/scripts/version_7/app_dir/resPath_config.npy', allow_pickle=True
    ).item()

    case_dir = '/home/quan/Desktop/tempary/openFoam_application/case0'

    optimizer = TopolyOpt_Helper()
    optimizer.expansion_mesh(
        resPaths=resPaths,
        expansionDist=5.0, 
        env_config=env_config,
        outputDir=case_dir
    )

if __name__ == '__main__':
    main()    

