import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn.neighbors import KDTree
from typing import Dict
import pyvista

from scripts.visulizer import VisulizerVista
from scripts.version_7.mesh_helper import MeshHelper
from build import mapf_pipeline

class TopolyOpt_Helper(object):
    def __init__(self):
        self.meshHelper = MeshHelper()

    def expansion_mesh(self, resPaths:Dict, expansionDist, env_config):
        obs_df = pd.read_csv(env_config['static_grid_obs_pcd'], index_col=0)
        wall_obs_df = pd.read_csv(env_config['wall_obs_pcd'], index_col=0)
        obs_df = pd.concat([obs_df, wall_obs_df], axis=0, ignore_index=True)

        obs_tree = KDTree(obs_df[['x', 'y', 'z']].values)
        xmin, xmax = 0, env_config['x'] - 1
        ymin, ymax = 0, env_config['y'] - 1
        zmin, zmax = 0, env_config['z'] - 1

        groupCells = {}
        for groupKey in resPaths.keys():
            groupCells[groupKey] = {}

            pathCells = []
            for pathIdx in resPaths[groupKey].keys():
                pathCells.append(resPaths[groupKey][pathIdx]['path_xyzr'][:, :3])
            
            path_xyz = np.concatenate(pathCells, axis=0)
            groupCells[groupKey]['path_xyz'] = path_xyz
            groupCells[groupKey]['tree'] = KDTree(groupCells[groupKey]['path_xyz'])
            groupCells[groupKey]['orig_path'] = resPaths[groupKey]

        print('Start Extract Box Area ...')
        for groupKey in tqdm(groupCells.keys()):
            search_info = groupCells[groupKey]
            searchTree = search_info['tree']
            search_info['expandCells'] = []

            searchCells = self.get_searchRange(
                search_info['path_xyz'], expansionDist, xmin, xmax, ymin, ymax, zmin, zmax
            )

            for (x, y, z) in searchCells:
                pos = np.array([[x, y, z]])

                obs_num = obs_tree.query_radius(pos, r=0.55, count_only=True)
                if obs_num > 0:
                    continue

                dist, _ = searchTree.query(pos, k=1)
                if dist > expansionDist:
                    continue

                cloest_to_current_group = True
                for sub_groupKey in groupCells.keys():
                    if sub_groupKey == groupKey:
                        continue
                        
                    sub_dist, _ = groupCells[sub_groupKey]['tree'].query(pos, k=1)
                    if sub_dist < dist:
                        cloest_to_current_group = False
                        break
                
                if cloest_to_current_group:
                    search_info['expandCells'].append([x, y, z])
        
        # for groupKey in groupCells.keys():
        #     self.debugSingleSearchCell(groupCells[groupKey])
        
        # self.debugWholeSearchCell(groupCells, obs_df)

        print('Start Union Meshes ...')
        for groupKey in groupCells.keys():
            search_info = groupCells[groupKey]
            expandCells = search_info['expandCells']

            boxes_mesh = VisulizerVista.create_many_boxs(expandCells, length=1.0)
            search_info['search_mesh'] = boxes_mesh.combine()
        
        self.debugWholeSearchMesh(groupCells, obs_df)

    def get_searchRange(self, path_xyz:np.array, expandDist, xmin, xmax, ymin, ymax, zmin, zmax):
        xyzs = []
        for (x, y, z) in path_xyz:
            xs = np.arange(int(x - expandDist), int(x + expandDist), 1)
            ys = np.arange(int(y - expandDist), int(y + expandDist), 1)
            zs = np.arange(int(z - expandDist), int(z + expandDist), 1)

            xs, ys, zs = np.meshgrid(xs, ys, zs)
            cell_xyzs = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1)
            cell_xyzs = cell_xyzs.reshape((-1, 3))

            xyzs.append(cell_xyzs)
        
        xyzs = np.concatenate(xyzs, axis=0)
        xyzs = pd.DataFrame(xyzs, columns=['x', 'y', 'z'])
        
        xyzs = xyzs[(xyzs['x'] >= xmin) & (xyzs['x'] <= xmax)]
        xyzs = xyzs[(xyzs['y'] >= ymin) & (xyzs['y'] <= ymax)]
        xyzs = xyzs[(xyzs['z'] >= zmin) & (xyzs['z'] <= zmax)]
        xyzs.drop_duplicates(inplace=True)

        return xyzs.values

    def debugSingleSearchCell(self, searchInfo):
        vis = VisulizerVista()

        for pathIdx in searchInfo['orig_path'].keys():
            path_info = searchInfo['orig_path'][pathIdx]
            path_xyzr = path_info['path_xyzr']
            tube_mesh = vis.create_tube(path_xyzr[:, :3], radius=0.5)
            vis.plot(tube_mesh, color=(1.0, 0.0, 0.0), opacity=1.0)

        # tube_mesh = vis.create_tube(searchInfo['path_xyz'], radius=0.5)
        boxes_mesh = vis.create_many_boxs(searchInfo['expandCells'])

        # vis.plot(tube_mesh, color=(1.0, 0.0, 0.0), opacity=1.0)
        vis.plot(boxes_mesh, color=(0.0, 1.0, 0.0), opacity=0.3)

        vis.show()

    def debugWholeSearchCell(self, groupCells, obs_df:pd.DataFrame):
        vis = VisulizerVista()
        
        pipe_randomColors = np.random.uniform(low=0.0, high=1.0, size=(len(groupCells), 3))
        field_randomColors = np.random.uniform(low=0.0, high=1.0, size=(len(groupCells), 3))

        for groupKey in groupCells.keys():
            searchInfo = groupCells[groupKey]

            for pathIdx in searchInfo['orig_path'].keys():
                path_info = searchInfo['orig_path'][pathIdx]
                path_xyzr = path_info['path_xyzr']
                tube_mesh = vis.create_tube(path_xyzr[:, :3], radius=path_info['grid_radius'])
                vis.plot(tube_mesh, color=pipe_randomColors[groupKey], opacity=1.0)

            # tube_mesh = vis.create_tube(searchInfo['path_xyz'], radius=0.5)
            boxes_mesh = vis.create_many_boxs(searchInfo['expandCells'])

            # vis.plot(tube_mesh, color=pipe_randomColors[groupKey], opacity=1.0)
            vis.plot(boxes_mesh, color=field_randomColors[groupKey], opacity=0.25)

        obstacle_mesh = vis.create_pointCloud(obs_df[['x', 'y', 'z']].values)
        vis.plot(obstacle_mesh, (1.0, 0.5, 0.25), opacity=0.8)

        vis.show()

    def debugWholeSearchMesh(self, groupCells, obs_df:pd.DataFrame):
        vis = VisulizerVista()
        
        pipe_randomColors = np.random.uniform(low=0.0, high=1.0, size=(len(groupCells), 3))
        field_randomColors = np.random.uniform(low=0.0, high=1.0, size=(len(groupCells), 3))

        for groupKey in groupCells.keys():
            searchInfo = groupCells[groupKey]

            for pathIdx in searchInfo['orig_path'].keys():
                path_info = searchInfo['orig_path'][pathIdx]
                path_xyzr = path_info['path_xyzr']
                tube_mesh = vis.create_tube(path_xyzr[:, :3], radius=path_info['grid_radius'])
                vis.plot(tube_mesh, color=pipe_randomColors[groupKey], opacity=1.0)

            vis.plot(searchInfo['search_mesh'], color=field_randomColors[groupKey], opacity=0.25)

        obstacle_mesh = vis.create_pointCloud(obs_df[['x', 'y', 'z']].values)
        vis.plot(obstacle_mesh, (1.0, 0.5, 0.25), opacity=0.8)

        vis.show()

def main():
    grid_json_file = '/home/quan/Desktop/MAPF_Pipeline/scripts/version_7/app_dir/grid_env_cfg.json'
    with open(grid_json_file, 'r') as f:
        env_config = json.load(f)
    
    resPaths = np.load(
        '/home/quan/Desktop/MAPF_Pipeline/scripts/version_7/app_dir/resPath_config.npy', allow_pickle=True
    ).item()

    optimizer = TopolyOpt_Helper()
    optimizer.expansion_mesh(
        resPaths=resPaths,
        expansionDist=5.0, 
        env_config=env_config
    )

if __name__ == '__main__':
    main()    

