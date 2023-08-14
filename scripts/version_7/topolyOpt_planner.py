import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn.neighbors import KDTree
from typing import Dict
import os
import shutil
import math

from tempfile import TemporaryDirectory
import pymeshlab
import pyvista
import trimesh
from pymeshfix import MeshFix

from scripts.visulizer import VisulizerVista
# from scripts.version_7.mesh_helper import MeshHelper

class TopolyOpt_Helper(object):
    def expansion_mesh(self, resPaths:Dict, expansionDist, env_config, outputDir):
        obs_json_file = '/home/quan/Desktop/tempary/application_pipe/condObs.json'
        with open(obs_json_file, 'r') as f:
            self.obs_config = json.load(f)

        self.obs_df = pd.read_csv(env_config['obstacleSavePath'], index_col=0)
        obs_xyzs = self.obs_df[['x', 'y', 'z']].values

        # self.compute_groupTubeStl(resPaths, obs_xyzs, expansionDist, outputDir)
        # self.compute_groupBoxStl(
        #     env_config,
        #     resPaths, obs_xyzs, expansionDist,
        #     xlims=[0, env_config['grid_x'] - 1],
        #     ylims=[0, env_config['grid_y'] - 1],
        #     zlims=[0, env_config['grid_z'] - 1],
        #     outputDir=outputDir
        # )
        
        # for groupTag in os.listdir(outputDir):
        #     groupDir = os.path.join(outputDir, groupTag)
        #     self.adjustDesignSpaceInOutlet(env_config, groupDir)

        # self.debugSearchPathTubeField(groupKeys=resPaths.keys(), saveDir=outputDir)
        self.vis_DesignSpace(groupKeys=resPaths.keys(), saveDir=outputDir, obs_config=self.obs_config)

    def compute_groupBoxStl(self, env_config, resPaths, obs_xyzs, expansionDist, xlims, ylims, zlims, outputDir):
        groupIdxs = resPaths.keys()
        # groupIdxs = [4]

        ### ------ Create Path Tree
        groupPaths = {}
        for groupKey in groupIdxs:
            groupInfo = {'paths': {}}

            path_xyz = []
            for pathIdx in resPaths[groupKey].keys():
                path_xyz.append(resPaths[groupKey][pathIdx]['path_xyzr'][:, :3])
                groupInfo['paths'][pathIdx] = resPaths[groupKey][pathIdx]

            group_xyzs = np.concatenate(path_xyz, axis=0)
            groupInfo['path_tree'] = KDTree(group_xyzs.copy())
            groupInfo['group_xyzs'] = group_xyzs
            groupPaths[groupKey] = groupInfo
        
        scale = 1.0
        ### ------ Create Box
        for groupKey in groupIdxs:
            group_dir = os.path.join(outputDir, 'group_%d' % groupKey)
            if os.path.exists(group_dir):
                shutil.rmtree(group_dir)
            os.mkdir(group_dir)

            groupInfo = groupPaths[groupKey]
            xyzs = self.create_outLayerBoxMesh(
                groupInfo['group_xyzs'], obs_xyzs, expansionDist, xlims, ylims, zlims, scale=scale
            )
            # ploter = pyvista.Plotter()
            # # ploter.add_mesh(pyvista.PointSet(obs_xyzs), color=(1.0, 0.0, 0.0))
            # ploter.add_mesh(pyvista.PointSet(xyzs), color=(0.0, 1.0, 0.0))
            # ploter.show()
            # mesh = self.createBoxUnionMesh(xyzs, scale=scale)
            # mesh.plot()

            select_bool = np.ones((xyzs.shape[0],), dtype=bool)
            ref_dists, _ = groupInfo['path_tree'].query(xyzs, k=1, return_distance=True)
            ref_dists = ref_dists.reshape(-1)
            for sub_groupKey in groupIdxs:
                if groupKey == sub_groupKey:
                    continue
                sub_groupInfo = groupPaths[sub_groupKey]
                pipe_dists, _ = sub_groupInfo['path_tree'].query(xyzs, k=1, return_distance=True)
                pipe_dists = pipe_dists.reshape(-1)
                select_bool[ref_dists >= pipe_dists] = False
            xyzs = xyzs[select_bool]

            ### ------ record remove ends
            removeEnd = {}
            groupSetting = {'path':{}}
            for pathIdx in groupInfo['paths'].keys():
                pathInfo = groupInfo['paths'][pathIdx]
                path_xyz = pathInfo['path_xyzr'][:, :3]
                path_mesh = VisulizerVista.create_tube(pathInfo['path_xyzr'][:, :3], radius=pathInfo['grid_radius'])

                path_mesh_file = os.path.join(group_dir, 'path_%d.stl'%pathIdx)
                path_mesh.save(path_mesh_file, binary=True)

                pathInfo['path_stl'] = path_mesh_file
                groupSetting['path'][pathIdx] = pathInfo

                startTag = f'{path_xyz[0, 0]}/{path_xyz[0, 1]}/{path_xyz[0, 2]}'
                if startTag not in removeEnd.keys():
                    removeEnd[startTag] = {
                        'pose': path_xyz[0, :], 'vec': self.polar2vec(pathInfo['startDire']), 
                        'radius': pathInfo['grid_radius'], 'type': 'start',
                    }
                
                endTag = f'{path_xyz[-1, 0]}/{path_xyz[-1, 1]}/{path_xyz[-1, 2]}'
                if endTag not in removeEnd.keys():
                    removeEnd[endTag] = {
                        'pose': path_xyz[-1, :], 'vec': self.polar2vec(pathInfo['endDire']), 
                        'radius': pathInfo['grid_radius'], 'type': 'end',
                    }

            # plter = pyvista.Plotter()
            xyzs_pcd = pyvista.PointSet(xyzs)
            select_bool = np.ones(xyzs.shape[0]).astype(bool)
            for tag in removeEnd.keys():
                tagInfo = removeEnd[tag]
                if tagInfo['type'] == 'start':
                    pose0 = tagInfo['pose'] - 2.0 * tagInfo['vec']
                    pose1 = tagInfo['pose'] + 0.2 * tagInfo['vec']
                else:
                    pose0 = tagInfo['pose'] - 0.2 * tagInfo['vec']
                    pose1 = tagInfo['pose'] + 2.0 * tagInfo['vec']

                cylinderArea = pyvista.Cylinder(
                    center = (pose0 + pose1) * 0.5,
                    direction = tagInfo['vec'],
                    radius = tagInfo['radius'] + 1.0,
                    height = 2.2
                )
                
                isInside = xyzs_pcd.select_enclosed_points(cylinderArea, check_surface=False)['SelectedPoints'].astype(bool)
                select_bool[isInside] = False

            #     plter.add_mesh(cylinderArea, opacity=0.85)
            # plter.add_mesh(xyzs_pcd)
            # plter.show()

            xyzs = xyzs_pcd.points[select_bool]
            mesh = self.createBoxUnionMesh(xyzs, scale=scale)
            # mesh.plot()

            design_mesh_path = os.path.join(group_dir, 'designSpace.stl')
            mesh.save(design_mesh_path, binary=True)
            groupSetting['designSpace_stl'] = design_mesh_path
            np.save(os.path.join(group_dir, 'groupSetting.npy'), groupSetting)

    def adjustDesignSpaceInOutlet(self, env_config, groupDir):
        groupSetting = np.load(os.path.join(groupDir, 'groupSetting.npy'), allow_pickle=True).item()

        with TemporaryDirectory() as tempDir:
            concat_InOutlets = {}
            info_InOutlets = {
                'inOutlet': {}
            }

            pathInfos = groupSetting['path']
            for pathIdx in pathInfos.keys():
                pathInfo = pathInfos[pathIdx]

                startPoint = pathInfo['path_xyzr'][0, :3]
                startTag = f'{startPoint[0]}/{startPoint[1]}/{startPoint[2]}'
                if startTag not in concat_InOutlets.keys():
                    vec = self.polar2vec(pathInfo['startDire'])
                    pose0 = startPoint
                    pose1 = startPoint + 0.6 * vec

                    inlet = pyvista.Cylinder(
                        center = (pose0 + pose1) * 0.5,
                        direction = vec,
                        radius = pathInfo['grid_radius'],
                        height = 0.6,
                        resolution=25
                    )

                    inlet_idx = len(concat_InOutlets)
                    inlet_save_path = os.path.join(tempDir, '%d.stl' % inlet_idx)
                    inlet.save(inlet_save_path)
                    concat_InOutlets[startTag] = inlet_save_path

                    settingDict = {}
                    settingDict['idx'] = inlet_idx
                    settingDict['type'] = 'inlet'
                    settingDict['pose'] = (startPoint[0], startPoint[1], startPoint[2])
                    settingDict['vec'] = (vec[0], vec[1], vec[2])
                    settingDict['radius'] = pathInfo['grid_radius']
                    info_InOutlets['inOutlet'][inlet_idx] = settingDict
                
                endPoint = pathInfo['path_xyzr'][-1, :3]
                endTag = f'{endPoint[0]}/{endPoint[1]}/{endPoint[2]}'
                if endTag not in concat_InOutlets.keys():
                    vec = self.polar2vec(pathInfo['endDire'])
                    pose0 = endPoint - 0.6 * vec
                    pose1 = endPoint

                    outlet = pyvista.Cylinder(
                        center = (pose0 + pose1) * 0.5,
                        direction = vec,
                        radius = pathInfo['grid_radius'],
                        height = 0.6,
                        resolution = 25
                    )

                    outlet_idx = len(concat_InOutlets)
                    outlet_save_path = os.path.join(tempDir, '%d.stl' % outlet_idx)
                    outlet.save(outlet_save_path)
                    concat_InOutlets[endTag] = outlet_save_path

                    settingDict = {}
                    settingDict['idx'] = outlet_idx
                    settingDict['type'] = 'outlet'
                    settingDict['pose'] = (endPoint[0], endPoint[1], endPoint[2])
                    settingDict['vec'] = (vec[0], vec[1], vec[2])
                    settingDict['radius'] = pathInfo['grid_radius']
                    info_InOutlets['inOutlet'][outlet_idx] = settingDict

            # ploter = pyvista.Plotter()
            # for tag in concat_InOutlets.keys():
            #     sub_mesh = pyvista.STLReader(concat_InOutlets[tag]).read()
            #     ploter.add_mesh(sub_mesh, style='surface', opacity=1.0)
            # ploter.add_mesh(pyvista.STLReader(groupSetting['designSpace_stl']).read())
            # ploter.show()

            ### ------ merge InOutlets
            # # ploter = pyvista.Plotter()
            # designMesh = pyvista.STLReader(groupSetting['designSpace_stl']).read()
            # designMesh.triangulate(inplace=True)
            # for i, tag in enumerate(concat_InOutlets.keys()):
            #     path = concat_InOutlets[tag]
            #     mesh = pyvista.STLReader(path).read()
            #     mesh.triangulate(inplace=True)
                
            #     ### TODO fuck why is not union
            #     # designMesh = designMesh.boolean_union(mesh)
            #     designMesh = designMesh.boolean_difference(mesh)
            #     designMesh.triangulate(inplace=True)
            #     # designMesh.plot()

            # designMesh_path = os.path.join(groupDir, 'designMesh.stl')
            # designMesh.save(designMesh_path, binary=True)

            ### TODO I really do not understand why
            paths = list(concat_InOutlets.values())
            paths.insert(0, groupSetting['designSpace_stl'])
            designMesh_path = os.path.join(groupDir, 'designMesh.stl')
            self.blender_union(paths, designMesh_path)
            # pyvista.STLReader(designMesh_path).read().plot()
            
            ### -------------------------------------------------------------
            designMesh = pyvista.STLReader(designMesh_path).read()

            isWaterTight = self.checkIsWaterTight(designMesh_path)
            if not isWaterTight:
                meshfix = MeshFix(designMesh)
                holes = meshfix.extract_holes()
                plotter = pyvista.Plotter()
                plotter.add_mesh(designMesh, color=True, show_edges=True)
                plotter.add_mesh(holes, color="r", line_width=5)
                plotter.enable_eye_dome_lighting()  # helps depth perception
                plotter.show()

            info_InOutlets['designMesh_stl'] = designMesh_path
            xmin, xmax, ymin, ymax, zmin, zmax = designMesh.bounds
            info_InOutlets['xmin'] = xmin
            info_InOutlets['xmax'] = xmax
            info_InOutlets['ymin'] = ymin
            info_InOutlets['ymax'] = ymax
            info_InOutlets['zmin'] = zmin
            info_InOutlets['zmax'] = zmax

            pathInfos = groupSetting['path']
            for pathIdx in pathInfos.keys():
                pathInfo = pathInfos[pathIdx]
                path_xyzs = pathInfo['path_xyzr'][:, :3]
                inside_point = path_xyzs[int(path_xyzs.shape[0] * 0.5), :]
                info_InOutlets['inside_pose'] = [inside_point[0], inside_point[1], inside_point[2]]
                break
            
            with open(os.path.join(groupDir, 'inOutletSetting.json'), 'w') as f:
                json.dump(info_InOutlets, f)
                
    def compute_groupTubeStl(self, resPaths, obs_xyzs, expansionDist, outputDir):
        obs_tree = KDTree(obs_xyzs)
        groupIdxs = resPaths.keys()

        groupPaths = {}
        for groupKey in groupIdxs:
            groupInfo = {'paths': {}}

            path_xyz = []
            for pathIdx in resPaths[groupKey].keys():
                path_xyz.append(resPaths[groupKey][pathIdx]['path_xyzr'][:, :3])
                groupInfo['paths'][pathIdx] = resPaths[groupKey][pathIdx]

            group_xyzs = np.concatenate(path_xyz, axis=0)
            groupInfo['path_tree'] = KDTree(group_xyzs)
            groupInfo['group_xyzs'] = group_xyzs

            groupPaths[groupKey] = groupInfo

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

                    dist, index = sub_groupInfo['path_tree'].query(
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

        ### ------ Output Result
        for idx, groupKey in tqdm(enumerate(groupIdxs)):
            groupInfo = groupPaths[groupKey]
            groupSetting = {}

            group_dir = os.path.join(outputDir, 'group_%d' % groupKey)
            if os.path.exists(group_dir):
                shutil.rmtree(group_dir)
            os.mkdir(group_dir)

            save_mesh_paths = []
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
                    np.array([[pathInfo['grid_radius']]]),
                    pathInfo['expandRadius'],
                    np.array([[pathInfo['grid_radius']]])
                ], axis=0)

                tube_mesh = VisulizerVista.create_complex_tube(
                    path_xyz, radius=None, capping=True, scalars=expandRadius
                )

                save_path = os.path.join(group_dir, 'path_%d.stl'%pathIdx)
                tube_mesh.save(save_path, binary=True)
                # pyvista.save_meshio(save_path, tube_mesh)
                save_mesh_paths.append(save_path)

                ### TODO for toposet
                settingDict['startPoint'] = [path_xyz[1, 0], path_xyz[1, 1], path_xyz[1, 2]]
                settingDict['pathStartPoint'] = [path_xyz[0, 0], path_xyz[0, 1], path_xyz[0, 2]]
                settingDict['startDire'] = pathInfo['startDire']
                settingDict['endPoint'] = [path_xyz[-2, 0], path_xyz[-2, 1], path_xyz[-2, 2]]
                settingDict['pathEndPoint'] = [path_xyz[-1, 0], path_xyz[-1, 1], path_xyz[-1, 2]]
                settingDict['endDire'] = pathInfo['endDire']
                settingDict['startRadius'] = pathInfo['grid_radius']
                settingDict['endRadius'] = pathInfo['grid_radius']
                settingDict['STL_path'] = save_path

                inside_point = path_xyz[int(path_xyz.shape[0] * 0.5), :3]
                settingDict['inside_point']=[inside_point[0], inside_point[1], inside_point[2]]

                groupSetting[pathIdx] = settingDict

            with open(os.path.join(group_dir, 'groupSetting.json'), 'w') as f:
                json.dump(groupSetting, f)

            self.meshUnion_pymeshlab(save_mesh_paths, os.path.join(group_dir, 'path_merge.stl'))

    def polar2vec(self, polarVec, length=1.0):
        dz = length * math.sin(polarVec[1])
        dl = length * math.cos(polarVec[1])
        dx = dl * math.cos(polarVec[0])
        dy = dl * math.sin(polarVec[0])
        return np.array([dx, dy, dz])

    def meshUnion_pymeshlab(self, meshPaths, outputFile):
        for i in range(1, len(meshPaths)):
            if i == 1:
                file0 = meshPaths[i-1]
            else:
                file0 = outputFile
            file1 = meshPaths[i]

            print(file0)
            print(file1)

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(file0)
            ms.load_new_mesh(file1)
            
            ms.generate_boolean_union()
            ms.save_current_mesh(outputFile)
    
    def meshIntersection_pymeshlab(self, meshPaths, outputFile):
        for i in range(1, len(meshPaths)):
            if i == 1:
                file0 = meshPaths[i-1]
            else:
                file0 = outputFile
            file1 = meshPaths[i]

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(file0)
            ms.load_new_mesh(file1)
            
            ms.generate_boolean_intersection()
            ms.save_current_mesh(outputFile)
    
    def meshDifference_pymeshlab(self, file_to_remove, file_to_restore, outputFile):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file_to_remove)
        ms.load_new_mesh(file_to_restore)
        ms.generate_boolean_difference()
        ms.save_current_mesh(outputFile)

    def blender_union(self, meshPaths, outputFile):
        mesh_lists = []
        for path in meshPaths:
            mesh_lists.append(trimesh.load_mesh(path))
        mesh_union = trimesh.boolean.union(mesh_lists, engine='blender')
        mesh_union.export(outputFile)

    def create_outLayerBoxMesh(self, xyzs, obs_xyzs, expansionDist, xlims, ylims, zlims, scale = 1.0):
        obs_tree = KDTree(obs_xyzs)
        # reso = np.sqrt((np.power(np.array([0.5, 0.5, 0.5]), 2).sum()))
        reso = scale

        xyzs = pd.DataFrame(xyzs.astype(int), columns=['x', 'y', 'z']).drop_duplicates().values
        for _ in range(expansionDist):
            xyzs = np.concatenate([
                xyzs,
                xyzs + np.array([1.0, 0.0, 0.0]) * scale,
                xyzs + np.array([-1.0, 0.0, 0.0]) * scale,
                xyzs + np.array([0.0, 1.0, 0.0]) * scale,
                xyzs + np.array([0.0, -1.0, 0.0]) * scale,
                xyzs + np.array([0.0, 0.0, 1.0]) * scale,
                xyzs + np.array([0.0, 0.0, -1.0]) * scale,

                ### watertight means every 2 cells can not only connect by 1 edge
                xyzs + np.array([1.0, 1.0, 0.0]) * scale,
                xyzs + np.array([1.0, -1.0, 0.0]) * scale,
                xyzs + np.array([1.0, 0.0, 1.0]) * scale,
                xyzs + np.array([1.0, 0.0, -1.0]) * scale,
                xyzs + np.array([-1.0, 1.0, 0.0]) * scale,
                xyzs + np.array([-1.0, -1.0, 0.0]) * scale,
                xyzs + np.array([-1.0, 0.0, 1.0]) * scale,
                xyzs + np.array([-1.0, 0.0, -1.0]) * scale,
                
                xyzs + np.array([1.0, 1.0, 1.0]) * scale,
                xyzs + np.array([1.0, 1.0, -1.0]) * scale,
                xyzs + np.array([1.0, -1.0, 1.0]) * scale,
                xyzs + np.array([1.0, -1.0, -1.0]) * scale,
                xyzs + np.array([-1.0, 1.0, 1.0]) * scale,
                xyzs + np.array([-1.0, 1.0, -1.0]) * scale,
                xyzs + np.array([-1.0, -1.0, 1.0]) * scale,
                xyzs + np.array([-1.0, -1.0, -1.0]) * scale,

            ], axis=0)

            xyzs = pd.DataFrame(xyzs, columns=['x', 'y', 'z'])
            xyzs = xyzs[
                (xyzs['x'] >= xlims[0]) & (xyzs['x'] <= xlims[1]) &
                (xyzs['y'] >= ylims[0]) & (xyzs['y'] <= ylims[1]) &
                (xyzs['z'] >= zlims[0]) & (xyzs['z'] <= zlims[1])
            ]
            xyzs = xyzs.drop_duplicates().values

            dists, index = obs_tree.query(xyzs, k=1, return_distance=True)
            dists = dists.reshape(-1)
            xyzs = xyzs[dists >= reso]
            
            ### ------ Just For Debug
            # ploter = pyvista.Plotter()
            # for cfg in self.obs_config:
            #     x_length, y_length, z_length = cfg['grid_x_length'], cfg['grid_y_length'], cfg['grid_z_length']
            #     obs_mesh = pyvista.Box(bounds=(
            #         cfg['grid_position'][0] - x_length/2.0, cfg['grid_position'][0] + x_length/2.0,
            #         cfg['grid_position'][1] - y_length/2.0, cfg['grid_position'][1] + y_length/2.0,
            #         cfg['grid_position'][2] - z_length/2.0, cfg['grid_position'][2] + z_length/2.0,
            #     ))
            #     ploter.add_mesh(obs_mesh, color=(0.5, 0.5, 0.5), opacity=0.9)
            # obs_xyzs_show = self.obs_df[self.obs_df['tag']=='Obstacle'][['x', 'y', 'z']].values
            # ploter.add_mesh(pyvista.PointSet(obs_xyzs_show), color=(1.0, 0.0, 0.0))
            # ploter.add_mesh(pyvista.PointSet(xyzs), color=(0.0, 1.0, 0.0))
            # ploter.show()
        
        return xyzs

    def createBoxUnionMesh(self, xyzs:np.array, scale=1.0):
        xyzs_tree = KDTree(xyzs)

        valid_xyzs = []
        faces = []
        for xyz in tqdm(xyzs):
            indexs, dists = xyzs_tree.query_radius(
                xyz.reshape((1, -1)), r=scale * 1.01, return_distance=True, sort_results=True
            )
            indexs = indexs[0]
            # dists = dists[0]
            indexs = indexs[1:]

            if len(indexs) == 0 or len(indexs) == 6:
                continue

            retain_faces = ['z+', 'z-', 'x+', 'x-', 'y+', 'y-']
            for subIdx in indexs:
                sub_xyz = xyzs[subIdx]

                if np.all(sub_xyz - xyz == np.array([1, 0, 0]) * scale):
                    retain_faces.remove('x+')
                elif np.all(sub_xyz - xyz == np.array([-1, 0, 0]) * scale):
                    retain_faces.remove('x-')
                elif np.all(sub_xyz - xyz == np.array([0, 1, 0]) * scale):
                    retain_faces.remove('y+')
                elif np.all(sub_xyz - xyz == np.array([0, -1, 0]) * scale):
                    retain_faces.remove('y-')
                elif np.all(sub_xyz - xyz == np.array([0, 0, 1]) * scale):
                    retain_faces.remove('z+')
                elif np.all(sub_xyz - xyz == np.array([0, 0, -1]) * scale):
                    retain_faces.remove('z-')
                else:
                    raise ValueError

            if len(retain_faces) == 0:
                continue
            
            valid_xyzs.append(xyz)
            x, y, z = xyz
            for retainFace in retain_faces:
                if retainFace == 'x+':
                    face = np.array([
                        xyz + np.array([0.5, -0.5, -0.5]) * scale,
                        xyz + np.array([0.5, +0.5, -0.5]) * scale,
                        xyz + np.array([0.5, +0.5, +0.5]) * scale,
                        xyz + np.array([0.5, -0.5, +0.5]) * scale,
                    ])
                
                elif retainFace == 'x-':
                    face = np.array([
                        xyz + np.array([-0.5, -0.5, -0.5]) * scale,
                        xyz + np.array([-0.5, +0.5, -0.5]) * scale,
                        xyz + np.array([-0.5, +0.5, +0.5]) * scale,
                        xyz + np.array([-0.5, -0.5, +0.5]) * scale,
                    ])
                
                elif retainFace == 'y+':
                    face = np.array([
                        xyz + np.array([-0.5, +0.5, -0.5]) * scale,
                        xyz + np.array([ 0.5, +0.5, -0.5]) * scale,
                        xyz + np.array([ 0.5, +0.5, +0.5]) * scale,
                        xyz + np.array([-0.5, +0.5, +0.5]) * scale,
                    ])
                
                elif retainFace == 'y-':
                    face = np.array([
                        xyz + np.array([-0.5, -0.5, -0.5]) * scale,
                        xyz + np.array([ 0.5, -0.5, -0.5]) * scale,
                        xyz + np.array([ 0.5, -0.5, +0.5]) * scale,
                        xyz + np.array([-0.5, -0.5, +0.5]) * scale,
                    ])
                
                elif retainFace == 'z+':
                    face = np.array([
                        xyz + np.array([-0.5, -0.5, +0.5]) * scale,
                        xyz + np.array([+0.5, -0.5, +0.5]) * scale,
                        xyz + np.array([+0.5, +0.5, +0.5]) * scale,
                        xyz + np.array([-0.5, +0.5, +0.5]) * scale,
                    ])
                
                elif retainFace == 'z-':
                    face = np.array([
                        xyz + np.array([-0.5, -0.5, -0.5]) * scale,
                        xyz + np.array([+0.5, -0.5, -0.5]) * scale,
                        xyz + np.array([+0.5, +0.5, -0.5]) * scale,
                        xyz + np.array([-0.5, +0.5, -0.5]) * scale,
                    ])

                faces.append(face)

        # print('removePoints Num: ', remove_points)
        faces = np.array(faces)
        face_xyzs_df = pd.DataFrame(faces.reshape(-1, 3), columns=['x', 'y', 'z'])
        face_xyzs_df['tag'] = face_xyzs_df.apply(lambda data: '%.1f/%.1f/%.1f' % (data['x'], data['y'], data['z']), axis=1)
        face_xyzs_count = face_xyzs_df['tag'].value_counts()
        face_xyzs_df.drop_duplicates(subset=['tag'], inplace=True)
        face_xyzs_df['id'] = np.arange(0, face_xyzs_df.shape[0], 1)
        face_xyzs_df.set_index(keys=['tag'], inplace=True)
        
        facesIdxs = []
        for face in faces:
            faceIdxs = [4]
            for point in face:
                tag = '%.1f/%.1f/%.1f' % (point[0], point[1], point[2])
                idx = face_xyzs_df.loc[tag, 'id']
                faceIdxs.append(idx)
            facesIdxs.append(faceIdxs)

        mesh = pyvista.PolyData(face_xyzs_df[['x', 'y', 'z']].values, np.hstack(facesIdxs))
        # mesh.clean(point_merging=True, tolerance=0.5, inplace=True)
        mesh.triangulate(inplace=True)

        mesh = self.fixMeshWaterTight(face_xyzs_count, face_xyzs_df, mesh)

        return mesh

    def fixMeshWaterTight(self, xyzs_count:pd.DataFrame, xyzs_df:pd.DataFrame, mesh:pyvista.PolyData):
        with TemporaryDirectory() as tempDir:
            save_path = os.path.join(tempDir, 'test.stl')
            mesh.save(save_path)
            isWaterTight = trimesh.load_mesh(save_path).is_watertight
            # print('Before Fix is WaterTight: ', isWaterTight)

            if not isWaterTight:
                meshfix = MeshFix(mesh)
                holes = meshfix.extract_holes()

                remove_pointIdxs = []
                for cell in holes.cell:
                    maxCount = -1
                    remove_tag = None
                    for point in cell.points:
                        tag = '%.1f/%.1f/%.1f' % (point[0], point[1], point[2])
                        count = xyzs_count[tag]
                        if count > maxCount:
                            remove_tag = tag
                    
                    remove_pointIdxs.append(xyzs_df.loc[remove_tag, 'id'])

                mesh.remove_points(remove_pointIdxs, inplace=True)
                # mesh.plot(style='surface', show_edges=True)

                meshfix = MeshFix(mesh)
                # holes = meshfix.extract_holes()
                # plotter = pyvista.Plotter()
                # plotter.add_mesh(mesh, color=True, show_edges=True)
                # plotter.add_mesh(holes, color="r", line_width=5)
                # plotter.enable_eye_dome_lighting()  # helps depth perception
                # plotter.show()

                meshfix.repair(verbose=False)
                mesh = meshfix.mesh
                mesh.save(save_path)
                isWaterTight = trimesh.load_mesh(save_path).is_watertight
                # print('After Fix is WaterTight: ', isWaterTight)
                assert isWaterTight

        # mesh.plot(style='surface', show_edges=True)

        return mesh

    def debugSearchPathTubeField(self, groupKeys, saveDir):
        vis = VisulizerVista()

        random_colors = np.random.uniform(0.0, 1.0, size=(len(groupKeys), 3))
        for idx, groupKey in enumerate(groupKeys):
            group_dir = os.path.join(saveDir, 'group_%d' % groupKey)

            for path_stl_file in os.listdir(group_dir):
                if path_stl_file.endswith('.stl') and 'path_merge' in path_stl_file:
                    path = os.path.join(group_dir, path_stl_file)
                    mesh = pyvista.STLReader(path).read()
                    vis.plot(mesh, color=random_colors[idx], opacity=1.0, style='surface')

        vis.show()

    def vis_DesignSpace(self, groupKeys, saveDir, obs_config=None):
        groupKeys = [0]

        vis = VisulizerVista()

        # random_colors = np.random.uniform(0.0, 1.0, size=(len(groupKeys), 3))
        random_colors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0]
        ])

        for idx, groupKey in enumerate(groupKeys):
            group_dir = os.path.join(saveDir, 'group_%d' % groupKey)

            for path_stl_file in os.listdir(group_dir):
                path = os.path.join(group_dir, path_stl_file)

                if path_stl_file.endswith('.stl') and ('path_' in path_stl_file):
                    mesh = pyvista.STLReader(path).read()
                    vis.plot(mesh, color=random_colors[idx], opacity=1.0, style='surface')
                
                # if path_stl_file == 'designSpace.stl':
                #     mesh = pyvista.STLReader(path).read()
                #     vis.plot(mesh, color=random_colors[idx], opacity=1.0, style='surface', show_edges=True)
                
                if path_stl_file == 'designMesh.stl':
                    mesh = pyvista.STLReader(path).read()
                    vis.plot(mesh, color=random_colors[idx], opacity=0.75, style='surface', show_edges=True)

        obs_xyzs = self.obs_df[self.obs_df['tag'] == 'Obstacle'][['x', 'y', 'z']].values
        obs_mesh = pyvista.PointSet(obs_xyzs)
        vis.plot(obs_mesh, color=(0.0, 1.0, 0.0), opacity=1.0)

        # if obs_config is not None:
        #     for cfg in obs_config:
        #         x_length, y_length, z_length = cfg['grid_x_length'], cfg['grid_y_length'], cfg['grid_z_length']
        #         obs_mesh = pyvista.Box(bounds=(
        #             cfg['grid_position'][0] - x_length/2.0, cfg['grid_position'][0] + x_length/2.0,
        #             cfg['grid_position'][1] - y_length/2.0, cfg['grid_position'][1] + y_length/2.0,
        #             cfg['grid_position'][2] - z_length/2.0, cfg['grid_position'][2] + z_length/2.0,
        #         ))
        #         vis.plot(obs_mesh, color=(0.5, 0.5, 0.5), opacity=0.8)

        vis.show()

    @staticmethod
    def checkIsWaterTight(path):
        return trimesh.load_mesh(path).is_watertight

def main():
    grid_json_file = '/home/quan/Desktop/tempary/application_pipe/cond.json'
    with open(grid_json_file, 'r') as f:
        env_config = json.load(f)
    
    resPaths = np.load(
        '/home/quan/Desktop/tempary/application_pipe/resPath_config.npy', allow_pickle=True
    ).item()

    case_dir = '/home/quan/Desktop/tempary/application_pipe/stl_case'

    optimizer = TopolyOpt_Helper()
    optimizer.expansion_mesh(
        resPaths=resPaths,
        expansionDist=6, 
        env_config=env_config,
        outputDir=case_dir
    )

if __name__ == '__main__':
    main()    

