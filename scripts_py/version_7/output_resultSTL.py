import numpy as np
import pyvista
import os
from tempfile import TemporaryDirectory
import trimesh
from tqdm import tqdm
import math
import pymeshlab
import json

from scripts_py.version_7.mesh_helper import MeshHelper_Vista

grid_json_file = '/home/quan/Desktop/tempary/application_pipe/cond.json'
with open(grid_json_file, 'r') as f:
    env_config = json.load(f)

res_config = np.load(os.path.join(env_config['projectDir'], 'resPath_config.npy'), allow_pickle=True).item()

def polar2vec(polarVec, length=1.0):
    dz = length * math.sin(polarVec[1])
    dl = length * math.cos(polarVec[1])
    dx = dl * math.cos(polarVec[0])
    dy = dl * math.sin(polarVec[0])
    return np.array([dx, dy, dz])

def meshlab_diffOpt(file0, file1, outputFile):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file0)
    ms.load_new_mesh(file1)
    ms.generate_boolean_difference()
    ms.save_current_mesh(outputFile)

def meshlab_booleanOpt(meshPaths, style, outputFile):
    for i in range(1, len(meshPaths)):
        if i == 1:
            file0 = meshPaths[i-1]
        else:
            file0 = outputFile
        file1 = meshPaths[i]

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file0)
        ms.load_new_mesh(file1)
        
        if style == 'union':
            ms.generate_boolean_union()
        # elif style == 'difference':
        #     ms.generate_boolean_difference()
        elif style == 'intersection':
            ms.generate_boolean_intersection()
        else:
            raise ValueError
    
        ms.save_current_mesh(outputFile)

        # plt = pyvista.Plotter()
        # plt.add_mesh(pyvista.STLReader(outputFile).read(), opacity=1.0, color='b')
        # plt.show()

def reconstruct_outerTubeMesh(xyzs, radius, startDire, endDire):
    # start_vec = polar2vec(startDire)
    # start_interplot_xyz = xyzs[0, :3] + start_vec * 0.1
    # end_vec = polar2vec(endDire)
    # end_interplot_xyz = xyzs[-1, :3] - end_vec * 0.1
    # path_xyzs = np.concatenate([
    #     xyzs[0, :].reshape((1, -1)),
    #     start_interplot_xyz.reshape((1, -1)),
    #     xyzs[1:-1, :],
    #     end_interplot_xyz.reshape((1, -1)),
    #     xyzs[-1, :].reshape((1, -1)),
    # ])
    # mesh0 = MeshHelper_Vista.create_Tube(path_xyzs, radius=radius)
    # mesh0.triangulate(inplace=True)

    start_vec = polar2vec(startDire)
    end_vec = polar2vec(endDire)
    paddingStart = (xyzs[0] - start_vec).reshape((1, -1))
    paddingEnd = (xyzs[-1] + end_vec).reshape((1, -1))
    padding_xyzs = np.concatenate([paddingStart, xyzs, paddingEnd], axis=0)
    mesh1 = MeshHelper_Vista.create_Tube(padding_xyzs, radius=radius)
    mesh1.triangulate(inplace=True)

    start_vec = polar2vec(startDire)
    startTubeMesh = MeshHelper_Vista.create_Tube(
        np.array([
            xyzs[0] - start_vec * 1.5, xyzs[0]
        ]), radius=radius * 1.25
    )
    end_vec = polar2vec(endDire)
    endTubeMesh = MeshHelper_Vista.create_Tube(
        np.array([
            xyzs[-1], xyzs[-1] + end_vec * 1.5
        ]), radius=radius * 1.25
    )
    # mesh1 = mesh1.boolean_difference(startTubeMesh).boolean_difference(endTubeMesh)

    with TemporaryDirectory() as tempDir:
        mesh_path = os.path.join(tempDir, 'mesh.stl')
        startTubePath = os.path.join(tempDir, 'start.stl')
        endTubePath = os.path.join(tempDir, 'end.stl')
        ouputFile = os.path.join(tempDir, 'output.stl')

        mesh1.save(mesh_path, binary=True)
        startTubeMesh.save(startTubePath, binary=True)
        endTubeMesh.save(endTubePath, binary=True)

        meshlab_diffOpt(startTubePath, mesh_path, ouputFile)
        meshlab_diffOpt(endTubePath, ouputFile, ouputFile)

        mesh2 = pyvista.STLReader(ouputFile).read()
    
    # plt = pyvista.Plotter()
    # # plt.add_mesh(mesh0, opacity=0.7, color='b', show_edges=False)
    # plt.add_mesh(mesh1, opacity=0.7, color='r', show_edges=False)
    # plt.add_mesh(mesh2, opacity=0.9, color='b', show_edges=True)
    # # plt.add_mesh(startTubeMesh, opacity=0.7, color='g', show_edges=True)
    # # plt.add_mesh(endTubeMesh, opacity=0.7, color='g', show_edges=True)
    # # plt.add_mesh(boundingBox, opacity=0.7, color='g', show_edges=True)
    # plt.show()

    return mesh2

def reconstruct_innerTubeMesh(xyzs, radius, startDire, endDire, paddingNum=1):
    start_vec = polar2vec(startDire)
    end_vec = polar2vec(endDire)
    paddingStart = np.array([xyzs[0] - (i+1) * start_vec for i in range(paddingNum, -1, -1)])
    paddingEnd = np.array([xyzs[-1] + (i+1) * end_vec for i in range(paddingNum)])
    padding_xyzs = np.concatenate([paddingStart, xyzs, paddingEnd], axis=0)

    mesh1 = MeshHelper_Vista.create_Tube(padding_xyzs, radius=radius)
    mesh1.triangulate(inplace=True)

    # plt = pyvista.Plotter()
    # plt.add_mesh(mesh1, opacity=0.9, color='b', show_edges=True)
    # plt.show()

    return mesh1

def createPaddingTube():
    group_keys = [0, 1, 2, 3, 4]

    group_saveDir = env_config['projectDir']
    for groupIdx in group_keys:
        res_info = res_config[groupIdx]

        with TemporaryDirectory() as tempDir:
            outer_files, inner_files = [], []
            for pathIdx in tqdm(res_info.keys()):
                path_info = res_info[pathIdx]
                path_xyzr = path_info['path_xyzr']
                outer_radius = path_info['grid_radius']
                inner_radius = outer_radius * 0.8

                outer_mesh = reconstruct_outerTubeMesh(
                    xyzs=path_xyzr[:, :3],
                    radius=outer_radius,
                    startDire=path_info['startDire'],
                    endDire=path_info['endDire'],
                )
                outer_path = os.path.join(tempDir, 'outer_%d.stl' % pathIdx)
                outer_mesh.save(outer_path)
                outer_files.append(outer_path)

                # inner_mesh = reconstruct_innerTubeMesh(
                #     xyzs=path_xyzr[:, :3],
                #     radius=inner_radius,
                #     startDire=path_info['startDire'],
                #     endDire=path_info['endDire'],
                #     paddingNum=2
                # )
                # inner_path = os.path.join(tempDir, 'inner_%d.stl' % pathIdx)
                # inner_mesh.save(inner_path)
                # inner_files.append(inner_path)

            # outer_merge_path = os.path.join(tempDir, 'outer_merge.stl')
            outer_merge_path = os.path.join(group_saveDir, 'groupPath_%d.stl'%groupIdx)
            meshlab_booleanOpt(outer_files, style='union', outputFile=outer_merge_path)

            # inner_merge_path = os.path.join(tempDir, 'inner_merge.stl')
            # meshlab_booleanOpt(inner_files, style='union', outputFile=inner_merge_path)

            # ### ------
            # final_merge_path = os.path.join(group_saveDir, 'groupPath_%d.stl'%groupIdx)
            # meshlab_diffOpt(file0=inner_merge_path, file1=outer_merge_path, outputFile=final_merge_path)
            # print("Output Path: %s" % final_merge_path)

            ### ------
            # outer_mesh = pyvista.STLReader(outer_merge_path).read().triangulate()
            # inner_mesh = pyvista.STLReader(inner_merge_path).read().triangulate()
            # final_mesh = outer_mesh.boolean_difference(inner_mesh)

            ### ------
            # outer_mesh = trimesh.load_mesh(outer_merge_path)
            # inner_mesh = trimesh.load_mesh(inner_merge_path)
            # trimesh.boolean.difference([outer_mesh, inner_mesh])

            # plt = pyvista.Plotter()
            # # plt.add_mesh(pyvista.STLReader(outer_merge_path).read(), opacity=0.5, color='g')
            # # plt.add_mesh(pyvista.STLReader(inner_merge_path).read(), opacity=0.75, color='b')
            # plt.add_mesh(pyvista.STLReader(final_merge_path).read(), opacity=1.0, color='r')
            # plt.show()

createPaddingTube()
