import numpy as np
import pandas as pd
import math

import pyvista

class MeshHelper(object):
    def create_cylinder(self, center_xyz, direction, radius, height):
        mesh = pyvista.Cylinder(center=center_xyz, direction=direction, radius=radius, height=height)
        return mesh
    
    def create_box(self, center_xyz, length=1.0):
        semi_length = length / 2.0
        mesh = pyvista.Box(bounds=(
            -semi_length, semi_length,
            -semi_length, semi_length,
            -semi_length, semi_length,
        ))
        mesh.translate(center_xyz, inplace=True)
        return mesh

    def union_mesh(self, mesh0:pyvista.PolyData, mesh1:pyvista.PolyData):
        mesh = mesh0.boolean_union(mesh1, progress_bar=True)
        return mesh
    
    def diff_mesh(self, mesh0:pyvista.PolyData, mesh1:pyvista.PolyData):
        mesh = mesh0.boolean_difference(mesh1, progress_bar=True)
        return mesh

    def intersect_mesh(self, mesh0:pyvista.PolyData, mesh1:pyvista.PolyData):
        mesh = mesh0.boolean_intersection(mesh1, progress_bar=True)
        return mesh
    
    def cut_mesh(self, mesh0:pyvista.PolyData, mesh1:pyvista.PolyData):
        mesh = mesh0.boolean_cut(mesh1, progress_bar=True)
        return mesh

    def sample_HexSurface(self, xyz, semiLength, direction, sampleReso):
        '''
        xyz: center
        '''

        mesh_pcds = []
        
        ### x-y-bottom
        if direction[0]:
            xs = np.arange(xyz[0]-semiLength, xyz[0]+semiLength + sampleReso, sampleReso)
            ys = np.arange(xyz[1]-semiLength, xyz[1]+semiLength + sampleReso, sampleReso)
            xs, ys = np.meshgrid(xs, ys)
            xys = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1)
            xys = xys.reshape((-1, 2))
            xyzs = np.concatenate([xys, np.ones((xys.shape[0], 1)) * (xyz[2] - semiLength)], axis=1)
            mesh_pcds.append(xyzs)
        
        ### x-z-front
        if direction[1]:
            xs = np.arange(xyz[0]-semiLength, xyz[0]+semiLength + sampleReso, sampleReso)
            zs = np.arange(xyz[2]-semiLength, xyz[2]+semiLength + sampleReso, sampleReso)
            xs, zs = np.meshgrid(xs, zs)
            xzs = np.concatenate([xs[..., np.newaxis], zs[..., np.newaxis]], axis=-1)
            xzs = xzs.reshape((-1, 2))
            xyzs = np.concatenate([xzs[:, 0:1], np.ones((xzs.shape[0], 1)) * (xyz[1] - semiLength), xzs[:, 1:2]], axis=1)
            mesh_pcds.append(xyzs)

        ### y-z-left
        if direction[1]:
            ys = np.arange(xyz[1]-semiLength, xyz[1]+semiLength + sampleReso, sampleReso)
            zs = np.arange(xyz[2]-semiLength, xyz[2]+semiLength + sampleReso, sampleReso)
            ys, zs = np.meshgrid(ys, zs)
            yzs = np.concatenate([ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1)
            yzs = yzs.reshape((-1, 2))
            xyzs = np.concatenate([np.ones((yzs.shape[0], 1)) * (xyz[0] - semiLength), yzs], axis=1)
            mesh_pcds.append(xyzs)
        
        ### x-y-top
        if direction[0]:
            xs = np.arange(xyz[0]-semiLength, xyz[0]+semiLength + sampleReso, sampleReso)
            ys = np.arange(xyz[1]-semiLength, xyz[1]+semiLength + sampleReso, sampleReso)
            xs, ys = np.meshgrid(xs, ys)
            xys = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1)
            xys = xys.reshape((-1, 2))
            xyzs = np.concatenate([xys, np.ones((xys.shape[0], 1)) * (xyz[2] + semiLength)], axis=1)
            mesh_pcds.append(xyzs)
        
        ### x-z-back
        if direction[1]:
            xs = np.arange(xyz[0]-semiLength, xyz[0]+semiLength + sampleReso, sampleReso)
            zs = np.arange(xyz[2]-semiLength, xyz[2]+semiLength + sampleReso, sampleReso)
            xs, zs = np.meshgrid(xs, zs)
            xzs = np.concatenate([xs[..., np.newaxis], zs[..., np.newaxis]], axis=-1)
            xzs = xzs.reshape((-1, 2))
            xyzs = np.concatenate([xzs[:, 0:1], np.ones((xzs.shape[0], 1)) * (xyz[1] + semiLength), xzs[:, 1:2]], axis=1)
            mesh_pcds.append(xyzs)

        ### y-z-right
        if direction[1]:
            ys = np.arange(xyz[1]-semiLength, xyz[1]+semiLength + sampleReso, sampleReso)
            zs = np.arange(xyz[2]-semiLength, xyz[2]+semiLength + sampleReso, sampleReso)
            ys, zs = np.meshgrid(ys, zs)
            yzs = np.concatenate([ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1)
            yzs = yzs.reshape((-1, 2))
            xyzs = np.concatenate([np.ones((yzs.shape[0], 1)) * (xyz[0] + semiLength), yzs], axis=1)
            mesh_pcds.append(xyzs)

        mesh_pcds = np.concatenate(mesh_pcds, axis=0)
        mesh_pcds = pd.DataFrame(mesh_pcds, columns=['x', 'y', 'z'])
        mesh_pcds.drop_duplicates(inplace=True)
        mesh_pcds = mesh_pcds.values

        return mesh_pcds

    def sample_XCylinderSurface(self, xyz, semiHeight, radius, sampleReso, sampleRadiusReso, direction):
        mesh_pcds = []

        ### y-z-left
        if direction[0]:
            xyzs = []
            for subRadius in np.arange(0.0, radius + sampleRadiusReso, sampleRadiusReso):
                radLength = 2 * subRadius * np.pi
                num = np.maximum(math.ceil(radLength / sampleReso), 3)
                rads = np.arange(0, num, 1) * (2.0 * np.pi / num)

                ys = (xyz[1] + np.cos(rads) * subRadius).reshape(-1, 1)
                zs = (xyz[2] + np.sin(rads) * subRadius).reshape(-1, 1)
                xs = np.ones((ys.shape[0], 1)) * (xyz[0] - semiHeight)

                subCircle_xyzs = np.concatenate([xs, ys, zs], axis=1)
                xyzs.append(subCircle_xyzs)
            
            xyzs = np.concatenate(xyzs, axis=0)
            mesh_pcds.append(xyzs)
        
        if direction[1]:
            xyzs = []

            radLength = 2 * radius * np.pi
            num = np.maximum(math.ceil(radLength / sampleReso), 3)
            rads = np.arange(0, num, 1) * (2.0 * np.pi / num)
            ys = (xyz[1] + np.cos(rads) * radius).reshape(-1, 1)
            zs = (xyz[2] + np.sin(rads) * radius).reshape(-1, 1)

            for height in np.arange(xyz[0] - semiHeight, xyz[0] + semiHeight + sampleReso, sampleReso):
                subCircle_xyzs = np.concatenate([np.ones((ys.shape[0], 1)) * height, ys, zs], axis=1)
                xyzs.append(subCircle_xyzs)
            
            xyzs = np.concatenate(xyzs, axis=0)
            mesh_pcds.append(xyzs)
                
        ### y-z-right
        if direction[2]:
            xyzs = []
            for subRadius in np.arange(0.0, radius + sampleRadiusReso, sampleRadiusReso):
                radLength = 2 * subRadius * np.pi
                num = np.maximum(math.ceil(radLength / sampleReso), 3)
                rads = np.arange(0, num, 1) * (2.0 * np.pi / num)

                ys = (xyz[1] + np.cos(rads) * subRadius).reshape(-1, 1)
                zs = (xyz[2] + np.sin(rads) * subRadius).reshape(-1, 1)
                xs = np.ones((ys.shape[0], 1)) * (xyz[0] + semiHeight)

                subCircle_xyzs = np.concatenate([xs, ys, zs], axis=1)
                xyzs.append(subCircle_xyzs)
            
            xyzs = np.concatenate(xyzs, axis=0)
            mesh_pcds.append(xyzs)

        mesh_pcds = np.concatenate(mesh_pcds, axis=0)
        mesh_pcds = pd.DataFrame(mesh_pcds, columns=['x', 'y', 'z'])
        mesh_pcds.drop_duplicates(inplace=True)
        mesh_pcds = mesh_pcds.values

        return mesh_pcds

    def sample_YCylinderSurface(self, xyz, semiHeight, radius, sampleReso, sampleRadiusReso, direction):
        mesh_pcds = []

        ### x-z-front
        if direction[0]:
            xyzs = []
            for subRadius in np.arange(0.0, radius + sampleRadiusReso, sampleRadiusReso):
                radLength = 2 * subRadius * np.pi
                num = np.maximum(math.ceil(radLength / sampleReso), 3)
                rads = np.arange(0, num, 1) * (2.0 * np.pi / num)

                xs = (xyz[0] + np.cos(rads) * subRadius).reshape(-1, 1)
                zs = (xyz[2] + np.sin(rads) * subRadius).reshape(-1, 1)
                ys = np.ones((xs.shape[0], 1)) * (xyz[1] - semiHeight)

                subCircle_xyzs = np.concatenate([xs, ys, zs], axis=1)
                xyzs.append(subCircle_xyzs)
            
            xyzs = np.concatenate(xyzs, axis=0)
            mesh_pcds.append(xyzs)
        
        if direction[1]:
            xyzs = []

            radLength = 2 * radius * np.pi
            num = np.maximum(math.ceil(radLength / sampleReso), 3)
            rads = np.arange(0, num, 1) * (2.0 * np.pi / num)
            xs = (xyz[0] + np.cos(rads) * radius).reshape(-1, 1)
            zs = (xyz[2] + np.sin(rads) * radius).reshape(-1, 1)

            for height in np.arange(xyz[1] - semiHeight, xyz[1] + semiHeight + sampleReso, sampleReso):
                subCircle_xyzs = np.concatenate([xs, np.ones((xs.shape[0], 1)) * height, zs], axis=1)
                xyzs.append(subCircle_xyzs)
            
            xyzs = np.concatenate(xyzs, axis=0)
            mesh_pcds.append(xyzs)
                
        ### x-z-back
        if direction[2]:
            xyzs = []
            for subRadius in np.arange(0.0, radius + sampleRadiusReso, sampleRadiusReso):
                radLength = 2 * subRadius * np.pi
                num = np.maximum(math.ceil(radLength / sampleReso), 3)
                rads = np.arange(0, num, 1) * (2.0 * np.pi / num)

                xs = (xyz[0] + np.cos(rads) * subRadius).reshape(-1, 1)
                zs = (xyz[2] + np.sin(rads) * subRadius).reshape(-1, 1)
                ys = np.ones((xs.shape[0], 1)) * (xyz[1] + semiHeight)

                subCircle_xyzs = np.concatenate([xs, ys, zs], axis=1)
                xyzs.append(subCircle_xyzs)
            
            xyzs = np.concatenate(xyzs, axis=0)
            mesh_pcds.append(xyzs)

        mesh_pcds = np.concatenate(mesh_pcds, axis=0)
        mesh_pcds = pd.DataFrame(mesh_pcds, columns=['x', 'y', 'z'])
        mesh_pcds.drop_duplicates(inplace=True)
        mesh_pcds = mesh_pcds.values

        return mesh_pcds
    
    def sample_ZCylinderSurface(self, xyz, semiHeight, radius, sampleReso, sampleRadiusReso, direction):
        mesh_pcds = []

        ### x-y-bottom
        if direction[0]:
            xyzs = []
            for subRadius in np.arange(0.0, radius + sampleRadiusReso, sampleRadiusReso):
                radLength = 2 * subRadius * np.pi
                num = np.maximum(math.ceil(radLength / sampleReso), 3)
                rads = np.arange(0, num, 1) * (2.0 * np.pi / num)

                xs = (xyz[0] + np.cos(rads) * subRadius).reshape(-1, 1)
                ys = (xyz[1] + np.sin(rads) * subRadius).reshape(-1, 1)
                zs = np.ones((xs.shape[0], 1)) * (xyz[2] - semiHeight)

                subCircle_xyzs = np.concatenate([xs, ys, zs], axis=1)
                xyzs.append(subCircle_xyzs)
            
            xyzs = np.concatenate(xyzs, axis=0)
            mesh_pcds.append(xyzs)
        
        if direction[1]:
            xyzs = []

            radLength = 2 * radius * np.pi
            num = np.maximum(math.ceil(radLength / sampleReso), 3)
            rads = np.arange(0, num, 1) * (2.0 * np.pi / num)
            xs = (xyz[0] + np.cos(rads) * radius).reshape(-1, 1)
            ys = (xyz[1] + np.sin(rads) * radius).reshape(-1, 1)

            for height in np.arange(xyz[2] - semiHeight, xyz[2] + semiHeight + sampleReso, sampleReso):
                subCircle_xyzs = np.concatenate([xs, ys, np.ones((xs.shape[0], 1)) * height], axis=1)
                xyzs.append(subCircle_xyzs)
            
            xyzs = np.concatenate(xyzs, axis=0)
            mesh_pcds.append(xyzs)
                
        ### x-y-top
        if direction[2]:
            xyzs = []
            for subRadius in np.arange(0.0, radius + sampleRadiusReso, sampleRadiusReso):
                radLength = 2 * subRadius * np.pi
                num = np.maximum(math.ceil(radLength / sampleReso), 3)
                rads = np.arange(0, num, 1) * (2.0 * np.pi / num)

                xs = (xyz[0] + np.cos(rads) * subRadius).reshape(-1, 1)
                ys = (xyz[1] + np.sin(rads) * subRadius).reshape(-1, 1)
                zs = np.ones((xs.shape[0], 1)) * (xyz[2] + semiHeight)

                subCircle_xyzs = np.concatenate([xs, ys, zs], axis=1)
                xyzs.append(subCircle_xyzs)
            
            xyzs = np.concatenate(xyzs, axis=0)
            mesh_pcds.append(xyzs)

        mesh_pcds = np.concatenate(mesh_pcds, axis=0)
        mesh_pcds = pd.DataFrame(mesh_pcds, columns=['x', 'y', 'z'])
        mesh_pcds.drop_duplicates(inplace=True)
        mesh_pcds = mesh_pcds.values

        return mesh_pcds

    def reconstructSurface(self, pointCloud:np.array, nbr_sz=None):
        pcd_mesh = pyvista.PolyData(pointCloud)
        mesh = pcd_mesh.reconstruct_surface(nbr_sz=nbr_sz, progress_bar=True)
        return mesh

    def plot_PointCloud(self, pointCloud:np.array, style='wireframe', opacity=1.0):
        '''
        style: wireframe, surface, None
        '''

        pcd_mesh = pyvista.PolyData(pointCloud)
        pcd_mesh.plot(style=style, opacity=opacity)
