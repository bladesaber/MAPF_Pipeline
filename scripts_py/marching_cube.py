import os

import numpy as np
import pyvista
import mcubes
from sklearn.neighbors import KDTree


class MarchingCubeUtils(object):
    @staticmethod
    def marching_cubes(
        xmin: float, xmax: float,
        ymin: float, ymax: float,
        zmin: float, zmax: float,
        source_positions: np.ndarray, 
        resolution: int,
        level: float = 0.1
    ):
        """ level should larger than min mesh size """
        x = np.linspace(xmin, xmax, resolution)
        y = np.linspace(ymin, ymax, resolution)
        z = np.linspace(zmin, zmax, resolution)
        xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')  # Critical: Use indexing='ij' and build grid in (z,y,x) order for PyMCubes
        xyzs = np.concatenate([xg[..., np.newaxis], yg[..., np.newaxis], zg[..., np.newaxis]], axis=-1)
        cube_points = xyzs.reshape((-1, 3))
        
        tree = KDTree(source_positions)
        dists_list, idxs_list = tree.query(cube_points, k=1, return_distance=True)
        dists = np.concatenate(dists_list, axis=-1)
        sdf_values = dists.reshape((resolution, resolution, resolution))
        
        vertices, triangles = mcubes.marching_cubes(sdf_values, level)  # should larger than min mesh size
        surface = pyvista.UnstructuredGrid({pyvista.CellType.TRIANGLE: triangles}, vertices)

        vertices[:, 0] = xmin + vertices[:, 0] / (resolution - 1) * (xmax - xmin)
        vertices[:, 1] = ymin + vertices[:, 1] / (resolution - 1) * (ymax - ymin)
        vertices[:, 2] = zmin + vertices[:, 2] / (resolution - 1) * (zmax - zmin)
        
        # ------ visualize the surface ------
        plotter = pyvista.Plotter()
        plotter.add_mesh(surface, color='cyan', opacity=0.8)
        plotter.add_mesh(pyvista.PointSet(source_positions), color='red', point_size=10)
        plotter.show()
        # ------ visualize the surface ------
        
        return surface