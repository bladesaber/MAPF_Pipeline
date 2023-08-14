import numpy as np
import pandas as pd

class GridEnv_Utils(object):
    @staticmethod
    def obstacle_convert_grid(xyzs:np.array, scale):
        scale_xyzs = xyzs * scale
        scale_xyzs = np.round(scale_xyzs, decimals=1)
        scale_xyzs = pd.DataFrame(scale_xyzs).drop_duplicates().values
        return scale_xyzs

    @staticmethod
    def pipe_convert_grid(xyz, scale, xmin, xmax, ymin, ymax, zmin, zmax):
        scale_xyz = xyz * scale
        xyz_grid = np.round(scale_xyz, decimals=0)
        xyz_grid[0] = np.maximum(np.minimum(xyz_grid[0], xmax), xmin)
        xyz_grid[1] = np.maximum(np.minimum(xyz_grid[1], ymax), ymin)
        xyz_grid[2] = np.maximum(np.minimum(xyz_grid[2], zmax), zmin)
        return scale_xyz, xyz_grid
