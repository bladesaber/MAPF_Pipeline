import open3d as o3d
import numpy as np


class Open3dUtils(object):
    @staticmethod
    def create_point_cloud(coords: np.ndarray, colors: np.ndarray = None, normals: np.ndarray = None):
        """
        colors: range [0, 1]
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)

        if colors is not None:
            if colors.ndim == 1:
                color_np = np.tile(colors.reshape((1, -1)), [pcd.shape[0], 1])
            else:
                color_np = colors
            pcd.colors = o3d.utility.Vector3dVector(color_np)

        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        return pcd

    @staticmethod
    def create_triangle_mesh():
        mesh = o3d.geometry.TriangleMesh()
        raise NotImplementedError

    @staticmethod
    def sample_from_mesh(mesh: o3d.geometry.TriangleMesh, num_of_points, method) -> o3d.geometry.PointCloud:
        if method == 'uniform':
            return mesh.sample_points_uniformly(num_of_points)
        elif method == 'poisson':
            return mesh.sample_points_poisson_disk(num_of_points)
        else:
            raise ValueError('[ERROR]: Non-Valid method')
