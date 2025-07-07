import numpy as np
import pandas as pd
import pyvista
import math
from sklearn.neighbors import KDTree
from typing import Dict, Union, Literal

from scripts_py.mapf_pkg.shape_utils import AxisTransform

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3

from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Advancing_front_surface_reconstruction import advancing_front_surface_reconstruction
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3

from CGAL.CGAL_Point_set_processing_3 import grid_simplify_point_set, compute_average_spacing


class CGALUtils:
    @staticmethod
    def delaunay_triangulation(points: np.ndarray) -> pyvista.UnstructuredGrid:
        """
        Perform Delaunay triangulation on a set of 3D points.
        Args:
            points: numpy array of shape (n, 3)
        Returns:
            numpy array of shape (n, 3)
        """
        point_to_index = {tuple(point): idx for idx, point in enumerate(points)}
        cgal_points = [Point_3(float(x), float(y), float(z)) for x, y, z in points]
        
        dt = Delaunay_triangulation_3()
        dt.insert(cgal_points)
        
        connectivity = []
        for cell in dt.all_cells():
            if dt.is_infinite(cell):
                continue
            vertices = []
            for i in range(4):
                vertex = cell.vertex(i)
                point = (vertex.point().x(), vertex.point().y(), vertex.point().z())
                vertices.append(point_to_index[point])
            connectivity.append(tuple(vertices))
        
        grid = pyvista.UnstructuredGrid({pyvista.CellType.TETRA: connectivity}, points)
        return grid


    @staticmethod
    def poisson_surface_reconstruction(points: np.ndarray) -> pyvista.UnstructuredGrid:
        """
        Reconstruct a surface from a point cloud using Poisson reconstruction
        Parameters:
        points : numpy array of shape (n, 3) containing the 3D points
        output_file : path to save the reconstructed surface (OFF format)
        """
        point_to_index = {tuple(point): idx for idx, point in enumerate(points)}

        # Create a CGAL Point_set_3 object
        point_set = Point_set_3()
        
        # Add points to the point set
        for p in points:
            point_set.insert(Point_3(float(p[0]), float(p[1]), float(p[2])))
        
        # Use advancing front surface reconstruction (no normals needed)
        polyhedron = Polyhedron_3()
        advancing_front_surface_reconstruction(point_set, polyhedron)

        connectivity = []
        for face in polyhedron.facets():
            vertices = []
            # Get the halfedge of the face and iterate through its vertices
            halfedge = face.halfedge()
            for i in range(3):  # Triangle faces have 3 vertices
                vertex = halfedge.vertex()
                point = (vertex.point().x(), vertex.point().y(), vertex.point().z())
                vertices.append(point_to_index[point])
                halfedge = halfedge.next()
            connectivity.append(tuple(vertices))
        connectivity = np.array(connectivity)
        grid = pyvista.UnstructuredGrid({pyvista.CellType.TRIANGLE: connectivity}, points)
        return grid


class PathSurfaceGenerator(object):


    class PathSurfaceItem(object):
        def __init__(
                self, name: str, path: np.ndarray, refine_reso: float = 0.1,
                left_vec: np.ndarray = None, right_vec: np.ndarray = None,
                is_left_input: bool = False, is_right_input: bool = False,
                is_left_cut: bool = False, is_right_cut: bool = False
            ):
            self.name = name
            self.path = path
            self.refined_path: np.ndarray = PathSurfaceGenerator.refine_path(path, refine_reso)
            self.left_vec = left_vec
            self.right_vec = right_vec
            self.is_left_input = is_left_input
            self.is_right_input = is_right_input
            self.is_left_cut = is_left_cut
            self.is_right_cut = is_right_cut
            self.points: np.ndarray = None

        def to_mesh(self):
            return pyvista.PointSet(self.points)


    @staticmethod
    def generate_surface_points_of_sphere(center: np.ndarray, radius: float, vector: np.ndarray, reso: float):
        """
        sample points on the surface of the sphere based on Spherical coordinate system and rotation it to align with the vector
        """
        xs_nums = math.ceil(np.pi * radius / reso)
        rads = np.linspace(0, np.pi, xs_nums)

        points_list = []
        for rad in rads:
            x = radius * np.cos(rad)
            sub_radius = np.sqrt(radius ** 2 - x ** 2)
            cross_section_perimeter = sub_radius * 2 * np.pi
            cross_section_nums = max(math.ceil(cross_section_perimeter / reso), 1)
            theta = np.linspace(0, 2 * np.pi, cross_section_nums)
            points = np.zeros(shape=(cross_section_nums, 3))
            points[:, 0] = x
            points[:, 1] = sub_radius * np.cos(theta)
            points[:, 2] = sub_radius * np.sin(theta)
            points = AxisTransform.pcd_transform(points, AxisTransform.rot_mat_from_vectors(vector))
            points = points + center
            points_list.append(points)
        points = np.concatenate(points_list, axis=0)
        return points

    @staticmethod
    def generate_surface_points_of_disk(center: np.ndarray, radius: float, vector: np.ndarray, reso: float):
        """
        sample points on the surface of the disk based on Spherical coordinate system and rotation it to align with the vector
        """
        num = max(math.ceil(radius / reso), 1)
        radius_list = np.linspace(0, radius, num)

        points_list = []
        for radius in radius_list:
            sub_length = 2.0 * radius * np.pi
            sub_num = max(math.ceil(sub_length / reso), 1)
            sub_rads = np.linspace(0, 2 * np.pi, sub_num)
            points = np.zeros(shape=(sub_num, 3))
            points[:, 0] = 0
            points[:, 1] = radius * np.cos(sub_rads)
            points[:, 2] = radius * np.sin(sub_rads)
            points_list.append(points)

        points = np.concatenate(points_list, axis=0)
        vector = vector / np.linalg.norm(vector)
        rot_mat = AxisTransform.rot_mat_from_vectors(vector)
        points = AxisTransform.pcd_transform(points, rot_mat)
        points = points + center
        return points

    @staticmethod
    def refine_path(path: np.ndarray, reso: float):
        """
        refine the path by adding more points
        """
        new_path = []
        for i in range(path.shape[0]):
            if i == path.shape[0] - 1:
                new_path.append(path[i])
                break
            vec = path[i + 1, :3] - path[i, :3]
            length = np.linalg.norm(vec)
            norm_vec = vec / length
            if length > reso:
                step_num = math.ceil(length / reso)
                for step in range(step_num):
                    path_i = path[i].copy()
                    path_i[:3] = path_i[:3] + norm_vec * step * length / step_num
                    new_path.append(path_i)
            else:
                new_path.append(path[i])
        return np.array(new_path)
        
    @staticmethod
    def remove_inner_points_from_path(points: np.ndarray, path: np.ndarray, relax_factor=0.15):
        """
        points: (n, 3), path: (n, 4) means (x, y, z, radius)
        """
        valid_bool = np.full(shape=(points.shape[0],), fill_value=True, dtype=bool)
        points_tree = KDTree(points)

        idxs_list = points_tree.query_radius(path[:, :3], r=path[:, 3] - relax_factor, return_distance=False)
        for path_idx, point_idxs in enumerate(idxs_list):
            if len(point_idxs) > 0:
                valid_bool[point_idxs] = False
        points = points[valid_bool]
        return points

    @staticmethod
    def cut_clamp(
        points: np.ndarray, 
        center: np.ndarray, radius: float, vec: np.ndarray, is_input: bool, 
        relax_factor: float = 0.15
    ):
        """
        points: (n, 3), center: (3,), radius: float, vec: (3,), is_input: bool
        """
        tree = KDTree(points)
        idxs = tree.query_radius(center.reshape(1, -1), r=radius + relax_factor, return_distance=False)[0]

        is_in_range_bool = np.full(shape=(points.shape[0],), fill_value=False, dtype=bool)
        is_in_range_bool[idxs] = True

        cen2pcd = points - center
        norm_vec = vec / np.linalg.norm(vec)
        dist_list = np.sum(cen2pcd * norm_vec, axis=1)
        if is_input:
            is_cut_bool = np.full(shape=(points.shape[0],), fill_value=False, dtype=bool)
            is_cut_bool[dist_list < 0.0] = True
        else:
            is_cut_bool = np.full(shape=(points.shape[0],), fill_value=False, dtype=bool)
            is_cut_bool[dist_list > 0.0] = True
        is_in_range_bool = is_in_range_bool & is_cut_bool
        return points[~is_in_range_bool]

    @staticmethod
    def generate_surface_points_of_path(
        path: np.ndarray, reso: float, inlet_vec: np.ndarray = None, outlet_vec: np.ndarray = None
    ):
        """
        path: (n, 4) means (x, y, z, radius)
        """
        vec_list = path[1:, :3] - path[:-1, :3]
        if inlet_vec is not None:
            vec_list[0, :] = inlet_vec
        if outlet_vec is not None:
            vec_list = np.concatenate([vec_list, outlet_vec.reshape(1, -1)], axis=0)
        else:
            vec_list = np.concatenate([vec_list, vec_list[-1, :].reshape(1, -1)], axis=0)

        points = []
        for point, vec in zip(path, vec_list):
            points.append(PathSurfaceGenerator.generate_surface_points_of_sphere(point[:3], point[3], vec, reso))
        points = np.concatenate(points, axis=0)
        return points

    @staticmethod
    def generate_surface_points_of_paths(
        paths: dict[str, PathSurfaceItem], reso: float, relax_factor: float = 1e-1
    ):
        for name, path_item in paths.items():
            path_item.points = PathSurfaceGenerator.generate_surface_points_of_path(
                path_item.path, reso, path_item.left_vec, path_item.right_vec
            )
            for other_name, other_path_item in paths.items():
                path_item.points = PathSurfaceGenerator.remove_inner_points_from_path(
                    path_item.points, other_path_item.refined_path, relax_factor
                )
            
            if path_item.is_left_cut:
                path_item.points = np.concatenate([
                    path_item.points, 
                    PathSurfaceGenerator.generate_surface_points_of_disk(
                    path_item.path[0, :3], path_item.path[0, 3], path_item.left_vec, reso
                    )
                ], axis=0)
            
            if path_item.is_right_cut:
                path_item.points = np.concatenate([
                    path_item.points, 
                    PathSurfaceGenerator.generate_surface_points_of_disk(
                        path_item.path[-1, :3], path_item.path[-1, 3], path_item.right_vec, reso
                    )
                ], axis=0)
        
        for _, path_item in paths.items():
            for _, other_path_item in paths.items():
                if other_path_item.is_left_cut:
                    if other_path_item.left_vec is None:
                        left_vec = other_path_item.path[1, :3] - other_path_item.path[0, :3]
                        is_left_input = True
                    else:
                        left_vec = other_path_item.left_vec
                        is_left_input = other_path_item.is_left_input

                    path_item.points = PathSurfaceGenerator.cut_clamp(
                        path_item.points, 
                        other_path_item.path[0, :3], 
                        other_path_item.path[0, 3], 
                        left_vec, is_left_input, 
                        relax_factor
                    )
                
                if other_path_item.is_right_cut:
                    if other_path_item.right_vec is None:
                        right_vec = other_path_item.path[-1, :3] - other_path_item.path[-2, :3]
                        is_right_input = False
                    else:
                        right_vec = other_path_item.right_vec
                        is_right_input = other_path_item.is_right_input
                    
                    path_item.points = PathSurfaceGenerator.cut_clamp(
                        path_item.points, 
                        other_path_item.path[-1, :3], 
                        other_path_item.path[-1, 3], 
                        right_vec, is_right_input, 
                        relax_factor
                    )

        return paths

    @staticmethod
    def merge_too_close_points(points: np.ndarray, voxel_size: float, method: Literal['mean', 'median'] = 'mean'):        
        points_df = pd.DataFrame(points, columns=['x', 'y', 'z'])
        points_df['x_int'] = (points_df['x'] / voxel_size).astype(int)
        points_df['y_int'] = (points_df['y'] / voxel_size).astype(int)
        points_df['z_int'] = (points_df['z'] / voxel_size).astype(int)

        points_df['key'] = (
            points_df['x_int'].astype(str) + '_' 
            + points_df['y_int'].astype(str) + '_' 
            + points_df['z_int'].astype(str)
        )
        if method == 'mean':
            points_df = points_df.groupby('key').mean()[['x', 'y', 'z']]
        elif method == 'median':
            points_df.drop_duplicates(subset=['key'], inplace=True)
            points_df['x'] = (points_df['x_int'] + 0.5) * voxel_size
            points_df['y'] = (points_df['y_int'] + 0.5) * voxel_size
            points_df['z'] = (points_df['z_int'] + 0.5) * voxel_size
        else:
            raise ValueError(f"Invalid method: {method}")

        points = points_df[['x', 'y', 'z']].values
        return points
        
    @staticmethod
    def compute_shorest_distance_of_points(points: np.ndarray) -> float:
        tree = KDTree(points)
        dists_list, idxs_list = tree.query(points, k=2, return_distance=True)
        min_dist = np.inf
        for i, (dists, idxs) in enumerate(zip(dists_list, idxs_list)):
            for dist, idx in zip(dists, idxs):
                if i != idx:
                    min_dist = min(min_dist, dist)
        return min_dist

    @staticmethod
    def generate_surface(
        paths: dict[str, np.ndarray | bool], path_reso: float, point_reso: float,
    ) -> pyvista.UnstructuredGrid:
        refine_reso = path_reso * 0.5
        relax_factor = min(point_reso, refine_reso) * 0.5
        merge_reso = point_reso * 0.25

        path_dict: dict[str, PathSurfaceGenerator.PathSurfaceItem] = {}
        for name, path_info in paths.items():
            path_dict[name] = PathSurfaceGenerator.PathSurfaceItem(
                name=name, path=path_info['path'], 
                refine_reso=path_reso,
                is_left_input=path_info.get('is_left_input', False), 
                is_right_input=path_info.get('is_right_input', False),
                is_left_cut=path_info.get('is_left_cut', False), 
                is_right_cut=path_info.get('is_right_cut', False),
                left_vec=path_info.get('left_vec', None),
                right_vec=path_info.get('right_vec', None),
            )
        
        path_dict = PathSurfaceGenerator.generate_surface_points_of_paths(path_dict, point_reso, relax_factor)
        # plotter = pyvista.Plotter()
        # for name, path_item in paths.items():
        #     plotter.add_mesh(path_item.to_mesh(), color=np.random.rand(3), point_size=10)
        # plotter.show()

        points = []
        for name, path_item in path_dict.items():
            points.append(path_item.points)
        points = np.concatenate(points, axis=0)
        merged_points = PathSurfaceGenerator.merge_too_close_points(points, merge_reso)

        # plotter = pyvista.Plotter()
        # plotter.add_mesh(pyvista.PointSet(points), color='blue', point_size=8)
        # plotter.add_mesh(pyvista.PointSet(merged_points), color='red', point_size=8)
        # plotter.show()

        mesh = CGALUtils.poisson_surface_reconstruction(merged_points)
        # plotter = pyvista.Plotter()
        # plotter.add_mesh(mesh, color='red', style='wireframe')
        # # plotter.add_mesh(pyvista.PointSet(points), color='blue', point_size=8)
        # plotter.show()

        return mesh


def test_generate_surface():
    path_reso = 1.0
    point_reso = 0.1

    paths = {
        'path1': {
            'name': 'path1',
            'path': np.array([
                [0.0, 0.0, 0.0, 1.0], 
                [1.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [4.0, 0.0, 0.0, 1.0],
                [5.0, 0.0, 0.0, 1.0],
                [6.0, 0.0, 0.0, 1.0],
                [7.0, 0.0, 0.0, 1.0],
                [8.0, 0.0, 0.0, 1.0],
                [9.0, 0.0, 0.0, 1.0],
                [10.0, 0.0, 0.0, 1.0],
                [11.0, 0.0, 0.0, 1.0],
                [12.0, 0.0, 0.0, 1.0],
                [13.0, 0.0, 0.0, 1.0],
                [14.0, 0.0, 0.0, 1.0],
                [15.0, 0.0, 0.0, 1.0],
            ]),
            'left_vec': np.array([1.0, 0.0, 0.0]),
            'right_vec': np.array([1.0, 0.0, 0.0]),
            'is_left_input': True,
            'is_right_input': False,
            'is_left_cut': True,
            'is_right_cut': True,
        },
        'path2': {
            'name': 'path2',
            'path': np.array([
                [8.0, 0.0, 0.0, 1.0], 
                [8.0, 1.0, 0.0, 1.0],
                [8.0, 2.0, 0.0, 1.0],
                [8.0, 3.0, 0.0, 1.0],
                [8.0, 4.0, 0.0, 1.0],
                [8.0, 5.0, 0.0, 1.0],
                [8.0, 6.0, 0.0, 1.0],
                [8.0, 7.0, 0.0, 1.0],
                [8.0, 8.0, 0.0, 1.0],
            ]),
            'right_vec': np.array([0.0, 1.0, 0.0]),
            'is_right_input': False,
            'is_right_cut': True,
        }
    }

    mesh = PathSurfaceGenerator.generate_surface(paths, path_reso, point_reso)


if __name__ == '__main__':
    test_generate_surface()

