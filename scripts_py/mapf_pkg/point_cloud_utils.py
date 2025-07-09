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
from CGAL.CGAL_Advancing_front_surface_reconstruction import (
    advancing_front_surface_reconstruction,
)
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3

from CGAL.CGAL_Point_set_processing_3 import (
    grid_simplify_point_set,
    compute_average_spacing,
)

import pymeshlab
from pymeshlab import pmeshlab


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
        grid = pyvista.UnstructuredGrid(
            {pyvista.CellType.TRIANGLE: connectivity}, points
        )
        return grid


class PathSurfaceGenerator(object):

    class PathSurfaceItem(object):
        def __init__(
            self,
            name: str,
            path: np.ndarray,
            refine_reso: float = 0.1,
            left_vec: np.ndarray = None,
            right_vec: np.ndarray = None,
            is_left_input: bool = False,
            is_right_input: bool = False,
            is_left_cut: bool = False,
            is_right_cut: bool = False,
        ):
            self.name = name

            max_seg_dist = np.max(np.linalg.norm(path[1:, :3] - path[:-1, :3], axis=1))
            if max_seg_dist > refine_reso:
                self.refined_path: np.ndarray = PathSurfaceGenerator.refine_path(
                    path, refine_reso
                )
            else:
                self.refined_path = path
            
            # self.path = path
            self.path = self.refined_path

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
    def generate_surface_points_of_sphere(
        center: np.ndarray, radius: float, vector: np.ndarray, reso: float
    ):
        """
        sample points on the surface of the sphere based on Spherical coordinate system and rotation it to align with the vector
        """
        xs_nums = math.ceil(np.pi * radius / reso)
        rads = np.linspace(0, np.pi, xs_nums)

        points_list = []
        for rad in rads:
            x = radius * np.cos(rad)
            sub_radius = np.sqrt(radius**2 - x**2)
            cross_section_perimeter = sub_radius * 2 * np.pi
            cross_section_nums = max(math.ceil(cross_section_perimeter / reso), 1)
            theta = np.linspace(0, 2 * np.pi, cross_section_nums)
            points = np.zeros(shape=(cross_section_nums, 3))
            points[:, 0] = x
            points[:, 1] = sub_radius * np.cos(theta)
            points[:, 2] = sub_radius * np.sin(theta)
            points = AxisTransform.pcd_transform(
                points, AxisTransform.rot_mat_from_vectors(vector)
            )
            points = points + center
            points_list.append(points)
        points = np.concatenate(points_list, axis=0)
        return points

    @staticmethod
    def generate_surface_points_of_disk(
        center: np.ndarray, radius: float, vector: np.ndarray, reso: float
    ):
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
            vec = path[i + 1] - path[i]
            length = np.linalg.norm(vec)
            norm_vec = vec / length
            if length > reso:
                step_num = math.ceil(length / reso)
                for step in range(step_num):
                    path_i = path[i].copy()
                    path_i = path_i + norm_vec * step * length / step_num
                    new_path.append(path_i)
            else:
                new_path.append(path[i])
        return np.array(new_path)

    @staticmethod
    def remove_inner_points_from_path(
        points: np.ndarray, path: np.ndarray, relax_factor=0.15
    ):
        """
        points: (n, 3), path: (n, 4) means (x, y, z, radius)
        """
        valid_bool = np.full(shape=(points.shape[0],), fill_value=True, dtype=bool)
        points_tree = KDTree(points)

        idxs_list = points_tree.query_radius(
            path[:, :3], r=path[:, 3] - relax_factor, return_distance=False
        )
        for path_idx, point_idxs in enumerate(idxs_list):
            if len(point_idxs) > 0:
                valid_bool[point_idxs] = False
        points = points[valid_bool]
        return points

    @staticmethod
    def cut_clamp(
        points: np.ndarray,
        center: np.ndarray,
        radius: float,
        vec: np.ndarray,
        is_input: bool,
        relax_factor: float = 0.15,
    ):
        """
        points: (n, 3), center: (3,), radius: float, vec: (3,), is_input: bool
        """
        tree = KDTree(points)
        idxs = tree.query_radius(
            center.reshape(1, -1), r=radius + relax_factor, return_distance=False
        )[0]

        is_in_range_bool = np.full(
            shape=(points.shape[0],), fill_value=False, dtype=bool
        )
        is_in_range_bool[idxs] = True

        cen2pcd = points - center
        norm_vec = vec / np.linalg.norm(vec)
        dist_list = np.sum(cen2pcd * norm_vec, axis=1)
        if is_input:
            is_cut_bool = np.full(
                shape=(points.shape[0],), fill_value=False, dtype=bool
            )
            is_cut_bool[dist_list < 0.0] = True
        else:
            is_cut_bool = np.full(
                shape=(points.shape[0],), fill_value=False, dtype=bool
            )
            is_cut_bool[dist_list > 0.0] = True
        is_in_range_bool = is_in_range_bool & is_cut_bool
        return points[~is_in_range_bool]

    @staticmethod
    def generate_surface_points_of_path(
        path: np.ndarray,
        reso: float,
        inlet_vec: np.ndarray = None,
        outlet_vec: np.ndarray = None,
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
            vec_list = np.concatenate(
                [vec_list, vec_list[-1, :].reshape(1, -1)], axis=0
            )
        
        points = []
        for point, vec in zip(path, vec_list):
            sphere_pcd = PathSurfaceGenerator.generate_surface_points_of_sphere(
                point[:3], point[3], vec, reso
            )
            points.append(sphere_pcd)
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
                path_item.points = np.concatenate(
                    [
                        path_item.points,
                        PathSurfaceGenerator.generate_surface_points_of_disk(
                            path_item.path[0, :3],
                            path_item.path[0, 3],
                            path_item.left_vec,
                            reso,
                        ),
                    ],
                    axis=0,
                )

            if path_item.is_right_cut:
                path_item.points = np.concatenate(
                    [
                        path_item.points,
                        PathSurfaceGenerator.generate_surface_points_of_disk(
                            path_item.path[-1, :3],
                            path_item.path[-1, 3],
                            path_item.right_vec,
                            reso,
                        ),
                    ],
                    axis=0,
                )

        for _, path_item in paths.items():
            for _, other_path_item in paths.items():
                if other_path_item.is_left_cut:
                    if other_path_item.left_vec is None:
                        left_vec = (
                            other_path_item.path[1, :3] - other_path_item.path[0, :3]
                        )
                        is_left_input = True
                    else:
                        left_vec = other_path_item.left_vec
                        is_left_input = other_path_item.is_left_input

                    path_item.points = PathSurfaceGenerator.cut_clamp(
                        path_item.points,
                        other_path_item.path[0, :3],
                        other_path_item.path[0, 3],
                        left_vec,
                        is_left_input,
                        relax_factor,
                    )

                if other_path_item.is_right_cut:
                    if other_path_item.right_vec is None:
                        right_vec = (
                            other_path_item.path[-1, :3] - other_path_item.path[-2, :3]
                        )
                        is_right_input = False
                    else:
                        right_vec = other_path_item.right_vec
                        is_right_input = other_path_item.is_right_input

                    path_item.points = PathSurfaceGenerator.cut_clamp(
                        path_item.points,
                        other_path_item.path[-1, :3],
                        other_path_item.path[-1, 3],
                        right_vec,
                        is_right_input,
                        relax_factor,
                    )

        return paths

    @staticmethod
    def merge_too_close_points(
        points: np.ndarray,
        voxel_size: float,
        method: Literal["mean", "median"] = "mean",
    ):
        points_df = pd.DataFrame(points, columns=["x", "y", "z"])
        points_df["x_int"] = (points_df["x"] / voxel_size).astype(int)
        points_df["y_int"] = (points_df["y"] / voxel_size).astype(int)
        points_df["z_int"] = (points_df["z"] / voxel_size).astype(int)

        points_df["key"] = (
            points_df["x_int"].astype(str)
            + "_"
            + points_df["y_int"].astype(str)
            + "_"
            + points_df["z_int"].astype(str)
        )
        if method == "mean":
            points_df = points_df.groupby("key").mean()[["x", "y", "z"]]
        elif method == "median":
            points_df.drop_duplicates(subset=["key"], inplace=True)
            points_df["x"] = (points_df["x_int"] + 0.5) * voxel_size
            points_df["y"] = (points_df["y_int"] + 0.5) * voxel_size
            points_df["z"] = (points_df["z_int"] + 0.5) * voxel_size
        else:
            raise ValueError(f"Invalid method: {method}")

        points = points_df[["x", "y", "z"]].values
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
        paths: dict[str, np.ndarray | bool],
        refine_path_reso: float,
        point_reso: float,
        relax_factor: float = None,
        merge_reso: float = None,
        method: Literal["cgal", "meshlab"] = "cgal",
        save_pcd_file: str = None,
        save_stl_file: str = None,
    ) -> pyvista.UnstructuredGrid:
        if relax_factor is None:
            relax_factor = min(point_reso, refine_path_reso) * 0.25
        if merge_reso is None:
            merge_reso = point_reso * 0.85

        path_dict: dict[str, PathSurfaceGenerator.PathSurfaceItem] = {}
        for name, path_info in paths.items():
            path_dict[name] = PathSurfaceGenerator.PathSurfaceItem(
                name=name,
                path=path_info["path"],
                refine_reso=refine_path_reso,
                is_left_input=path_info.get("is_left_input", False),
                is_right_input=path_info.get("is_right_input", False),
                is_left_cut=path_info.get("is_left_cut", False),
                is_right_cut=path_info.get("is_right_cut", False),
                left_vec=path_info.get("left_vec", None),
                right_vec=path_info.get("right_vec", None),
            )

        path_dict = PathSurfaceGenerator.generate_surface_points_of_paths(
            path_dict, point_reso, relax_factor
        )
        # plotter = pyvista.Plotter()
        # for name, path_item in path_dict.items():
        #     plotter.add_mesh(path_item.to_mesh(), color=np.random.rand(3), point_size=10)
        # plotter.show()

        points = []
        for name, path_item in path_dict.items():
            points.append(path_item.points)
        points = np.concatenate(points, axis=0)
        merged_points = PathSurfaceGenerator.merge_too_close_points(points, merge_reso)
        print(f"generate_surface.merged_points: {merged_points.shape}")

        # plotter = pyvista.Plotter()
        # # plotter.add_mesh(pyvista.PointSet(points), color='blue', point_size=8)
        # plotter.add_mesh(pyvista.PointSet(merged_points), color='red', point_size=8)
        # plotter.show()

        if method == "cgal":
            mesh = CGALUtils.poisson_surface_reconstruction(merged_points)
            # plotter = pyvista.Plotter()
            # plotter.add_mesh(mesh, color='red', style='wireframe')
            # # plotter.add_mesh(pyvista.PointSet(points), color='blue', point_size=8)
            # plotter.show()
        elif method == "meshlab":
            pyvista.PolyData(merged_points).save(save_pcd_file)
            PathSurfaceGenerator.pcd2mesh_by_meshlab(
                save_pcd_file, save_stl_file, sample_num=-1, threshold=0.8
            )

        # return mesh

    @staticmethod
    def pcd2mesh_by_meshlab(pcd_file: str, save_file: str, sample_num=-1, threshold: float = 0.8):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(pcd_file)
        ms.apply_filter('compute_normal_for_point_clouds')
        ms.apply_filter('generate_surface_reconstruction_screened_poisson')

        if sample_num > 0:
            ms.apply_filter('generate_sampling_poisson_disk', samplenum=sample_num)
            ms.apply_filter('generate_surface_reconstruction_screened_poisson')

        if threshold > 0.0:
            ms.apply_filter('meshing_decimation_clustering', threshold=pmeshlab.PercentageValue(threshold))

        ms.save_current_mesh(save_file)


def test_generate_surface():
    paths = {
        "path1": {
            "name": "path1",
            "path": np.array(
                [
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
                ]
            ),
            "left_vec": np.array([1.0, 0.0, 0.0]),
            "right_vec": np.array([1.0, 0.0, 0.0]),
            "is_left_input": True,
            "is_right_input": False,
            "is_left_cut": True,
            "is_right_cut": True,
        },
        "path2": {
            "name": "path2",
            "path": np.array(
                [
                    [8.0, 0.0, 0.0, 1.0],
                    [8.0, 1.0, 0.0, 1.0],
                    [8.0, 2.0, 0.0, 1.0],
                    [8.0, 3.0, 0.0, 1.0],
                    [8.0, 4.0, 0.0, 1.0],
                    [8.0, 5.0, 0.0, 1.0],
                    [8.0, 6.0, 0.0, 1.0],
                    [8.0, 7.0, 0.0, 1.0],
                    [8.0, 8.0, 0.0, 1.0],
                ]
            ),
            "right_vec": np.array([0.0, 1.0, 0.0]),
            "is_right_input": False,
            "is_right_cut": True,
        },
    }

    # mesh = PathSurfaceGenerator.generate_surface(paths, refine_path_reso=0.1, point_reso=0.1)
    # mesh.plot(style="wireframe")
    # pyvista.save_meshio("/home/admin123456/Desktop/work/test/surface.stl", mesh)

    mesh = PathSurfaceGenerator.generate_surface(
        paths, refine_path_reso=0.5, point_reso=0.1,
        method="meshlab",
        save_pcd_file="/home/admin123456/Desktop/work/test/surface.ply",
        save_stl_file="/home/admin123456/Desktop/work/test/surface.stl"
    )


def test_generate_surface2():
    paths = {
        "path1": {
            "name": "path1",
            "path": np.array(
                [
                    [0.0, 10.0, 10.0, 1.0],
                    [0.5234366, 10.04705617, 9.95987696, 1.0],
                    [0.87980304, 10.08051623, 9.91442354, 1.0],
                    [1.14739, 10.11515237, 9.87336266, 1.0],
                    [1.37853283, 10.15813953, 9.83980449, 1.0],
                    [1.59137966, 10.20664589, 9.80814916, 1.0],
                    [1.80045905, 10.25723196, 9.77275845, 1.0],
                    [2.01046134, 10.30703295, 9.73224487, 1.0],
                    [2.22153347, 10.35345675, 9.6871802, 1.0],
                    [2.43325357, 10.39585866, 9.63740377, 1.0],
                    [2.64392987, 10.43813291, 9.58102179, 1.0],
                    [2.85178406, 10.48473795, 9.51612297, 1.0],
                    [3.05657106, 10.53880196, 9.44452683, 1.0],
                    [3.25929324, 10.6023708, 9.37108823, 1.0],
                    [3.46044136, 10.67652738, 9.29912579, 1.0],
                    [3.65813492, 10.75832211, 9.22492684, 1.0],
                    [3.84988352, 10.8437748, 9.14295544, 1.0],
                    [4.03468693, 10.93174964, 9.05139991, 1.0],
                    [4.21364902, 11.02512698, 8.95370705, 1.0],
                    [4.3878646, 11.12678758, 8.85330154, 1.0],
                    [4.55702902, 11.23708803, 8.75011467, 1.0],
                    [4.72014762, 11.35512706, 8.64234733, 1.0],
                    [4.87815146, 11.47824595, 8.52988486, 1.0],
                    [5.03682413, 11.59939167, 8.41687761, 1.0],
                    [5.20232362, 11.71122133, 8.30780958, 1.0],
                    [5.37397444, 11.81347358, 8.20124221, 1.0],
                    [5.54514497, 11.9120594, 8.09057475, 1.0],
                    [5.71034669, 12.01250655, 7.97098085, 1.0],
                    [5.87038131, 12.11757877, 7.84675769, 1.0],
                    [6.02790188, 12.22917326, 7.7248297, 1.0],
                    [6.18616389, 12.3459331, 7.60972308, 1.0],
                    [6.34933463, 12.46157426, 7.50233228, 1.0],
                    [6.52087611, 12.57019899, 7.40270695, 1.0],
                    [6.6971438, 12.67315751, 7.3059645, 1.0],
                    [6.87070448, 12.77566429, 7.20459389, 1.0],
                    [7.03711471, 12.88002904, 7.09382535, 1.0],
                    [7.20013175, 12.98058915, 6.97638672, 1.0],
                    [7.36477959, 13.07069544, 6.8563081, 1.0],
                    [7.53385248, 13.15085872, 6.73851914, 1.0],
                    [7.70806282, 13.22827473, 6.62878938, 1.0],
                    [7.88782053, 13.30852292, 6.53142746, 1.0],
                    [8.07240531, 13.38768408, 6.44316735, 1.0],
                    [8.26074666, 13.4588976, 6.35840686, 1.0],
                    [8.45227529, 13.51964531, 6.27565088, 1.0],
                    [8.64723642, 13.57446537, 6.2000781, 1.0],
                    [8.84588484, 13.62804982, 6.13692679, 1.0],
                    [9.04782813, 13.68018785, 6.08594326, 1.0],
                    [9.25230258, 13.72785847, 6.04372532, 1.0],
                    [9.45839707, 13.76933591, 6.00747944, 1.0],
                    [9.66478193, 13.8067816, 5.97630235, 1.0],
                    [9.87010208, 13.84284885, 5.94973562, 1.0],
                    [10.07416934, 13.8752382, 5.929643, 1.0],
                    [10.27796244, 13.89669743, 5.92021036, 1.0],
                    [10.48264002, 13.9025366, 5.92398714, 1.0],
                    [10.69007087, 13.90333285, 5.93397318, 1.0],
                    [10.9023404, 13.91475505, 5.9399783, 1.0],
                    [11.11933555, 13.9418423, 5.93803049, 1.0],
                    [11.33710875, 13.9710876, 5.93500699, 1.0],
                    [11.55151469, 13.98831235, 5.93820163, 1.0],
                    [11.76045516, 13.99234682, 5.94755646, 1.0],
                    [11.96309257, 13.99002849, 5.95848836, 1.0],
                    [12.15988026, 13.98710065, 5.96694501, 1.0],
                    [12.35529381, 13.98547764, 5.97076123, 1.0],
                    [12.55444019, 13.98624583, 5.96829243, 1.0],
                    [12.75820964, 13.98858034, 5.96177439, 1.0],
                    [12.96297602, 13.98960918, 5.95759908, 1.0],
                    [13.16533476, 13.98714409, 5.96099117, 1.0],
                    [13.36452186, 13.98366983, 5.969004, 1.0],
                    [13.56074773, 13.98337175, 5.97570804, 1.0],
                    [13.75425033, 13.98794894, 5.97853759, 1.0],
                    [13.94531635, 13.99443763, 5.9812376, 1.0],
                    [14.13428943, 13.99967131, 5.9878194, 1.0],
                    [14.32213232, 14.00420252, 5.99716476, 1.0],
                    [14.51021661, 14.01103998, 6.00476786, 1.0],
                    [14.69957862, 14.02305467, 6.00736098, 1.0],
                    [14.89008406, 14.04249803, 6.00611262, 1.0],
                    [15.08135019, 14.07139705, 6.00313098, 1.0],
                    [15.27303459, 14.10902491, 6.00030934, 1.0],
                    [15.46484122, 14.15149533, 5.99929451, 1.0],
                    [15.65636964, 14.1951922, 6.00120172, 1.0],
                    [15.84652813, 14.23937082, 6.00379631, 1.0],
                    [16.03396079, 14.28443011, 6.00352628, 1.0],
                    [16.21882123, 14.33092655, 5.99898268, 1.0],
                    [16.40430424, 14.37972816, 5.99307917, 1.0],
                    [16.59371103, 14.43162696, 5.98910525, 1.0],
                    [16.78691856, 14.48591379, 5.98845774, 1.0],
                    [16.98137861, 14.54081633, 5.99119292, 1.0],
                    [17.17519984, 14.59440253, 5.99682823, 1.0],
                    [17.36910879, 14.64423318, 6.00299399, 1.0],
                    [17.5644107, 14.68787632, 6.00692999, 1.0],
                    [17.76151228, 14.7271859, 6.00743936, 1.0],
                    [17.95971582, 14.76928386, 6.00524678, 1.0],
                    [18.15834076, 14.8200864, 6.00132876, 1.0],
                    [18.35730355, 14.87476365, 5.99760197, 1.0],
                    [18.55679448, 14.92389673, 5.99637757, 1.0],
                    [18.76033312, 14.96143397, 5.99835631, 1.0],
                    [18.97864944, 14.98865648, 6.0007469, 1.0],
                    [19.22543543, 15.00741776, 6.00057831, 1.0],
                    [19.5439711, 15.01390428, 5.99893912, 1.0],
                    [20.0, 15.0, 6.0, 1.0],
                ]
            ),
            "left_vec": np.array([1.0, 0.0, 0.0]),
            "right_vec": np.array([1.0, 0.0, 0.0]),
            "is_left_input": False,
            "is_right_input": False,
            "is_left_cut": False,
            "is_right_cut": False,
        }
    }

    PathSurfaceGenerator.generate_surface(
        paths, 
        refine_path_reso=0.2, 
        point_reso=0.1, 
        method="meshlab",
        save_pcd_file="/home/admin123456/Desktop/work/test/surface.ply",
        save_stl_file="/home/admin123456/Desktop/work/test/surface.stl"
    )
    # # mesh.plot(style="wireframe")
    # pyvista.save_meshio("/home/admin123456/Desktop/work/test/surface.stl", mesh)

    


if __name__ == "__main__":
    test_generate_surface2()
