import numpy as np
import pandas as pd
import pyvista
import math
from sklearn.neighbors import KDTree
from typing import Dict, Union

from scripts_py.version_9.mapf_pkg.shape_utils import ShapeUtils, AxisTransform
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils


class Pcd2MeshUtils(object):
    @staticmethod
    def pcd2mesh_by_meshlab(pcd_file: str, output_cfg: Dict, sample_num=-1, threshold: float = 0.8):
        import pymeshlab
        from pymeshlab import pmeshlab

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(pcd_file)
        ms.apply_filter('compute_normal_for_point_clouds')
        ms.apply_filter('generate_surface_reconstruction_screened_poisson')

        if sample_num > 0:
            ms.apply_filter('generate_sampling_poisson_disk', samplenum=sample_num)
            ms.apply_filter('generate_surface_reconstruction_screened_poisson')

        if threshold > 0.0:
            ms.apply_filter('meshing_decimation_clustering', threshold=pmeshlab.PercentageValue(threshold))

        ms.save_current_mesh(output_cfg['file'], **output_cfg['cfg'])

    @staticmethod
    def pcd2mesh_by_vtk(pcd: np.ndarray, alpha=0.0, tol=0.001, offset=2.5) -> pyvista.PolyData:
        # cloud = pyvista.PolyData(pcd)
        # return cloud.delaunay_3d(
        #     alpha=alpha,
        #     tol=tol,
        #     offset=offset,
        #     progress_bar=True
        # ).extract_geometry()
        raise NotImplementedError

    @staticmethod
    def save_ply(mesh: pyvista.PolyData, file: str):
        assert file.endswith('.ply')
        mesh.save(file)


class PathSurfaceGenerator(object):
    def __init__(
            self, path_xyz: np.ndarray, radius: np.ndarray,
            left_dire: np.ndarray = None, right_dire: np.ndarray = None
    ):
        self.path_xyz = path_xyz
        self.radius = radius
        self.left_dire = left_dire
        self.right_dire = right_dire

        self.pcd_data: np.ndarray = None
        self.left_clamp_pcd: np.ndarray = None
        self.right_clamp_pcd: np.ndarray = None
        self.sphere_centers: np.ndarray = None
        self.sphere_radius: np.ndarray = None

    def generate_pcd_of_sphere(self, length_reso=0.2, sphere_reso=0.15, relax_factor=0.01, deplicate_format=3):
        pcd_data, sphere_centers, sphere_radius = [], [], []
        num = self.path_xyz.shape[0]
        for i in range(num - 1):
            left_xyz, right_xyz = self.path_xyz[i, :], self.path_xyz[i + 1, :]
            left_radius, right_radius = self.radius[i], self.radius[i + 1]
            vec = right_xyz[:3] - left_xyz[:3]
            length = np.linalg.norm(vec, ord=2)
            norm_vec = vec / length
            step_num = math.ceil(length / length_reso)
            for step_length in np.linspace(0.0, length, step_num):
                center = left_xyz + norm_vec * step_length
                radius = left_radius + (right_radius - left_radius) * step_length / length
                pcd = ShapeUtils.create_sphere_pcd(center, norm_vec, radius, sphere_reso)

                pcd_data.append(pcd)
                sphere_centers.append(center.reshape((1, -1)))
                sphere_radius.append(radius)

        pcd_data = np.concatenate(pcd_data, axis=0)
        self.sphere_centers = np.concatenate(sphere_centers, axis=0)
        self.sphere_radius = np.array(sphere_radius)
        pcd_data = self.remove_inner_sphere(pcd_data, self.sphere_centers, self.sphere_radius, relax_factor)

        pcd_data = np.round(pcd_data, decimals=deplicate_format)
        pcd_data = pd.DataFrame(data=pcd_data, columns=['x', 'y', 'z'])
        pcd_data.drop_duplicates(subset=['x', 'y', 'z'], inplace=True)
        pcd_data = pcd_data[['x', 'y', 'z']].values
        self.pcd_data = pcd_data

    @staticmethod
    def remove_inner_sphere(pcd_data: np.ndarray, spheres_xyz: np.ndarray, radius: np.ndarray, relax_factor=0.15):
        valid_bool = np.full(shape=(pcd_data.shape[0],), fill_value=True, dtype=bool)
        pcd_tree = KDTree(pcd_data)
        idxs_list = pcd_tree.query_radius(spheres_xyz, r=radius - relax_factor, return_distance=False)
        idxs_list = np.concatenate(idxs_list, axis=0)
        valid_bool[idxs_list] = False
        pcd_data = pcd_data[valid_bool]
        return pcd_data

    @staticmethod
    def generate_clamp_plane(radius: float, vec: np.ndarray, center: np.ndarray, reso=0.2):
        uvs, u_steps = [], math.ceil(radius / reso)
        for cell_radius in np.linspace(0, radius, u_steps):
            v_length = 2.0 * cell_radius * np.pi
            num = max(math.ceil(v_length / reso), 1)
            rads = np.deg2rad(np.linspace(0.0, 360.0, num))
            uv = np.zeros(shape=(num, 3))
            uv[:, 1] = np.cos(rads) * cell_radius
            uv[:, 2] = np.sin(rads) * cell_radius
            uvs.append(uv)
        uvs = np.concatenate(uvs, axis=0)
        norm_vec = vec / np.linalg.norm(vec)
        rot_mat = AxisTransform.rot_mat_from_vectors(norm_vec)
        pcd = AxisTransform.pcd_transform(uvs, rot_mat)
        pcd = pcd + center
        return pcd

    @staticmethod
    def cut_clamp(pcd: np.ndarray, radius: float, vec: np.ndarray, center: np.ndarray, is_invert, relax_factor=1e-2):
        pcd_tree = KDTree(pcd)
        idxs = pcd_tree.query_radius(center.reshape(1, -1), r=radius + relax_factor, return_distance=False)[0]
        inner_pcd = pcd[idxs, :]
        cen2pcd = inner_pcd - center
        norm_vec = vec / np.linalg.norm(vec)
        if is_invert:
            outer_idxs = idxs[np.sum(cen2pcd * norm_vec, axis=1) >= 0.0]
        else:
            outer_idxs = idxs[np.sum(cen2pcd * norm_vec, axis=1) <= 0.0]
        valid_bool = np.ones(pcd.shape[0], dtype=bool)
        valid_bool[outer_idxs] = False
        pcd = pcd[valid_bool]
        return pcd

    def generate_pcd(self, cfg: Dict):
        self.generate_pcd_of_sphere(
            cfg['length_reso'], cfg['sphere_reso'], cfg['relax_factor'], cfg['deplicate_format']
        )
        if cfg['with_left_clamp']:
            self.left_clamp_pcd = self.generate_clamp_plane(
                self.radius[0], self.left_dire, self.path_xyz[0, :], cfg['left_clamp_reso']
            )
            self.pcd_data = self.cut_clamp(
                self.pcd_data, self.radius[0], self.left_dire, self.path_xyz[0, :],
                cfg['is_left_clamp_invert'], cfg['left_clamp_relax_factor']
            )
        if cfg['with_right_clamp']:
            self.right_clamp_pcd = self.generate_clamp_plane(
                self.radius[-1], self.right_dire, self.path_xyz[-1, :], cfg['right_clamp_reso']
            )
            self.pcd_data = self.cut_clamp(
                self.pcd_data, self.radius[-1], self.right_dire, self.path_xyz[-1, :],
                cfg['is_right_clamp_invert'], cfg['right_clamp_relax_factor']
            )

    def merge_pcd(self) -> np.ndarray:
        pcd_list = [self.pcd_data]
        if self.left_clamp_pcd is not None:
            pcd_list.append(self.left_clamp_pcd)
        if self.right_clamp_pcd is not None:
            pcd_list.append(self.right_clamp_pcd)

        if len(pcd_list) == 1:
            return pcd_list[0]
        else:
            return np.concatenate(pcd_list, axis=0)

    def draw(self, with_path=True, with_pcd=True, vis: VisUtils = None):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        if with_path:
            line_set = VisUtils.create_line_set(np.arange(0, self.path_xyz.shape[0], 1))
            line_mesh = VisUtils.create_line(self.path_xyz, line_set)
            vis.plot(line_mesh)

        if with_pcd:
            pcd_mesh = VisUtils.create_point_cloud(self.pcd_data)
            vis.plot(pcd_mesh, color=(0.5, 0.5, 0.5))

            if self.left_clamp_pcd is not None:
                left_mesh = VisUtils.create_point_cloud(self.left_clamp_pcd)
                vis.plot(left_mesh, color=(1.0, 0.0, 0.0))

            if self.right_clamp_pcd is not None:
                right_mesh = VisUtils.create_point_cloud(self.right_clamp_pcd)
                vis.plot(right_mesh, color=(0.0, 1.0, 0.0))

        if show_plot:
            vis.show()


