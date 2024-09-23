import numpy as np
import pandas as pd
import pyvista
import math
from sklearn.neighbors import KDTree
from typing import Dict, Union
from tqdm import tqdm

from scripts_py.version_9.mapf_pkg.shape_utils import ShapeUtils, AxisTransform
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils


class Pcd2MeshUtils(object):
    @staticmethod
    def pcd2mesh(pcd_file: str, sample_num: int, output_cfg: Dict):
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(pcd_file)
        ms.apply_filter('compute_normal_for_point_clouds')
        ms.apply_filter('generate_surface_reconstruction_screened_poisson')
        ms.apply_filter('generate_sampling_poisson_disk', samplenum=sample_num)
        ms.apply_filter('generate_surface_reconstruction_screened_poisson')
        ms.save_current_mesh(output_cfg['file'], **output_cfg['cfg'])


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

    def generate_pcd_by_sphere(
            self, length_reso: float, sphere_reso: float, relax_factor: float, deplicate_format: int = 3
    ):
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
    def remove_inner_sphere(
            pcd_data: np.ndarray, spheres_xyz: np.ndarray, sphere_radius: np.ndarray, relax_factor: float
    ):
        valid_bool = np.full(shape=(pcd_data.shape[0],), fill_value=True, dtype=bool)
        pcd_tree = KDTree(pcd_data)
        idxs_list = pcd_tree.query_radius(
            spheres_xyz, r=sphere_radius - relax_factor, return_distance=False
        )
        idxs_list = np.concatenate(idxs_list, axis=0)
        valid_bool[idxs_list] = False
        pcd_data = pcd_data[valid_bool]
        return pcd_data

    @staticmethod
    def generate_clamp(radius: float, vec: np.ndarray, center: np.ndarray, reso: float):
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
    def cut_clamp(pcd: np.ndarray, radius: float, vec: np.ndarray, center: np.ndarray, is_invert, relax_factor=1e-1):
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
        self.generate_pcd_by_sphere(
            cfg['length_reso'], cfg['sphere_reso'], cfg['relax_factor'], cfg['deplicate_format']
        )
        if cfg['with_left_clamp']:
            self.left_clamp_pcd = self.generate_clamp(
                self.radius[0], self.left_dire, self.path_xyz[0, :], cfg['left_clamp_reso']
            )
            self.pcd_data = self.cut_clamp(
                self.pcd_data, self.radius[0], self.left_dire, self.path_xyz[0, :],
                cfg['is_left_clamp_invert'], cfg['left_clamp_relax_factor']
            )
        if cfg['with_right_clamp']:
            self.right_clamp_pcd = self.generate_clamp(
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


"""
class SegmentSurfaceCell(object):
    def __init__(
            self, path: np.ndarray, radius: float,
            left_direction: np.ndarray = None, right_direction: np.ndarray = None
    ):
        self.path = path
        self.radius = radius
        self.left_direction = left_direction
        self.right_direction = right_direction

        self.pcd_data: np.ndarray = None
        self.left_clamp_pcd: np.ndarray = None
        self.right_clamp_pcd: np.ndarray = None

    @staticmethod
    def pcd2line_dist(pcd: np.ndarray, line_0: np.ndarray, line_1: np.ndarray):
        line02pcd = pcd - line_0
        line12pcd = pcd - line_1
        line02line1 = line_1 - line_0

        line02pcd_dist = np.linalg.norm(line02pcd, ord=2, axis=1)
        line12pcd_dist = np.linalg.norm(line12pcd, ord=2, axis=1)
        line02line1_dist = np.linalg.norm(line02line1, ord=2)

        cos_dist = np.sum(line02pcd * line02line1, axis=1) / line02line1_dist

        res_dist = np.zeros(shape=(pcd.shape[0],))

        left_clamp_bool = cos_dist < 0
        res_dist[left_clamp_bool] = line02pcd_dist[left_clamp_bool]

        right_clamp_bool = cos_dist > line02line1_dist
        res_dist[right_clamp_bool] = line12pcd_dist[right_clamp_bool]

        inner_bool = np.bitwise_and(cos_dist >= 0., cos_dist <= line02line1_dist)
        res_dist[inner_bool] = np.sqrt(np.power(line02pcd_dist[inner_bool], 2.0) - np.power(cos_dist[inner_bool], 2.0))

        return res_dist

    @staticmethod
    def remove_inner_pcd(pcd_data: np.ndarray, path: np.ndarray, radius, relax_factor: float):
        pcd_tree = KDTree(pcd_data)
        valid_bool = np.full(shape=(pcd_data.shape[0],), fill_value=True, dtype=bool)
        for i in range(path.shape[0] - 1):
            l0, l1 = path[i, :], path[i + 1, :]
            line_length = np.linalg.norm(l1 - l0, ord=2)
            search_radius = np.sqrt(np.power(line_length * 0.5, 2.0) + np.power(radius, 2.0)) + 1e-6
            search_line_points = np.concatenate([l0.reshape((1, -1)), l1.reshape((1, -1))], axis=0)
            idxs_list, dist_list = pcd_tree.query_radius(search_line_points, r=search_radius, return_distance=True)

            for pcd_idxs, dists in zip(idxs_list, dist_list):
                if len(pcd_idxs) == 0:
                    continue

                real_dist = SegmentSurfaceCell.pcd2line_dist(pcd_data[pcd_idxs, :], l0, l1)
                fail_pcd_idxs = pcd_idxs[real_dist < radius - relax_factor]
                valid_bool[fail_pcd_idxs] = False

        pcd_data = pcd_data[valid_bool]
        return pcd_data

    def generate_pcd_by_sphere(
            self, length_reso: float, sphere_reso: float, relax_factor: float, deplicate_format: int = 4
    ):
        pcd_data = []
        for i in range(self.path.shape[0] - 1):
            last_xyz = self.path[i, :]
            cur_xyz = self.path[i + 1, :]

            xyz_dif = cur_xyz[:3] - last_xyz[:3]
            length = np.linalg.norm(xyz_dif, ord=2)
            norm_vec = xyz_dif / length
            step = math.ceil(length / length_reso)
            step_length = length / step

            for j in range(step + 1):
                center = last_xyz + j * norm_vec * step_length
                pcd = ShapeUtils.create_sphere_pcd(center, norm_vec, self.radius, sphere_reso)
                pcd_data.append(pcd)

        pcd_data = np.concatenate(pcd_data, axis=0)

        pcd_data = np.round(pcd_data, decimals=deplicate_format)
        pcd_data = pd.DataFrame(data=pcd_data, columns=['x', 'y', 'z'])
        pcd_data.drop_duplicates(subset=['x', 'y', 'z'], inplace=True)
        pcd_data = pcd_data[['x', 'y', 'z']].values

        pcd_data = self.remove_inner_pcd(pcd_data, self.path, self.radius, relax_factor)

        self.pcd_data = pcd_data

    @staticmethod
    def generate_clamp(radius: float, direction: np.ndarray, center: np.ndarray, reso: float):
        uvs = []
        u_steps = math.ceil(radius / reso)
        for cell_radius in np.linspace(0, radius, u_steps):
            length = 2 * cell_radius * np.pi
            num = max(math.ceil(length / reso), 1)
            rads = np.deg2rad(np.linspace(0, 360.0, num))
            uv = np.zeros(shape=(num, 2))
            uv[:, 0] = np.cos(rads) * cell_radius
            uv[:, 1] = np.sin(rads) * cell_radius
            uvs.append(uv)
        uvs = np.concatenate(uvs, axis=0)
        pcd = np.concatenate([np.zeros((uvs.shape[0], 1)), uvs[:, 0:1], uvs[:, 1:2]], axis=1)

        norm_vec = direction / np.linalg.norm(direction)
        rot_mat = AxisTransform.rot_mat_from_vectors(norm_vec)
        pcd = AxisTransform.pcd_transform(pcd, rot_mat)
        pcd = pcd + center
        return pcd

    def generate_left_clamp(self, reso):
        center = self.path[0, :]
        if self.left_direction is None:
            direction = self.path[1, :] - self.path[0, :]
        else:
            direction = self.left_direction
        return self.generate_clamp(self.radius, direction, center, reso)

    def generate_right_clamp(self, reso):
        center = self.path[-1, :]
        if self.right_direction is None:
            direction = self.path[-1, :] - self.path[-2, :]
        else:
            direction = self.right_direction
        return self.generate_clamp(self.radius, direction, center, reso)

    def draw(self, with_path=True, with_pcd=True, vis: VisUtils = None):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        if with_path:
            line_set = VisUtils.create_line_set(np.arange(0, self.path.shape[0], 1))
            line_mesh = VisUtils.create_line(self.path, line_set)
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
"""


class Path2MeshConverter(object):
    def __init__(self):
        self.segment_cell: Dict[str, Dict[str, Union[PathSurfaceGenerator, bool]]] = {}

    def add_segment(
            self, name: str,
            path: np.ndarray, radius: np.ndarray, cfg: Dict,
            left_dire: np.ndarray, right_dire: np.ndarray,
            is_left_clamp_invert, is_right_clamp_invert,
            with_left_clamp=False, with_right_clamp=False,
    ):
        cfg.update({
            'is_left_clamp_invert': is_left_clamp_invert,
            'is_right_clamp_invert': is_right_clamp_invert,
            'with_left_clamp': with_left_clamp,
            'with_right_clamp': with_right_clamp
        })
        self.segment_cell[name] = {
            'path_cell': PathSurfaceGenerator(path, radius, left_dire, right_dire), 'cfg': cfg
        }

    def generate_pcd_data(self):
        for name, info in tqdm(self.segment_cell.items()):
            info['path_cell'].generate_pcd(info['cfg'])

    def remove_inner_pcd(self):
        for _, info_i in tqdm(self.segment_cell.items()):
            cell_i: PathSurfaceGenerator = info_i['path_cell']
            for _, info_j in self.segment_cell.items():
                cell_j: PathSurfaceGenerator = info_j['path_cell']

                cell_i.pcd_data = PathSurfaceGenerator.remove_inner_sphere(
                    cell_i.pcd_data, cell_j.sphere_centers, cell_j.sphere_radius, relax_factor=1e-4,
                )
                # clamp can't be cut
                # if cell_i.left_clamp_pcd is not None:
                #     cell_i.left_clamp_pcd = PathSurfaceGenerator.remove_inner_sphere(
                #         cell_i.left_clamp_pcd, cell_j.sphere_centers, cell_j.sphere_radius, relax_factor=1e-4,
                #     )
                # if cell_i.right_clamp_pcd is not None:
                #     cell_i.right_clamp_pcd = PathSurfaceGenerator.remove_inner_sphere(
                #         cell_i.right_clamp_pcd, cell_j.sphere_centers, cell_j.sphere_radius, relax_factor=1e-4,
                #     )

    def get_pcd_data(self):
        pcds = []
        for name, info in self.segment_cell.items():
            cell = info['path_cell']
            pcds.append(cell.merge_pcd())
        pcds = np.concatenate(pcds, axis=0)
        return pcds

    @staticmethod
    def pcd2mesh(pcd: np.ndarray, alpha=0.0, tol=0.001, offset=2.5) -> pyvista.PolyData:
        cloud = pyvista.PolyData(pcd)
        return cloud.delaunay_3d(
            alpha=alpha,
            tol=tol,
            offset=offset,
            progress_bar=True
        ).extract_geometry()

    @staticmethod
    def save_ply(mesh: pyvista.PolyData, file: str):
        assert file.endswith('.ply')
        mesh.save(file)
